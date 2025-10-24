//network/core/connection_manager.rs
use std::collections::{HashMap, HashSet};
use std::net::{SocketAddr, IpAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore, oneshot};
use tokio::net::{TcpStream, TcpListener};
use tokio_tungstenite::{WebSocketStream, MaybeTlsStream, tungstenite::protocol::Message};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, error, instrument};
use metrics::{counter, gauge, histogram};
use thiserror::Error;
use rand::Rng;

// Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub handshake_timeout: Duration,
    pub health_check_interval: Duration,
    pub max_retries: u32,
    pub backoff_base: Duration,
    pub max_backoff: Duration,
    pub quality_threshold: f64,
    pub stability_threshold: f64,
    pub enable_encryption: bool,
    pub enable_compression: bool,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            max_connections: 250,
            connection_timeout: Duration::from_secs(30),
            handshake_timeout: Duration::from_secs(10),
            health_check_interval: Duration::from_secs(15),
            max_retries: 5,
            backoff_base: Duration::from_secs(1),
            max_backoff: Duration::from_secs(60),
            quality_threshold: 0.6,
            stability_threshold: 0.7,
            enable_encryption: true,
            enable_compression: true,
        }
    }
}

// Core data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: String,
    pub public_key: Vec<u8>,
    pub address: IpAddr,
    pub port: u16,
    pub protocol: ProtocolType,
    pub reputation: i32,
    pub latency: Duration,
    pub last_seen: Instant,
    pub connection_count: u32,
    pub failed_attempts: u32,
    pub capabilities: Vec<Capability>,
    pub version: String,
    pub stake_amount: u64,
    pub validator_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolType {
    Tcp,
    Udp,
    WebSocket,
    Quic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Capability {
    BlockPropagation,
    TransactionRelay,
    StateSync,
    Archive,
    Validator,
}

#[derive(Debug, Clone)]
pub struct ConnectionState {
    pub peer_info: PeerInfo,
    pub connection_id: String,
    pub protocol: ProtocolType,
    pub established_at: Instant,
    pub last_activity: Instant,
    pub stream: ConnectionStream,
    pub quality_score: f64,
    pub stability_score: f64,
    pub metrics: ConnectionMetrics,
    pub force_attraction: f64,
    pub force_repulsion: f64,
    pub net_force: f64,
    pub failure_count: u32,
}

#[derive(Debug, Clone)]
pub enum ConnectionStream {
    Tcp(Arc<RwLock<TcpStream>>),
    WebSocket(Arc<RwLock<WebSocketStream<MaybeTlsStream<TcpStream>>>>),
    Quic(Arc<quinn::Connection>),
}

#[derive(Debug, Clone)]
pub struct ConnectionMetrics {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub error_count: u64,
    pub success_rate: f64,
    pub average_latency: Duration,
    pub last_latency: Duration,
    pub message_rate: f64,
    pub health_score: f64,
}

impl Default for ConnectionMetrics {
    fn default() -> Self {
        Self {
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            error_count: 0,
            success_rate: 1.0,
            average_latency: Duration::from_millis(0),
            last_latency: Duration::from_millis(0),
            message_rate: 0.0,
            health_score: 1.0,
        }
    }
}

// Message types
#[derive(Debug)]
pub enum ConnectionMessage {
    Connect {
        peer: PeerInfo,
        retry_count: u32,
        result_tx: oneshot::Sender<Result<ConnectionState, ConnectionError>>,
    },
    Disconnect {
        peer_id: String,
        reason: DisconnectReason,
        result_tx: Option<oneshot::Sender<Result<(), ConnectionError>>>,
    },
    HealthCheck,
    ForceRecalculation,
    TopologyOptimization,
    UpdatePeerReputation {
        peer_id: String,
        delta: i32,
    },
    BroadcastMessage {
        message: Vec<u8>,
        exclude_peers: HashSet<String>,
    },
}

#[derive(Debug, Clone)]
pub enum DisconnectReason {
    Graceful,
    Timeout,
    PoorQuality,
    ForceEviction,
    ProtocolViolation,
    HealthCheckFailed,
}

// Error types
#[derive(Debug, Error)]
pub enum ConnectionError {
    #[error("Connection limit reached: {current}/{max}")]
    ConnectionLimitReached { current: usize, max: usize },
    #[error("Connection timeout after {duration:?}")]
    Timeout { duration: Duration },
    #[error("Handshake failed: {reason}")]
    HandshakeFailed { reason: String },
    #[error("Protocol error: {reason}")]
    ProtocolError { reason: String },
    #[error("Security violation: {reason}")]
    SecurityError { reason: String },
    #[error("Peer banned: {address}")]
    PeerBanned { address: IpAddr },
    #[error("IO error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },
    #[error("WebSocket error: {source}")]
    WebSocketError {
        #[from]
        source: tokio_tungstenite::tungstenite::Error,
    },
    #[error("QUIC error: {source}")]
    QuicError {
        #[from]
        source: quinn::ConnectionError,
    },
    #[error("Serialization error: {source}")]
    SerializationError {
        #[from]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

// Security manager trait
pub trait SecurityManager: Send + Sync {
    fn validate_peer(&self, peer: &PeerInfo) -> Result<(), ConnectionError>;
    fn perform_handshake(&self, stream: &mut ConnectionStream, peer: &PeerInfo) -> impl std::future::Future<Output = Result<(), ConnectionError>> + Send;
    fn encrypt_message(&self, message: &[u8]) -> Result<Vec<u8>, ConnectionError>;
    fn decrypt_message(&self, encrypted: &[u8]) -> Result<Vec<u8>, ConnectionError>;
}

// Main connection manager
pub struct ConnectionManager {
    config: ConnectionConfig,
    node_id: String,
    connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
    pending_connections: Arc<RwLock<HashMap<String, Instant>>>,
    connection_semaphore: Arc<Semaphore>,
    message_tx: mpsc::Sender<ConnectionMessage>,
    security_manager: Arc<dyn SecurityManager>,
    validator_scores: Arc<RwLock<HashMap<String, f64>>>,
    stake_distribution: Arc<RwLock<HashMap<String, u64>>>,
    network_entropy: Arc<RwLock<f64>>,
    is_running: Arc<atomic::AtomicBool>,
}

impl ConnectionManager {
    pub fn new(
        config: ConnectionConfig,
        node_id: String,
        security_manager: Arc<dyn SecurityManager>,
    ) -> Self {
        let (message_tx, message_rx) = mpsc::channel(1000);
        
        let manager = Self {
            config,
            node_id,
            connections: Arc::new(RwLock::new(HashMap::new())),
            pending_connections: Arc::new(RwLock::new(HashMap::new())),
            connection_semaphore: Arc::new(Semaphore::new(config.max_connections)),
            message_tx,
            security_manager,
            validator_scores: Arc::new(RwLock::new(HashMap::new())),
            stake_distribution: Arc::new(RwLock::new(HashMap::new())),
            network_entropy: Arc::new(RwLock::new(0.5)),
            is_running: Arc::new(atomic::AtomicBool::new(false)),
        };

        // Start worker tasks
        tokio::spawn(Self::connection_worker(
            manager.connections.clone(),
            manager.pending_connections.clone(),
            manager.connection_semaphore.clone(),
            manager.security_manager.clone(),
            manager.validator_scores.clone(),
            manager.stake_distribution.clone(),
            manager.network_entropy.clone(),
            message_rx,
            manager.is_running.clone(),
            manager.config.clone(),
        ));

        tokio::spawn(Self::health_monitor_worker(
            manager.connections.clone(),
            manager.is_running.clone(),
            manager.config.health_check_interval,
        ));

        tokio::spawn(Self::force_calculation_worker(
            manager.connections.clone(),
            manager.validator_scores.clone(),
            manager.stake_distribution.clone(),
            manager.network_entropy.clone(),
            manager.is_running.clone(),
        ));

        manager
    }

    pub async fn start(&self) -> Result<(), ConnectionError> {
        self.is_running.store(true, atomic::Ordering::SeqCst);
        info!("Connection manager started for node {}", self.node_id);
        Ok(())
    }

    pub async fn stop(&self) -> Result<(), ConnectionError> {
        self.is_running.store(false, atomic::Ordering::SeqCst);
        
        // Gracefully disconnect all peers
        let connections = self.connections.read().await;
        let mut disconnect_futures = Vec::new();
        
        for (peer_id, _) in connections.iter() {
            let message_tx = self.message_tx.clone();
            let peer_id = peer_id.clone();
            
            disconnect_futures.push(tokio::spawn(async move {
                let (result_tx, result_rx) = oneshot::channel();
                let message = ConnectionMessage::Disconnect {
                    peer_id,
                    reason: DisconnectReason::Graceful,
                    result_tx: Some(result_tx),
                };
                
                if message_tx.send(message).await.is_ok() {
                    let _ = result_rx.await;
                }
            }));
        }
        
        // Wait for all disconnections to complete
        futures::future::join_all(disconnect_futures).await;
        
        info!("Connection manager stopped for node {}", self.node_id);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn connect_to_peer(&self, peer: PeerInfo) -> Result<ConnectionState, ConnectionError> {
        // Check if already connected
        if self.is_connected(&peer.node_id).await {
            return self.get_connection_state(&peer.node_id).await
                .ok_or_else(|| ConnectionError::ProtocolError { reason: "Peer disconnected during check".to_string() });
        }

        let (result_tx, result_rx) = oneshot::channel();
        
        let message = ConnectionMessage::Connect {
            peer,
            retry_count: 0,
            result_tx,
        };

        self.message_tx.send(message).await
            .map_err(|_| ConnectionError::ProtocolError { reason: "Connection manager channel closed".to_string() })?;

        // Wait for connection result with timeout
        tokio::time::timeout(self.config.connection_timeout, result_rx)
            .await
            .map_err(|_| ConnectionError::Timeout { duration: self.config.connection_timeout })?
            .map_err(|_| ConnectionError::ProtocolError { reason: "Result channel closed".to_string() })?
    }

    #[instrument(skip(self))]
    pub async fn disconnect_peer(&self, peer_id: &str, reason: DisconnectReason) -> Result<(), ConnectionError> {
        let (result_tx, result_rx) = oneshot::channel();
        
        let message = ConnectionMessage::Disconnect {
            peer_id: peer_id.to_string(),
            reason,
            result_tx: Some(result_tx),
        };

        self.message_tx.send(message).await
            .map_err(|_| ConnectionError::ProtocolError { reason: "Connection manager channel closed".to_string() })?;

        result_rx.await
            .map_err(|_| ConnectionError::ProtocolError { reason: "Result channel closed".to_string() })?
    }

    pub async fn get_connection_count(&self) -> usize {
        self.connections.read().await.len()
    }

    pub async fn is_connected(&self, peer_id: &str) -> bool {
        self.connections.read().await.contains_key(peer_id)
    }

    pub async fn get_connection_state(&self, peer_id: &str) -> Option<ConnectionState> {
        self.connections.read().await.get(peer_id).cloned()
    }

    pub async fn get_active_peers(&self) -> Vec<PeerInfo> {
        self.connections.read().await.values()
            .map(|state| state.peer_info.clone())
            .collect()
    }

    pub async fn broadcast_message(&self, message: Vec<u8>, exclude_peers: HashSet<String>) -> Result<(), ConnectionError> {
        let msg = ConnectionMessage::BroadcastMessage {
            message,
            exclude_peers,
        };

        self.message_tx.send(msg).await
            .map_err(|_| ConnectionError::ProtocolError { reason: "Connection manager channel closed".to_string() })?;

        Ok(())
    }

    pub async fn update_peer_reputation(&self, peer_id: String, delta: i32) -> Result<(), ConnectionError> {
        let msg = ConnectionMessage::UpdatePeerReputation {
            peer_id,
            delta,
        };

        self.message_tx.send(msg).await
            .map_err(|_| ConnectionError::ProtocolError { reason: "Connection manager channel closed".to_string() })?;

        Ok(())
    }

    // Core worker implementation
    async fn connection_worker(
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
        pending_connections: Arc<RwLock<HashMap<String, Instant>>>,
        semaphore: Arc<Semaphore>,
        security_manager: Arc<dyn SecurityManager>,
        validator_scores: Arc<RwLock<HashMap<String, f64>>>,
        stake_distribution: Arc<RwLock<HashMap<String, u64>>>,
        network_entropy: Arc<RwLock<f64>>,
        mut message_rx: mpsc::Receiver<ConnectionMessage>,
        is_running: Arc<atomic::AtomicBool>,
        config: ConnectionConfig,
    ) {
        while is_running.load(atomic::Ordering::SeqCst) {
            tokio::select! {
                Some(message) = message_rx.recv() => {
                    match message {
                        ConnectionMessage::Connect { peer, retry_count, result_tx } => {
                            let connections_clone = connections.clone();
                            let pending_clone = pending_connections.clone();
                            let semaphore_clone = semaphore.clone();
                            let security_clone = security_manager.clone();
                            let validator_clone = validator_scores.clone();
                            let stake_clone = stake_distribution.clone();
                            let entropy_clone = network_entropy.clone();
                            let config_clone = config.clone();
                            
                            tokio::spawn(async move {
                                let result = Self::establish_connection(
                                    peer,
                                    retry_count,
                                    connections_clone,
                                    pending_clone,
                                    semaphore_clone,
                                    security_clone,
                                    validator_clone,
                                    stake_clone,
                                    entropy_clone,
                                    config_clone,
                                ).await;
                                
                                let _ = result_tx.send(result);
                            });
                        }
                        ConnectionMessage::Disconnect { peer_id, reason, result_tx } => {
                            let result = Self::disconnect_peer_internal(
                                &peer_id,
                                reason,
                                connections.clone(),
                            ).await;
                            
                            if let Some(tx) = result_tx {
                                let _ = tx.send(result);
                            }
                        }
                        ConnectionMessage::HealthCheck => {
                            Self::perform_health_check(connections.clone()).await;
                        }
                        ConnectionMessage::ForceRecalculation => {
                            Self::recalculate_forces(
                                connections.clone(),
                                validator_scores.clone(),
                                stake_distribution.clone(),
                                network_entropy.clone(),
                            ).await;
                        }
                        ConnectionMessage::TopologyOptimization => {
                            Self::optimize_topology(
                                connections.clone(),
                                config.quality_threshold,
                                config.stability_threshold,
                            ).await;
                        }
                        ConnectionMessage::UpdatePeerReputation { peer_id, delta } => {
                            Self::update_reputation_internal(&peer_id, delta, connections.clone()).await;
                        }
                        ConnectionMessage::BroadcastMessage { message, exclude_peers } => {
                            Self::broadcast_message_internal(message, exclude_peers, connections.clone()).await;
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    // Periodic maintenance
                    if is_running.load(atomic::Ordering::SeqCst) {
                        Self::cleanup_pending_connections(pending_connections.clone()).await;
                    }
                }
            }
        }
    }

    async fn establish_connection(
        peer: PeerInfo,
        retry_count: u32,
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
        pending_connections: Arc<RwLock<HashMap<String, Instant>>>,
        semaphore: Arc<Semaphore>,
        security_manager: Arc<dyn SecurityManager>,
        validator_scores: Arc<RwLock<HashMap<String, f64>>>,
        stake_distribution: Arc<RwLock<HashMap<String, u64>>>,
        network_entropy: Arc<RwLock<f64>>,
        config: ConnectionConfig,
    ) -> Result<ConnectionState, ConnectionError> {
        // Check connection limits
        let current_connections = connections.read().await.len();
        if current_connections >= config.max_connections {
            return Err(ConnectionError::ConnectionLimitReached {
                current: current_connections,
                max: config.max_connections,
            });
        }

        // Acquire connection slot with timeout
        let permit = tokio::time::timeout(config.connection_timeout, semaphore.acquire())
            .await
            .map_err(|_| ConnectionError::Timeout { duration: config.connection_timeout })??;

        // Mark as pending
        {
            let mut pending = pending_connections.write().await;
            pending.insert(peer.node_id.clone(), Instant::now());
        }

        // Validate peer security
        security_manager.validate_peer(&peer)?;

        let start_time = Instant::now();
        
        // Establish connection based on protocol
        let stream = match peer.protocol {
            ProtocolType::Tcp => {
                let socket_addr = SocketAddr::new(peer.address, peer.port);
                let tcp_stream = TcpStream::connect(socket_addr).await?;
                ConnectionStream::Tcp(Arc::new(RwLock::new(tcp_stream)))
            }
            ProtocolType::WebSocket => {
                let socket_addr = SocketAddr::new(peer.address, peer.port);
                let tcp_stream = TcpStream::connect(socket_addr).await?;
                let ws_stream = tokio_tungstenite::connect_async(
                    format!("ws://{}:{}", peer.address, peer.port)
                ).await?.0;
                ConnectionStream::WebSocket(Arc::new(RwLock::new(ws_stream)))
            }
            ProtocolType::Quic => {
                // QUIC implementation would go here
                return Err(ConnectionError::ProtocolError { reason: "QUIC not implemented".to_string() });
            }
            ProtocolType::Udp => {
                return Err(ConnectionError::ProtocolError { reason: "UDP connections not supported".to_string() });
            }
        };

        // Perform security handshake
        let mut stream_mut = match &stream {
            ConnectionStream::Tcp(arc_stream) => {
                let mut stream_guard = arc_stream.write().await;
                security_manager.perform_handshake(&mut ConnectionStream::Tcp(Arc::new(RwLock::new(*stream_guard))), &peer).await
            }
            ConnectionStream::WebSocket(arc_stream) => {
                let mut stream_guard = arc_stream.write().await;
                security_manager.perform_handshake(&mut ConnectionStream::WebSocket(Arc::new(RwLock::new(stream_guard))), &peer).await
            }
            _ => Err(ConnectionError::ProtocolError { reason: "Unsupported protocol for handshake".to_string() }),
        }?;

        let latency = start_time.elapsed();
        
        // Calculate connection scores
        let quality_score = Self::calculate_quality_score(&peer, latency, &validator_scores, &stake_distribution).await;
        let stability_score = Self::calculate_stability_score(&peer);
        let (attraction, repulsion, net_force) = Self::calculate_connection_forces(
            &peer, 
            &validator_scores, 
            &stake_distribution, 
            &network_entropy,
            latency,
        ).await;

        let connection_state = ConnectionState {
            peer_info: peer.clone(),
            connection_id: format!("{}_{}_{}", peer.node_id, peer.protocol, start_time.elapsed().as_micros()),
            protocol: peer.protocol,
            established_at: Instant::now(),
            last_activity: Instant::now(),
            stream,
            quality_score,
            stability_score,
            metrics: ConnectionMetrics::default(),
            force_attraction: attraction,
            force_repulsion: repulsion,
            net_force,
            failure_count: 0,
        };

        // Add to active connections
        {
            let mut connections_guard = connections.write().await;
            connections_guard.insert(peer.node_id.clone(), connection_state.clone());
        }

        // Cleanup pending
        {
            let mut pending = pending_connections.write().await;
            pending.remove(&peer.node_id);
        }

        // Update metrics
        counter!("connections_established", 1);
        gauge!("active_connections", connections.read().await.len() as f64);
        histogram!("connection_latency", latency.as_secs_f64());

        info!(
            "Connection established with {}: quality={:.3}, force={:.3}, latency={:?}",
            peer.node_id, quality_score, net_force, latency
        );

        // Don't forget the permit is held until connection ends
        // In real implementation, you'd want to release it when connection closes
        drop(permit);

        Ok(connection_state)
    }

    async fn disconnect_peer_internal(
        peer_id: &str,
        reason: DisconnectReason,
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
    ) -> Result<(), ConnectionError> {
        let mut connections_guard = connections.write().await;
        
        if let Some(state) = connections_guard.remove(peer_id) {
            // Perform protocol-specific cleanup
            match state.stream {
                ConnectionStream::Tcp(stream) => {
                    let mut stream_guard = stream.write().await;
                    let _ = stream_guard.shutdown().await;
                }
                ConnectionStream::WebSocket(ws_stream) => {
                    let mut ws_guard = ws_stream.write().await;
                    let _ = ws_guard.close(None).await;
                }
                ConnectionStream::Quic(quic_conn) => {
                    quic_conn.close(0u32.into(), b"graceful");
                }
            }

            counter!("connections_closed", 1);
            gauge!("active_connections", connections_guard.len() as f64);
            
            debug!("Disconnected from peer {}: {:?}", peer_id, reason);
        }

        Ok(())
    }

    async fn calculate_quality_score(
        peer: &PeerInfo,
        latency: Duration,
        validator_scores: &RwLock<HashMap<String, f64>>,
        stake_distribution: &RwLock<HashMap<String, u64>>,
    ) -> f64 {
        let validator_scores_guard = validator_scores.read().await;
        let stake_guard = stake_distribution.read().await;
        
        let validator_score = validator_scores_guard.get(&peer.node_id).copied().unwrap_or(0.5);
        let stake_amount = stake_guard.get(&peer.node_id).copied().unwrap_or(0);
        let normalized_stake = (stake_amount as f64).min(1_000_000.0) / 1_000_000.0; // Normalize to 0-1
        
        let reputation = (peer.reputation.max(0) as f64) / 100.0;
        let latency_score = 1.0 / (1.0 + latency.as_secs_f64());
        let stability = peer.connection_count as f64 / (peer.connection_count + peer.failed_attempts + 1) as f64;
        
        // Weighted combination
        let weights = [0.25, 0.20, 0.15, 0.20, 0.20]; // validator, stake, reputation, latency, stability
        let scores = [validator_score, normalized_stake, reputation, latency_score, stability];
        
        scores.iter().zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum()
    }

    async fn calculate_stability_score(peer: &PeerInfo) -> f64 {
        let total_attempts = peer.connection_count + peer.failed_attempts;
        if total_attempts == 0 {
            return 0.5;
        }
        
        let success_ratio = peer.connection_count as f64 / total_attempts as f64;
        let time_weight = 1.0 - (peer.last_seen.elapsed().as_secs_f64() / 86400.0).min(1.0); // 24-hour decay
        
        success_ratio * time_weight
    }

    async fn calculate_connection_forces(
        peer: &PeerInfo,
        validator_scores: &RwLock<HashMap<String, f64>>,
        stake_distribution: &RwLock<HashMap<String, u64>>,
        network_entropy: &RwLock<f64>,
        latency: Duration,
    ) -> (f64, f64, f64) {
        let validator_scores_guard = validator_scores.read().await;
        let stake_guard = stake_distribution.read().await;
        let entropy = *network_entropy.read().await;
        
        // Attraction components
        let validator_attraction = validator_scores_guard.get(&peer.node_id).copied().unwrap_or(0.5);
        let stake_attraction = (stake_guard.get(&peer.node_id).copied().unwrap_or(0) as f64).min(1_000_000.0) / 1_000_000.0;
        let reputation_attraction = (peer.reputation.max(0) as f64) / 100.0;
        
        let attraction = (validator_attraction + stake_attraction + reputation_attraction) / 3.0;
        
        // Repulsion components  
        let latency_repulsion = 1.0 / (1.0 + (-latency.as_secs_f64()).exp());
        let entropy_repulsion = entropy;
        let failure_repulsion = peer.failed_attempts as f64 / (peer.connection_count + peer.failed_attempts + 1) as f64;
        
        let repulsion = (latency_repulsion + entropy_repulsion + failure_repulsion) / 3.0;
        
        (attraction, repulsion, attraction - repulsion)
    }

    async fn perform_health_check(connections: Arc<RwLock<HashMap<String, ConnectionState>>>) {
        let mut unhealthy_peers = Vec::new();
        let current_time = Instant::now();
        
        {
            let connections_guard = connections.read().await;
            
            for (peer_id, state) in connections_guard.iter() {
                // Check for stale connections
                if current_time.duration_since(state.last_activity) > Duration::from_secs(60) {
                    unhealthy_peers.push(peer_id.clone());
                    continue;
                }
                
                // Check quality thresholds
                if state.quality_score < 0.3 || state.stability_score < 0.4 {
                    unhealthy_peers.push(peer_id.clone());
                    continue;
                }
                
                // Check metrics degradation
                if state.metrics.success_rate < 0.5 || state.metrics.error_count > 10 {
                    unhealthy_peers.push(peer_id.clone());
                }
            }
        }
        
        // Disconnect unhealthy peers
        for peer_id in unhealthy_peers {
            let connections_clone = connections.clone();
            tokio::spawn(async move {
                let _ = Self::disconnect_peer_internal(
                    &peer_id,
                    DisconnectReason::HealthCheckFailed,
                    connections_clone,
                ).await;
            });
        }
    }

    async fn recalculate_forces(
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
        validator_scores: Arc<RwLock<HashMap<String, f64>>>,
        stake_distribution: Arc<RwLock<HashMap<String, u64>>>,
        network_entropy: Arc<RwLock<f64>>,
    ) {
        let mut connections_guard = connections.write().await;
        
        for (_, state) in connections_guard.iter_mut() {
            let (attraction, repulsion, net_force) = Self::calculate_connection_forces(
                &state.peer_info,
                &validator_scores,
                &stake_distribution,
                &network_entropy,
                state.metrics.average_latency,
            ).await;
            
            state.force_attraction = attraction;
            state.force_repulsion = repulsion;
            state.net_force = net_force;
            state.quality_score = Self::calculate_quality_score(
                &state.peer_info,
                state.metrics.average_latency,
                &validator_scores,
                &stake_distribution,
            ).await;
        }
    }

    async fn optimize_topology(
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
        quality_threshold: f64,
        stability_threshold: f64,
    ) {
        let mut eviction_candidates = Vec::new();
        
        {
            let connections_guard = connections.read().await;
            
            for (peer_id, state) in connections_guard.iter() {
                if state.quality_score < quality_threshold || state.stability_score < stability_threshold {
                    eviction_candidates.push(peer_id.clone());
                }
            }
        }
        
        for peer_id in eviction_candidates {
            let connections_clone = connections.clone();
            tokio::spawn(async move {
                let _ = Self::disconnect_peer_internal(
                    &peer_id,
                    DisconnectReason::PoorQuality,
                    connections_clone,
                ).await;
            });
        }
    }

    async fn update_reputation_internal(
        peer_id: &str,
        delta: i32,
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
    ) {
        let mut connections_guard = connections.write().await;
        
        if let Some(state) = connections_guard.get_mut(peer_id) {
            state.peer_info.reputation = (state.peer_info.reputation + delta).clamp(0, 100);
        }
    }

    async fn broadcast_message_internal(
        message: Vec<u8>,
        exclude_peers: HashSet<String>,
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
    ) {
        let connections_guard = connections.read().await;
        
        for (peer_id, state) in connections_guard.iter() {
            if exclude_peers.contains(peer_id) {
                continue;
            }
            
            // In real implementation, this would actually send the message
            // through the appropriate protocol handler
            debug!("Broadcasting message to {} ({} bytes)", peer_id, message.len());
        }
    }

    async fn cleanup_pending_connections(pending_connections: Arc<RwLock<HashMap<String, Instant>>>) {
        let mut pending_guard = pending_connections.write().await;
        let now = Instant::now();
        
        pending_guard.retain(|_, &mut timestamp| now.duration_since(timestamp) < Duration::from_secs(30));
    }

    async fn health_monitor_worker(
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
        is_running: Arc<atomic::AtomicBool>,
        interval: Duration,
    ) {
        let mut interval = tokio::time::interval(interval);
        
        while is_running.load(atomic::Ordering::SeqCst) {
            interval.tick().await;
            Self::perform_health_check(connections.clone()).await;
        }
    }

    async fn force_calculation_worker(
        connections: Arc<RwLock<HashMap<String, ConnectionState>>>,
        validator_scores: Arc<RwLock<HashMap<String, f64>>>,
        stake_distribution: Arc<RwLock<HashMap<String, u64>>>,
        network_entropy: Arc<RwLock<f64>>,
        is_running: Arc<atomic::AtomicBool>,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        
        while is_running.load(atomic::Ordering::SeqCst) {
            interval.tick().await;
            Self::recalculate_forces(
                connections.clone(),
                validator_scores.clone(),
                stake_distribution.clone(),
                network_entropy.clone(),
            ).await;
        }
    }
}

// Implement required traits
impl Drop for ConnectionManager {
    fn drop(&mut self) {
        if self.is_running.load(atomic::Ordering::SeqCst) {
            let manager = self.clone();
            tokio::spawn(async move {
                let _ = manager.stop().await;
            });
        }
    }
}

// Clone implementation for ConnectionManager
impl Clone for ConnectionManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            node_id: self.node_id.clone(),
            connections: self.connections.clone(),
            pending_connections: self.pending_connections.clone(),
            connection_semaphore: self.connection_semaphore.clone(),
            message_tx: self.message_tx.clone(),
            security_manager: self.security_manager.clone(),
            validator_scores: self.validator_scores.clone(),
            stake_distribution: self.stake_distribution.clone(),
            network_entropy: self.network_entropy.clone(),
            is_running: self.is_running.clone(),
        }
    }
}