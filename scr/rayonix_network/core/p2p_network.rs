// network/core/p2p_network.rs
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, error, instrument};
use metrics::{counter, gauge, histogram};

use crate::{
    config::{NodeConfig, NetworkType, ProtocolType, ConnectionState, MessageType},
    models::{PeerInfo, NetworkMessage},
    errors::NetworkError,
    core::{
        ConnectionManager, 
        MessageProcessor, 
        PeerDiscovery, 
        SecurityManager,
        PhysicsModel  // New physics-inspired component
    },
    protocols::{TcpHandler, WebSocketHandler, HttpHandler},
    utils::{
        RateLimiter, 
        BanManager, 
        MetricsCollector,
        Dht
    },
};

#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    pub node_id: String,
    pub uptime: Duration,
    pub peers_count: usize,
    pub connections_count: usize,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub banned_peers: usize,
    pub network_entropy: f64,
    pub average_latency: f64,
    pub message_throughput: f64,
}

pub struct AdvancedP2PNetwork {
    // Core identity and configuration
    pub node_id: String,
    pub config: NodeConfig,
    pub magic: [u8; 4],
    
    // Core subsystems
    pub connection_manager: Arc<ConnectionManager>,
    pub message_processor: Arc<MessageProcessor>,
    pub peer_discovery: Arc<PeerDiscovery>,
    pub security_manager: Arc<SecurityManager>,
    pub physics_model: Arc<PhysicsModel>,  // Hybrid physics-inspired model
    
    // Protocol handlers
    pub tcp_handler: Arc<TcpHandler>,
    pub websocket_handler: Arc<WebSocketHandler>,
    pub http_handler: Arc<HttpHandler>,
    
    // Utility components
    pub rate_limiter: Arc<RateLimiter>,
    pub ban_manager: Arc<BanManager>,
    pub metrics_collector: Arc<MetricsCollector>,
    pub dht: Arc<Dht>,
    
    // State management
    pub is_running: Arc<atomic::AtomicBool>,
    pub start_time: Instant,
    pub peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    pub connections: Arc<RwLock<HashMap<String, crate::models::Connection>>>,
    
    // Message handling
    pub message_handlers: Arc<RwLock<HashMap<MessageType, Vec<Arc<dyn MessageHandler>>>>>,
    
    // Control channels
    pub control_tx: mpsc::Sender<ControlMessage>,
    pub control_rx: Arc<RwLock<mpsc::Receiver<ControlMessage>>>,
    
    // Task management
    maintenance_task: Option<tokio::task::JoinHandle<()>>,
    metrics_task: Option<tokio::task::JoinHandle<()>>,
    physics_task: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Debug)]
pub enum ControlMessage {
    Shutdown,
    BroadcastMessage(NetworkMessage, Option<Vec<String>>),
    ConnectToPeer(String),  // address:port
    DisconnectPeer(String), // peer_id
    UpdateConfig(NodeConfig),
    ForceTopologyOptimization,
    InjectEntropy(f64),
}

#[async_trait::async_trait]
pub trait MessageHandler: Send + Sync {
    async fn handle_message(&self, connection_id: &str, message: &NetworkMessage) -> Result<(), NetworkError>;
}

#[async_trait::async_trait]
pub trait ConsensusCallback: Send + Sync {
    async fn on_block_proposal(&self, block: &BlockData) -> Result<(), NetworkError>;
    async fn on_validator_score_update(&self, validator: &str, score: f64) -> Result<(), NetworkError>;
    async fn on_network_metrics(&self, metrics: &NetworkMetrics) -> Result<(), NetworkError>;
    async fn get_validator_scores(&self) -> HashMap<String, f64>;
    async fn get_stake_distribution(&self) -> HashMap<String, u64>;
}

impl AdvancedP2PNetwork {
    pub fn new(config: NodeConfig, private_key: Option<Vec<u8>>) -> Result<Self, NetworkError> {
        let node_id = Self::generate_node_id(private_key.as_deref());
        let magic = Self::get_magic_number(config.network_type);
        
        // Create control channel
        let (control_tx, control_rx) = mpsc::channel(1000);
        
        let network = Arc::new(RwLock::new(None));
        
        // Initialize utility components first
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit_per_peer));
        let ban_manager = Arc::new(BanManager::new(config.ban_threshold, config.ban_duration));
        let metrics_collector = Arc::new(MetricsCollector::new());
        let dht = Arc::new(Dht::new(node_id.clone(), config.dht_bootstrap_nodes.clone()));
        
        // Initialize core subsystems
        let security_manager = Arc::new(SecurityManager::new(
            network.clone(),
            private_key,
            config.enable_encryption,
        ));
        
        let connection_manager = Arc::new(ConnectionManager::new(
            config.connection_config.clone(),
            node_id.clone(),
            security_manager.clone(),
        ));
        
        let message_processor = Arc::new(MessageProcessor::new(network.clone()));
        let peer_discovery = Arc::new(PeerDiscovery::new(network.clone(), config.bootstrap_nodes.clone()));
        
        // Initialize physics-inspired model (hybrid approach)
        let physics_model = Arc::new(PhysicsModel::new(
            network.clone(),
            config.physics_config.clone(),
        ));
        
        // Initialize protocol handlers
        let tcp_handler = Arc::new(TcpHandler::new(
            network.clone(),
            config.listen_ip.clone(),
            config.listen_port,
            config.ssl_context.clone(),
        ));
        
        let websocket_handler = Arc::new(WebSocketHandler::new(
            network.clone(),
            config.listen_ip.clone(),
            config.websocket_port,
            config.ssl_context.clone(),
        ));
        
        let http_handler = Arc::new(HttpHandler::new(
            network.clone(),
            config.listen_ip.clone(),
            config.http_port,
        ));
        
        let network_instance = Self {
            node_id,
            config,
            magic,
            connection_manager,
            message_processor,
            peer_discovery,
            security_manager,
            physics_model,
            tcp_handler,
            websocket_handler,
            http_handler,
            rate_limiter,
            ban_manager,
            metrics_collector,
            dht,
            is_running: Arc::new(atomic::AtomicBool::new(false)),
            start_time: Instant::now(),
            peers: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            control_tx,
            control_rx: Arc::new(RwLock::new(control_rx)),
            maintenance_task: None,
            metrics_task: None,
            physics_task: None,
        };
        
        // Set the network reference
        *network.write().await = Some(Arc::new(network_instance));
        
        Ok(network_instance)
    }
    
    fn generate_node_id(private_key: Option<&[u8]>) -> String {
        if let Some(key) = private_key {
            // Generate from private key for consistency
            let hash = blake3::hash(key);
            format!("node_{}", hex::encode(&hash.as_bytes()[..16]))
        } else {
            // Fallback to random ID
            format!("node_{}", uuid::Uuid::new_v4())
        }
    }
    
    fn get_magic_number(network_type: NetworkType) -> [u8; 4] {
        match network_type {
            NetworkType::Mainnet => *b"RAYX",
            NetworkType::Testnet => *b"RAYT", 
            NetworkType::Devnet => *b"RAYD",
            _ => *b"RAYX",
        }
    }
    
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<(), NetworkError> {
        if self.is_running.load(atomic::Ordering::SeqCst) {
            return Err(NetworkError::AlreadyRunning);
        }
        
        info!("Starting Advanced P2P Network (Node ID: {})", self.node_id);
        
        // Start protocol handlers
        self.tcp_handler.start().await?;
        if self.config.websocket_port != self.config.listen_port {
            self.websocket_handler.start().await?;
        }
        if self.config.http_port != self.config.listen_port && self.config.http_port != self.config.websocket_port {
            self.http_handler.start().await?;
        }
        
        // Start core subsystems
        self.security_manager.initialize().await?;
        self.connection_manager.start().await?;
        self.message_processor.start().await?;
        
        // Bootstrap network
        if let Err(e) = self.peer_discovery.bootstrap_network().await {
            warn!("Initial bootstrap failed: {}", e);
            // Continue anyway - we can discover peers later
        }
        
        // Start background tasks
        self.start_background_tasks().await;
        
        self.is_running.store(true, atomic::Ordering::SeqCst);
        self.start_time = Instant::now();
        
        info!("Advanced P2P Network started successfully");
        Ok(())
    }
    
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<(), NetworkError> {
        if !self.is_running.swap(false, atomic::Ordering::SeqCst) {
            return Ok(());
        }
        
        info!("Stopping Advanced P2P Network");
        
        // Stop background tasks
        self.stop_background_tasks().await;
        
        // Stop protocol handlers
        self.tcp_handler.stop().await?;
        self.websocket_handler.stop().await?;
        self.http_handler.stop().await?;
        
        // Stop core subsystems
        self.connection_manager.stop().await?;
        self.message_processor.stop().await?;
        
        // Close all connections
        let mut connections = self.connections.write().await;
        connections.clear();
        
        info!("Advanced P2P Network stopped successfully");
        Ok(())
    }
    
    async fn start_background_tasks(&self) {
        let network = Arc::new(self.clone());
        
        // Maintenance task (from Python version)
        self.maintenance_task = Some(tokio::spawn(Self::maintenance_loop(network.clone())));
        
        // Metrics task (from Python version)  
        self.metrics_task = Some(tokio::spawn(Self::metrics_loop(network.clone())));
        
        // Physics model task (from Go version, simplified)
        self.physics_task = Some(tokio::spawn(Self::physics_loop(network)));
    }
    
    async fn stop_background_tasks(&self) {
        if let Some(task) = self.maintenance_task.take() {
            task.abort();
        }
        if let Some(task) = self.metrics_task.take() {
            task.abort();
        }
        if let Some(task) = self.physics_task.take() {
            task.abort();
        }
    }
    
    async fn maintenance_loop(network: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        while network.is_running.load(atomic::Ordering::SeqCst) {
            interval.tick().await;
            
            // Combined maintenance operations from Python version
            tokio::join!(
                network.peer_discovery.maintain_peer_list(),
                network.peer_discovery.request_peer_lists(),
                network.ban_manager.cleanup_expired_bans(),
                network.connection_manager.check_connection_health(),
                network.send_pings()
            );
        }
    }
    
    async fn metrics_loop(network: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        while network.is_running.load(atomic::Ordering::SeqCst) {
            interval.tick().await;
            
            let banned_count = network.ban_manager.get_banned_count().await;
            let connections_count = network.connections.read().await.len();
            let peers_count = network.peers.read().await.len();
            
            network.metrics_collector.log_metrics(
                connections_count,
                peers_count,
                banned_count,
            ).await;
        }
    }
    
    async fn physics_loop(network: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(5)); // Slower than Go version
        
        while network.is_running.load(atomic::Ordering::SeqCst) {
            interval.tick().await;
            
            // Simplified physics model from Go version
            if let Err(e) = network.physics_model.update().await {
                warn!("Physics model update failed: {}", e);
            }
            
            // Topology optimization based on physics model
            if let Err(e) = network.optimize_topology_based_on_physics().await {
                debug!("Topology optimization skipped: {}", e);
            }
        }
    }
    
    async fn send_pings(&self) {
        let connections = self.connections.read().await;
        for connection_id in connections.keys() {
            let ping_message = NetworkMessage::ping(
                &self.node_id,
                Some(connection_id.clone()),
            );
            
            if let Err(e) = self.message_processor.send_message(connection_id, ping_message).await {
                debug!("Failed to send ping to {}: {}", connection_id, e);
            }
        }
    }
    
    async fn optimize_topology_based_on_physics(&self) -> Result<(), NetworkError> {
        let physics_state = self.physics_model.get_state().await;
        
        // Use physics scores to optimize connections
        let connections = self.connections.read().await;
        let mut eviction_candidates = Vec::new();
        
        for (connection_id, connection) in connections.iter() {
            if let Some(peer_info) = &connection.peer_info {
                let physics_score = physics_state.get_connection_score(&peer_info.node_id);
                
                // Evict low-scoring connections (simplified from Go version)
                if physics_score < 0.3 {
                    eviction_candidates.push(connection_id.clone());
                }
            }
        }
        
        // Disconnect low-quality peers
        for connection_id in eviction_candidates {
            if let Err(e) = self.connection_manager.disconnect_peer(&connection_id, crate::core::DisconnectReason::PoorQuality).await {
                warn!("Failed to disconnect peer {}: {}", connection_id, e);
            }
        }
        
        Ok(())
    }
    
    // Public API methods combining Python simplicity with Go advanced features
    
    #[instrument(skip(self))]
    pub async fn connect_to_peer(&self, address: &str) -> Result<String, NetworkError> {
        // Parse address (supports "ip:port" and "protocol://ip:port")
        let (protocol, ip, port) = Self::parse_address(address)?;
        
        // Create peer info
        let peer_info = PeerInfo {
            node_id: "".to_string(), // Will be set during handshake
            address: ip.parse().map_err(|e| NetworkError::InvalidAddress(e.to_string()))?,
            port,
            protocol,
            // ... other fields with defaults
            ..Default::default()
        };
        
        // Use physics model to check if we should connect
        if !self.physics_model.should_connect_to_peer(&peer_info).await {
            return Err(NetworkError::PhysicsModelRejection);
        }
        
        // Establish connection through connection manager
        let connection_state = self.connection_manager.connect_to_peer(peer_info).await?;
        
        Ok(connection_state.connection_id)
    }
    
    #[instrument(skip(self))]
    pub async fn disconnect_peer(&self, peer_id: &str) -> Result<(), NetworkError> {
        self.connection_manager.disconnect_peer(peer_id, crate::core::DisconnectReason::Graceful).await
    }
    
    #[instrument(skip(self, message))]
    pub async fn send_message(&self, peer_id: &str, message: NetworkMessage) -> Result<(), NetworkError> {
        // Use physics-aware routing (simplified from Go version)
        if self.physics_model.should_route_message(peer_id, &message).await {
            self.message_processor.send_message(peer_id, message).await
        } else {
            Err(NetworkError::RoutingRejected)
        }
    }
    
    #[instrument(skip(self, message))]
    pub async fn broadcast_message(&self, message: NetworkMessage, exclude: Option<Vec<String>>) -> HashMap<String, Result<(), NetworkError>> {
        // Use physics-aware broadcasting (simplified from Go version)
        let peers_to_broadcast = self.physics_model.select_broadcast_peers(&message, exclude.clone()).await;
        
        let mut results = HashMap::new();
        for peer_id in peers_to_broadcast {
            let result = self.message_processor.send_message(&peer_id, message.clone()).await;
            results.insert(peer_id, result);
        }
        
        results
    }
    
    pub async fn register_message_handler(&self, message_type: MessageType, handler: Arc<dyn MessageHandler>) {
        let mut handlers = self.message_handlers.write().await;
        handlers.entry(message_type).or_insert_with(Vec::new).push(handler);
    }
    
    pub async fn get_network_stats(&self) -> NetworkMetrics {
        let global_metrics = self.metrics_collector.get_global_metrics().await;
        let physics_state = self.physics_model.get_state().await;
        
        NetworkMetrics {
            node_id: self.node_id.clone(),
            uptime: self.start_time.elapsed(),
            peers_count: self.peers.read().await.len(),
            connections_count: self.connections.read().await.len(),
            bytes_sent: global_metrics.bytes_sent,
            bytes_received: global_metrics.bytes_received,
            messages_sent: global_metrics.messages_sent,
            messages_received: global_metrics.messages_received,
            banned_peers: self.ban_manager.get_banned_count().await,
            network_entropy: physics_state.entropy,
            average_latency: global_metrics.average_latency,
            message_throughput: global_metrics.message_rate,
        }
    }
    
    pub async fn get_connected_peers(&self) -> Vec<PeerInfo> {
        self.connection_manager.get_active_peers().await
    }
    
    // Helper methods
    fn parse_address(address: &str) -> Result<(ProtocolType, String, u16), NetworkError> {
        // Implementation from Python version
        if let Some(protocol_end) = address.find("://") {
            let protocol_str = &address[..protocol_end];
            let addr_port = &address[protocol_end + 3..];
            
            let protocol = ProtocolType::from_str(protocol_str)
                .map_err(|_| NetworkError::InvalidProtocol(protocol_str.to_string()))?;
            
            let (ip, port) = Self::parse_ip_port(addr_port)?;
            Ok((protocol, ip, port))
        } else {
            let (ip, port) = Self::parse_ip_port(address)?;
            Ok((ProtocolType::Tcp, ip, port)) // Default to TCP
        }
    }
    
    fn parse_ip_port(addr_port: &str) -> Result<(String, u16), NetworkError> {
        if let Some(port_start) = addr_port.rfind(':') {
            let ip = addr_port[..port_start].to_string();
            let port_str = &addr_port[port_start + 1..];
            let port = port_str.parse().map_err(|e| NetworkError::InvalidPort(e.to_string()))?;
            Ok((ip, port))
        } else {
            // No port specified, use default
            Ok((addr_port.to_string(), 30303))
        }
    }
}

impl Clone for AdvancedP2PNetwork {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id.clone(),
            config: self.config.clone(),
            magic: self.magic,
            connection_manager: self.connection_manager.clone(),
            message_processor: self.message_processor.clone(),
            peer_discovery: self.peer_discovery.clone(),
            security_manager: self.security_manager.clone(),
            physics_model: self.physics_model.clone(),
            tcp_handler: self.tcp_handler.clone(),
            websocket_handler: self.websocket_handler.clone(),
            http_handler: self.http_handler.clone(),
            rate_limiter: self.rate_limiter.clone(),
            ban_manager: self.ban_manager.clone(),
            metrics_collector: self.metrics_collector.clone(),
            dht: self.dht.clone(),
            is_running: self.is_running.clone(),
            start_time: self.start_time,
            peers: self.peers.clone(),
            connections: self.connections.clone(),
            message_handlers: self.message_handlers.clone(),
            control_tx: self.control_tx.clone(),
            control_rx: self.control_rx.clone(),
            maintenance_task: None,
            metrics_task: None,
            physics_task: None,
        }
    }
}

impl Drop for AdvancedP2PNetwork {
    fn drop(&mut self) {
        if self.is_running.load(atomic::Ordering::SeqCst) {
            let network = self.clone();
            tokio::spawn(async move {
                let _ = network.stop().await;
            });
        }
    }
}