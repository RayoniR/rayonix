use std::collections::HashSet;
use std::net::{IpAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use thiserror::Error;

// Configuration types (would normally be in config module)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolType {
    Tcp,
    Udp,
    WebSocket,
    Quic,
    Http,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportType {
    Plaintext,
    Tls,
    Noise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Disconnecting,
    Failed,
    Banned,
}

#[derive(Debug, Error)]
pub enum PeerError {
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

type Result<T> = std::result::Result<T, PeerError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    // Core identification and network properties
    pub node_id: String,
    pub public_key: Vec<u8>,
    pub address: String,
    pub port: u16,
    pub protocol: ProtocolType,
    pub transport: TransportType,
    
    // Network characteristics and capabilities
    pub version: String,
    pub capabilities: HashSet<String>,
    pub user_agent: String,
    pub network_id: u32,
    
    // State management
    pub state: ConnectionState,
    pub last_seen: SystemTime,
    pub first_seen: SystemTime,
    pub connected_at: Option<SystemTime>,
    pub disconnected_at: Option<SystemTime>,
    
    // Performance metrics (atomic for concurrent access)
    reputation: Arc<AtomicI32>,
    latency: Arc<RwLock<Duration>>,
    response_time: Arc<RwLock<Duration>>,
    uptime: Arc<RwLock<Duration>>,
    
    // Connection statistics (atomic counters)
    connection_count: Arc<AtomicU32>,
    failed_attempts: Arc<AtomicU32>,
    successful_pings: Arc<AtomicU32>,
    failed_pings: Arc<AtomicU32>,
    
    // Bandwidth and message statistics (atomic counters)
    bytes_sent: Arc<AtomicU64>,
    bytes_received: Arc<AtomicU64>,
    messages_sent: Arc<AtomicU64>,
    messages_received: Arc<AtomicU64>,
    
    // Physics-inspired properties
    potential_energy: Arc<RwLock<f64>>,
    kinetic_energy: Arc<RwLock<f64>>,
    entropy_contribution: Arc<RwLock<f64>>,
    force_vector: Arc<RwLock<ForceVector>>,
    
    // Consensus integration
    validator_score: Arc<RwLock<f64>>,
    stake_amount: Arc<RwLock<f64>>,
    voting_power: Arc<RwLock<f64>>,
    
    // Security and validation
    is_bootstrap: bool,
    is_validator: Arc<RwLock<bool>>,
    is_banned: Arc<RwLock<bool>>,
    ban_reason: Arc<RwLock<String>>,
    ban_expiry: Arc<RwLock<Option<SystemTime>>>,
    
    // Geographic and network location
    country_code: Arc<RwLock<String>>,
    region: Arc<RwLock<String>>,
    asn: Arc<RwLock<u32>>,
    coordinates: Arc<RwLock<Option<GeoCoordinates>>>,
    
    // Resource constraints
    max_connections: u32,
    rate_limit: u64,
    memory_usage: Arc<RwLock<u64>>,
    
    // Synchronization and versioning
    last_updated: Arc<RwLock<SystemTime>>,
    version: Arc<AtomicU64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceVector {
    pub magnitude: f64,
    pub direction_x: f64,
    pub direction_y: f64,
    pub direction_z: f64,
    pub attraction: f64,
    pub repulsion: f64,
    pub last_updated: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoCoordinates {
    pub latitude: f64,
    pub longitude: f64,
    pub accuracy: f64, // Accuracy in kilometers
}

impl PeerInfo {
    pub fn new(
        node_id: String,
        address: String,
        port: u16,
        protocol: ProtocolType,
    ) -> Result<Self> {
        let now = SystemTime::now();
        
        let mut capabilities = HashSet::new();
        capabilities.insert("tcp".to_string());
        capabilities.insert("udp".to_string());
        capabilities.insert("gossip".to_string());
        capabilities.insert("syncing".to_string());
        
        let peer = Self {
            node_id,
            public_key: Vec::new(),
            address,
            port,
            protocol,
            transport: TransportType::Plaintext,
            version: "1.0.0".to_string(),
            capabilities,
            user_agent: "RayX-P2P/1.0.0".to_string(),
            network_id: 1,
            state: ConnectionState::Disconnected,
            last_seen: now,
            first_seen: now,
            connected_at: None,
            disconnected_at: None,
            reputation: Arc::new(AtomicI32::new(50)), // Neutral starting reputation
            latency: Arc::new(RwLock::new(Duration::ZERO)),
            response_time: Arc::new(RwLock::new(Duration::ZERO)),
            uptime: Arc::new(RwLock::new(Duration::ZERO)),
            connection_count: Arc::new(AtomicU32::new(0)),
            failed_attempts: Arc::new(AtomicU32::new(0)),
            successful_pings: Arc::new(AtomicU32::new(0)),
            failed_pings: Arc::new(AtomicU32::new(0)),
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            potential_energy: Arc::new(RwLock::new(1.0)),
            kinetic_energy: Arc::new(RwLock::new(0.0)),
            entropy_contribution: Arc::new(RwLock::new(0.5)),
            force_vector: Arc::new(RwLock::new(ForceVector {
                magnitude: 0.0,
                direction_x: 0.0,
                direction_y: 0.0,
                direction_z: 0.0,
                attraction: 0.5,
                repulsion: 0.5,
                last_updated: now,
            })),
            validator_score: Arc::new(RwLock::new(0.0)),
            stake_amount: Arc::new(RwLock::new(0.0)),
            voting_power: Arc::new(RwLock::new(0.0)),
            is_bootstrap: false,
            is_validator: Arc::new(RwLock::new(false)),
            is_banned: Arc::new(RwLock::new(false)),
            ban_reason: Arc::new(RwLock::new(String::new())),
            ban_expiry: Arc::new(RwLock::new(None)),
            country_code: Arc::new(RwLock::new(String::new())),
            region: Arc::new(RwLock::new(String::new())),
            asn: Arc::new(RwLock::new(0)),
            coordinates: Arc::new(RwLock::new(None)),
            max_connections: 50,
            rate_limit: 1000,
            memory_usage: Arc::new(RwLock::new(0)),
            last_updated: Arc::new(RwLock::new(now)),
            version: Arc::new(AtomicU64::new(1)),
        };

        peer.validate()?;
        Ok(peer)
    }

    pub fn validate(&self) -> Result<()> {
        if self.node_id.is_empty() {
            return Err(PeerError::Validation("node ID cannot be empty".to_string()));
        }
        
        if self.node_id.len() > 256 {
            return Err(PeerError::Validation(format!(
                "node ID too long: {} characters",
                self.node_id.len()
            )));
        }
        
        if self.address.is_empty() {
            return Err(PeerError::Validation("address cannot be empty".to_string()));
        }
        
        // Validate IP address format
        if let Ok(ip_addr) = self.address.parse::<IpAddr>() {
            if ip_addr.is_unspecified() {
                return Err(PeerError::Validation("address cannot be unspecified".to_string()));
            }
            if ip_addr.is_multicast() {
                return Err(PeerError::Validation("address cannot be multicast".to_string()));
            }
        } else {
            // Check if it's a valid hostname by attempting resolution
            let socket_addr = format!("{}:{}", self.address, self.port);
            if socket_addr.to_socket_addrs().is_err() {
                return Err(PeerError::Validation(format!(
                    "invalid address format: {}",
                    self.address
                )));
            }
        }
        
        if self.port == 0 || self.port > 65535 {
            return Err(PeerError::Validation(format!(
                "invalid port number: {}",
                self.port
            )));
        }
        
        if self.version.is_empty() {
            return Err(PeerError::Validation("version cannot be empty".to_string()));
        }
        
        let reputation = self.reputation.load(Ordering::Acquire);
        if reputation < -100 || reputation > 100 {
            return Err(PeerError::Validation(format!(
                "reputation out of range: {}",
                reputation
            )));
        }
        
        let latency = self.get_latency();
        if latency < Duration::ZERO {
            return Err(PeerError::Validation(format!(
                "latency cannot be negative: {:?}",
                latency
            )));
        }
        
        let potential_energy = self.get_potential_energy();
        if potential_energy < 0.0 {
            return Err(PeerError::Validation(format!(
                "potential energy cannot be negative: {}",
                potential_energy
            )));
        }
        
        Ok(())
    }

    pub fn update_reputation(&self, delta: i32) -> i32 {
        let new_rep = self.reputation.fetch_add(delta, Ordering::AcqRel) + delta;
        
        // Clamp reputation to valid range [-100, 100]
        if new_rep < -100 {
            self.reputation.store(-100, Ordering::Release);
            -100
        } else if new_rep > 100 {
            self.reputation.store(100, Ordering::Release);
            100
        } else {
            new_rep
        }
    }

    pub async fn record_success(
        &self,
        latency: Duration,
        bytes_sent: u64,
        bytes_received: u64,
    ) {
        let now = SystemTime::now();
        
        // Update timestamps
        {
            let mut last_updated = self.last_updated.write().await;
            *last_updated = now;
        }
        self.last_seen = now;
        
        // Update latency
        {
            let mut latency_guard = self.latency.write().await;
            *latency_guard = latency;
        }
        
        // Update atomic counters
        self.successful_pings.fetch_add(1, Ordering::AcqRel);
        self.bytes_sent.fetch_add(bytes_sent, Ordering::AcqRel);
        self.bytes_received.fetch_add(bytes_received, Ordering::AcqRel);
        
        // Positive reputation adjustment for success
        self.update_reputation(1);
        
        self.increment_version();
    }

    pub async fn record_failure(&self, reason: &str) {
        self.failed_attempts.fetch_add(1, Ordering::AcqRel);
        self.failed_pings.fetch_add(1, Ordering::AcqRel);
        
        // Update ban reason if provided
        if !reason.is_empty() {
            let mut ban_reason = self.ban_reason.write().await;
            *ban_reason = reason.to_string();
        }
        
        // Negative reputation adjustment for failure
        self.update_reputation(-5);
        
        self.increment_version();
    }

    pub async fn update_physics_state(
        &self,
        potential: f64,
        kinetic: f64,
        entropy: f64,
        force: Option<ForceVector>,
    ) {
        {
            let mut potential_energy = self.potential_energy.write().await;
            *potential_energy = potential;
        }
        {
            let mut kinetic_energy = self.kinetic_energy.write().await;
            *kinetic_energy = kinetic;
        }
        {
            let mut entropy_contribution = self.entropy_contribution.write().await;
            *entropy_contribution = entropy;
        }
        
        if let Some(force) = force {
            let mut force_vector = self.force_vector.write().await;
            *force_vector = force;
            force_vector.last_updated = SystemTime::now();
        }
        
        self.increment_version();
    }

    pub async fn update_consensus_data(
        &self,
        score: f64,
        stake: f64,
        power: f64,
        is_validator: bool,
    ) {
        {
            let mut validator_score = self.validator_score.write().await;
            *validator_score = score;
        }
        {
            let mut stake_amount = self.stake_amount.write().await;
            *stake_amount = stake;
        }
        {
            let mut voting_power = self.voting_power.write().await;
            *voting_power = power;
        }
        {
            let mut is_validator_guard = self.is_validator.write().await;
            *is_validator_guard = is_validator;
        }
        
        self.increment_version();
    }

    pub async fn calculate_quality_score(&self) -> f64 {
        // Base quality factors
        let reputation = self.reputation.load(Ordering::Acquire);
        let reputation_factor = (reputation + 100) as f64 / 200.0; // Normalize to [0,1]
        
        let latency = self.get_latency();
        let latency_factor = if latency > Duration::ZERO {
            1.0 / (1.0 + latency.as_secs_f64())
        } else {
            1.0
        };
        
        let successful_pings = self.successful_pings.load(Ordering::Acquire);
        let failed_pings = self.failed_pings.load(Ordering::Acquire);
        let success_rate = (successful_pings + 1) as f64 / (successful_pings + failed_pings + 1) as f64;
        
        // Physics factors
        let potential_energy = self.get_potential_energy();
        let entropy_contribution = self.get_entropy_contribution();
        let physics_factor = (potential_energy + (1.0 - entropy_contribution)) / 2.0;
        
        // Consensus factors
        let validator_score = self.get_validator_score().await;
        let stake_amount = self.get_stake_amount().await;
        let consensus_factor = (validator_score + stake_amount) / 2.0;
        
        // Weighted combination
        let quality = (reputation_factor * 0.25) +
                     (latency_factor * 0.20) +
                     (success_rate * 0.20) +
                     (physics_factor * 0.15) +
                     (consensus_factor * 0.20);
        
        quality.clamp(0.0, 1.0)
    }

    pub async fn should_evict(&self) -> bool {
        let reputation = self.reputation.load(Ordering::Acquire);
        
        // Check reputation threshold
        if reputation <= -80 {
            return true;
        }
        
        // Check failure rate
        let successful_pings = self.successful_pings.load(Ordering::Acquire);
        let failed_pings = self.failed_pings.load(Ordering::Acquire);
        let total_pings = successful_pings + failed_pings;
        
        if total_pings > 10 {
            let failure_rate = failed_pings as f64 / total_pings as f64;
            if failure_rate > 0.7 {
                return true;
            }
        }
        
        // Check if banned
        let is_banned = self.is_banned.read().await;
        if *is_banned {
            let ban_expiry = self.ban_expiry.read().await;
            if ban_expiry.is_none() || ban_expiry.unwrap() > SystemTime::now() {
                return true;
            }
        }
        
        // Check physics stability
        let entropy_contribution = self.get_entropy_contribution();
        let potential_energy = self.get_potential_energy();
        if entropy_contribution > 0.9 && potential_energy < 0.1 {
            return true;
        }
        
        false
    }

    pub async fn get_connection_weight(&self) -> f64 {
        let quality = self.calculate_quality_score().await;
        
        // Enhanced with physics model
        let force_vector = self.force_vector.read().await;
        let physics_weight = force_vector.attraction - force_vector.repulsion;
        let normalized_physics = (physics_weight + 1.0) / 2.0; // Normalize to [0,1]
        
        // Combine quality and physics
        let combined = (quality * 0.7) + (normalized_physics * 0.3);
        
        combined.clamp(0.0, 1.0)
    }

    pub fn get_age(&self) -> Duration {
        let last_updated = self.last_updated.blocking_read();
        SystemTime::now().duration_since(*last_updated).unwrap_or(Duration::ZERO)
    }

    pub async fn is_stale(&self, max_age: Duration) -> bool {
        self.get_age() > max_age
    }

    pub fn get_reputation(&self) -> i32 {
        self.reputation.load(Ordering::Acquire)
    }

    pub fn get_latency(&self) -> Duration {
        // For non-async contexts, we use blocking read
        let latency = self.latency.blocking_read();
        *latency
    }

    pub fn get_potential_energy(&self) -> f64 {
        let potential_energy = self.potential_energy.blocking_read();
        *potential_energy
    }

    pub fn get_entropy_contribution(&self) -> f64 {
        let entropy_contribution = self.entropy_contribution.blocking_read();
        *entropy_contribution
    }

    pub async fn get_validator_score(&self) -> f64 {
        let validator_score = self.validator_score.read().await;
        *validator_score
    }

    pub async fn get_stake_amount(&self) -> f64 {
        let stake_amount = self.stake_amount.read().await;
        *stake_amount
    }

    pub async fn ban(&self, reason: String, duration: Option<Duration>) {
        {
            let mut is_banned = self.is_banned.write().await;
            *is_banned = true;
        }
        {
            let mut ban_reason = self.ban_reason.write().await;
            *ban_reason = reason;
        }
        {
            let mut ban_expiry = self.ban_expiry.write().await;
            *ban_expiry = duration.map(|d| SystemTime::now() + d);
        }
        
        self.state = ConnectionState::Banned;
        self.increment_version();
    }

    pub async fn unban(&self) {
        {
            let mut is_banned = self.is_banned.write().await;
            *is_banned = false;
        }
        {
            let mut ban_reason = self.ban_reason.write().await;
            ban_reason.clear();
        }
        {
            let mut ban_expiry = self.ban_expiry.write().await;
            *ban_expiry = None;
        }
        
        if self.state == ConnectionState::Banned {
            self.state = ConnectionState::Disconnected;
        }
        
        self.increment_version();
    }

    pub async fn set_connected(&self) {
        let now = SystemTime::now();
        self.state = ConnectionState::Connected;
        self.connected_at = Some(now);
        self.last_seen = now;
        self.connection_count.fetch_add(1, Ordering::AcqRel);
        
        {
            let mut last_updated = self.last_updated.write().await;
            *last_updated = now;
        }
        
        self.increment_version();
    }

    pub async fn set_disconnected(&self) {
        let now = SystemTime::now();
        self.state = ConnectionState::Disconnected;
        self.disconnected_at = Some(now);
        
        {
            let mut last_updated = self.last_updated.write().await;
            *last_updated = now;
        }
        
        self.increment_version();
    }

    fn increment_version(&self) {
        self.version.fetch_add(1, Ordering::AcqRel);
    }

    pub fn get_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    // Getters for atomic counters
    pub fn get_connection_count(&self) -> u32 {
        self.connection_count.load(Ordering::Acquire)
    }

    pub fn get_failed_attempts(&self) -> u32 {
        self.failed_attempts.load(Ordering::Acquire)
    }

    pub fn get_successful_pings(&self) -> u32 {
        self.successful_pings.load(Ordering::Acquire)
    }

    pub fn get_failed_pings(&self) -> u32 {
        self.failed_pings.load(Ordering::Acquire)
    }

    pub fn get_bytes_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::Acquire)
    }

    pub fn get_bytes_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Acquire)
    }

    pub fn get_messages_sent(&self) -> u64 {
        self.messages_sent.load(Ordering::Acquire)
    }

    pub fn get_messages_received(&self) -> u64 {
        self.messages_received.load(Ordering::Acquire)
    }
}

impl Clone for PeerInfo {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id.clone(),
            public_key: self.public_key.clone(),
            address: self.address.clone(),
            port: self.port,
            protocol: self.protocol,
            transport: self.transport,
            version: self.version.clone(),
            capabilities: self.capabilities.clone(),
            user_agent: self.user_agent.clone(),
            network_id: self.network_id,
            state: self.state,
            last_seen: self.last_seen,
            first_seen: self.first_seen,
            connected_at: self.connected_at,
            disconnected_at: self.disconnected_at,
            reputation: Arc::new(AtomicI32::new(self.reputation.load(Ordering::Acquire))),
            latency: Arc::new(RwLock::new(*self.latency.blocking_read())),
            response_time: Arc::new(RwLock::new(*self.response_time.blocking_read())),
            uptime: Arc::new(RwLock::new(*self.uptime.blocking_read())),
            connection_count: Arc::new(AtomicU32::new(self.connection_count.load(Ordering::Acquire))),
            failed_attempts: Arc::new(AtomicU32::new(self.failed_attempts.load(Ordering::Acquire))),
            successful_pings: Arc::new(AtomicU32::new(self.successful_pings.load(Ordering::Acquire))),
            failed_pings: Arc::new(AtomicU32::new(self.failed_pings.load(Ordering::Acquire))),
            bytes_sent: Arc::new(AtomicU64::new(self.bytes_sent.load(Ordering::Acquire))),
            bytes_received: Arc::new(AtomicU64::new(self.bytes_received.load(Ordering::Acquire))),
            messages_sent: Arc::new(AtomicU64::new(self.messages_sent.load(Ordering::Acquire))),
            messages_received: Arc::new(AtomicU64::new(self.messages_received.load(Ordering::Acquire))),
            potential_energy: Arc::new(RwLock::new(*self.potential_energy.blocking_read())),
            kinetic_energy: Arc::new(RwLock::new(*self.kinetic_energy.blocking_read())),
            entropy_contribution: Arc::new(RwLock::new(*self.entropy_contribution.blocking_read())),
            force_vector: Arc::new(RwLock::new(self.force_vector.blocking_read().clone())),
            validator_score: Arc::new(RwLock::new(*self.validator_score.blocking_read())),
            stake_amount: Arc::new(RwLock::new(*self.stake_amount.blocking_read())),
            voting_power: Arc::new(RwLock::new(*self.voting_power.blocking_read())),
            is_bootstrap: self.is_bootstrap,
            is_validator: Arc::new(RwLock::new(*self.is_validator.blocking_read())),
            is_banned: Arc::new(RwLock::new(*self.is_banned.blocking_read())),
            ban_reason: Arc::new(RwLock::new(self.ban_reason.blocking_read().clone())),
            ban_expiry: Arc::new(RwLock::new(*self.ban_expiry.blocking_read())),
            country_code: Arc::new(RwLock::new(self.country_code.blocking_read().clone())),
            region: Arc::new(RwLock::new(self.region.blocking_read().clone())),
            asn: Arc::new(RwLock::new(*self.asn.blocking_read())),
            coordinates: Arc::new(RwLock::new(self.coordinates.blocking_read().clone())),
            max_connections: self.max_connections,
            rate_limit: self.rate_limit,
            memory_usage: Arc::new(RwLock::new(*self.memory_usage.blocking_read())),
            last_updated: Arc::new(RwLock::new(*self.last_updated.blocking_read())),
            version: Arc::new(AtomicU64::new(self.version.load(Ordering::Acquire))),
        }
    }
}

impl std::fmt::Display for PeerInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let node_id_short = if self.node_id.len() >= 8 {
            &self.node_id[..8]
        } else {
            &self.node_id
        };
        
        let reputation = self.reputation.load(Ordering::Acquire);
        
        // For display, we'll use a blocking read for quality score (simplified)
        let quality = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.calculate_quality_score().await
            })
        });
        
        write!(
            f,
            "Peer[{}@{}:{}, Rep:{}, State:{:?}, Quality:{:.3}]",
            node_id_short, self.address, self.port, reputation, self.state, quality
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_peer_creation() {
        let peer = PeerInfo::new(
            "node123".to_string(),
            "192.168.1.1".to_string(),
            8080,
            ProtocolType::Tcp,
        ).unwrap();

        assert_eq!(peer.node_id, "node123");
        assert_eq!(peer.address, "192.168.1.1");
        assert_eq!(peer.port, 8080);
        assert_eq!(peer.state, ConnectionState::Disconnected);
        assert_eq!(peer.get_reputation(), 50);
    }

    #[tokio::test]
    async fn test_peer_validation() {
        // Test invalid node ID
        let result = PeerInfo::new(
            "".to_string(),
            "192.168.1.1".to_string(),
            8080,
            ProtocolType::Tcp,
        );
        assert!(result.is_err());

        // Test invalid port
        let result = PeerInfo::new(
            "node123".to_string(),
            "192.168.1.1".to_string(),
            0,
            ProtocolType::Tcp,
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_reputation_updates() {
        let peer = PeerInfo::new(
            "node123".to_string(),
            "192.168.1.1".to_string(),
            8080,
            ProtocolType::Tcp,
        ).unwrap();

        // Test positive reputation update
        let new_rep = peer.update_reputation(10);
        assert_eq!(new_rep, 60);
        assert_eq!(peer.get_reputation(), 60);

        // Test negative reputation update
        let new_rep = peer.update_reputation(-20);
        assert_eq!(new_rep, 40);
        assert_eq!(peer.get_reputation(), 40);

        // Test clamping at upper bound
        let new_rep = peer.update_reputation(100);
        assert_eq!(new_rep, 100);
        assert_eq!(peer.get_reputation(), 100);

        // Test clamping at lower bound
        let new_rep = peer.update_reputation(-300);
        assert_eq!(new_rep, -100);
        assert_eq!(peer.get_reputation(), -100);
    }

    #[tokio::test]
    async fn test_quality_score_calculation() {
        let peer = PeerInfo::new(
            "node123".to_string(),
            "192.168.1.1".to_string(),
            8080,
            ProtocolType::Tcp,
        ).unwrap();

        let quality = peer.calculate_quality_score().await;
        assert!(quality >= 0.0 && quality <= 1.0);
    }

    #[tokio::test]
    async fn test_ban_management() {
        let peer = PeerInfo::new(
            "node123".to_string(),
            "192.168.1.1".to_string(),
            8080,
            ProtocolType::Tcp,
        ).unwrap();

        // Test banning
        peer.ban("test reason".to_string(), Some(Duration::from_secs(3600))).await;
        assert_eq!(peer.state, ConnectionState::Banned);

        // Test unbanning
        peer.unban().await;
        assert_eq!(peer.state, ConnectionState::Disconnected);
    }
}