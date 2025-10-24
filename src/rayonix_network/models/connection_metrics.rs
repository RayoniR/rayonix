use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionMetrics {
    // Connection identification
    pub connection_id: String,
    pub peer_id: String,
    pub protocol: String,
    
    // Timing and duration
    pub established_at: Instant,
    pub last_activity: Instant,
    pub uptime: Duration,
    
    // Atomic counters for high-frequency updates
    bytes_sent: Arc<AtomicU64>,
    bytes_received: Arc<AtomicU64>,
    messages_sent: Arc<AtomicU64>,
    messages_received: Arc<AtomicU64>,
    errors_encountered: Arc<AtomicU64>,
    retries_attempted: Arc<AtomicU64>,
    success_count: Arc<AtomicU64>,
    failure_count: Arc<AtomicU64>,
    timeout_count: Arc<AtomicU64>,
    
    // Statistics tracking
    latency_stats: Arc<RwLock<RollingStatistics>>,
    rtt_stats: Arc<RwLock<RollingStatistics>>,
    message_size_stats: Arc<RwLock<RollingStatistics>>,
    
    // Bandwidth calculations
    bandwidth_stats: Arc<RwLock<BandwidthCalculator>>,
    
    // Connection quality metrics
    quality_score: Arc<RwLock<f64>>,
    stability_score: Arc<RwLock<f64>>,
    reliability_score: Arc<RwLock<f64>>,
    
    // Physics-inspired metrics
    potential_energy: Arc<RwLock<f64>>,
    kinetic_energy: Arc<RwLock<f64>>,
    entropy_contribution: Arc<RwLock<f64>>,
    force_magnitude: Arc<RwLock<f64>>,
    
    // Resource utilization
    cpu_usage: Arc<RwLock<f64>>,
    memory_usage: Arc<RwLock<u64>>,
    buffer_usage: Arc<RwLock<u64>>,
    
    // Protocol-specific metrics
    tcp_metrics: Arc<RwLock<TcpMetrics>>,
    udp_metrics: Arc<RwLock<UdpMetrics>>,
    ws_metrics: Arc<RwLock<WebSocketMetrics>>,
    
    // Security metrics
    security_metrics: Arc<RwLock<SecurityMetrics>>,
    
    // Version for change tracking
    version: Arc<AtomicU64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionMetricsSnapshot {
    pub connection_id: String,
    pub peer_id: String,
    pub protocol: String,
    pub established_at: Instant,
    pub last_activity: Instant,
    pub uptime: Duration,
    
    // Counters
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub errors_encountered: u64,
    pub retries_attempted: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub timeout_count: u64,
    
    // Quality metrics
    pub quality_score: f64,
    pub stability_score: f64,
    pub reliability_score: f64,
    
    // Physics metrics
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub entropy_contribution: f64,
    pub force_magnitude: f64,
    
    // Performance statistics
    pub average_latency: Duration,
    pub latency_variance: f64,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub latency_percentiles: Vec<f64>,
    
    pub average_rtt: Duration,
    pub rtt_variance: f64,
    
    pub average_message_size: f64,
    pub message_size_variance: f64,
    
    // Bandwidth
    pub bytes_per_second_sent: f64,
    pub bytes_per_second_received: f64,
    
    // Rates
    pub success_rate: f64,
    pub messages_per_second: f64,
    
    // Resource usage
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub buffer_usage: u64,
    
    // Metadata
    pub timestamp: Instant,
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingStatistics {
    values: VecDeque<f64>,
    max_size: usize,
    sum: f64,
    sum_squares: f64,
    min: f64,
    max: f64,
}

impl RollingStatistics {
    pub fn new(max_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(max_size),
            max_size,
            sum: 0.0,
            sum_squares: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }
    
    pub fn add(&mut self, value: f64) {
        if self.values.len() == self.max_size {
            if let Some(removed) = self.values.pop_front() {
                self.sum -= removed;
                self.sum_squares -= removed * removed;
            }
        }
        
        self.values.push_back(value);
        self.sum += value;
        self.sum_squares += value * value;
        
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }
    
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }
    
    pub fn variance(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            let mean = self.mean();
            (self.sum_squares / self.values.len() as f64) - (mean * mean)
        }
    }
    
    pub fn min(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.min
        }
    }
    
    pub fn max(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.max
        }
    }
    
    pub fn percentiles(&self, percentiles: &[f64]) -> Vec<f64> {
        if self.values.is_empty() {
            return vec![0.0; percentiles.len()];
        }
        
        let mut sorted: Vec<f64> = self.values.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        percentiles.iter().map(|&p| {
            let index = (p * (sorted.len() - 1) as f64).round() as usize;
            sorted[index]
        }).collect()
    }
    
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
        self.sum_squares = 0.0;
        self.min = f64::MAX;
        self.max = f64::MIN;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthCalculator {
    windows: VecDeque<BandwidthWindow>,
    current_window: BandwidthWindow,
    window_size: Duration,
    max_windows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthWindow {
    start_time: Instant,
    end_time: Instant,
    bytes_sent: u64,
    bytes_received: u64,
    message_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpMetrics {
    pub segments_sent: u64,
    pub segments_received: u64,
    pub retransmissions: u64,
    pub congestion_window: u32,
    pub rtt_variance: f64,
    pub ssthresh: u32,
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UdpMetrics {
    pub datagrams_sent: u64,
    pub datagrams_received: u64,
    pub datagrams_lost: u64,
    pub jitter: Duration,
    pub reorder_buffer: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMetrics {
    pub frames_sent: u64,
    pub frames_received: u64,
    pub ping_count: u64,
    pub pong_count: u64,
    pub close_count: u64,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub handshakes_completed: u64,
    pub handshakes_failed: u64,
    pub auth_failures: u64,
    pub encryption_failures: u64,
    pub decryption_failures: u64,
    pub replay_attempts: u64,
    pub signature_failures: u64,
}

impl ConnectionMetrics {
    pub fn new(connection_id: String, peer_id: String, protocol: String) -> Self {
        let now = Instant::now();
        
        Self {
            connection_id,
            peer_id,
            protocol,
            established_at: now,
            last_activity: now,
            uptime: Duration::ZERO,
            
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            errors_encountered: Arc::new(AtomicU64::new(0)),
            retries_attempted: Arc::new(AtomicU64::new(0)),
            success_count: Arc::new(AtomicU64::new(0)),
            failure_count: Arc::new(AtomicU64::new(0)),
            timeout_count: Arc::new(AtomicU64::new(0)),
            
            latency_stats: Arc::new(RwLock::new(RollingStatistics::new(1000))),
            rtt_stats: Arc::new(RwLock::new(RollingStatistics::new(1000))),
            message_size_stats: Arc::new(RwLock::new(RollingStatistics::new(5000))),
            
            bandwidth_stats: Arc::new(RwLock::new(BandwidthCalculator::new(
                Duration::from_secs(60),
                10
            ))),
            
            quality_score: Arc::new(RwLock::new(1.0)),
            stability_score: Arc::new(RwLock::new(1.0)),
            reliability_score: Arc::new(RwLock::new(1.0)),
            
            potential_energy: Arc::new(RwLock::new(1.0)),
            kinetic_energy: Arc::new(RwLock::new(0.0)),
            entropy_contribution: Arc::new(RwLock::new(0.0)),
            force_magnitude: Arc::new(RwLock::new(0.0)),
            
            cpu_usage: Arc::new(RwLock::new(0.0)),
            memory_usage: Arc::new(RwLock::new(0)),
            buffer_usage: Arc::new(RwLock::new(0)),
            
            tcp_metrics: Arc::new(RwLock::new(TcpMetrics::default())),
            udp_metrics: Arc::new(RwLock::new(UdpMetrics::default())),
            ws_metrics: Arc::new(RwLock::new(WebSocketMetrics::default())),
            
            security_metrics: Arc::new(RwLock::new(SecurityMetrics::default())),
            
            version: Arc::new(AtomicU64::new(1)),
        }
    }
    
    pub async fn record_message_sent(&self, message_size: u64, latency: Duration) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(message_size, Ordering::Relaxed);
        
        let mut latency_stats = self.latency_stats.write().await;
        latency_stats.add(latency.as_secs_f64());
        drop(latency_stats);
        
        let mut message_size_stats = self.message_size_stats.write().await;
        message_size_stats.add(message_size as f64);
        drop(message_size_stats);
        
        let mut bandwidth_stats = self.bandwidth_stats.write().await;
        bandwidth_stats.record_sent(message_size);
        drop(bandwidth_stats);
        
        self.update_last_activity().await;
        self.update_uptime();
        self.update_quality_metrics().await;
        self.increment_version();
    }
    
    pub async fn record_message_received(&self, message_size: u64) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
        self.bytes_received.fetch_add(message_size, Ordering::Relaxed);
        
        let mut message_size_stats = self.message_size_stats.write().await;
        message_size_stats.add(message_size as f64);
        drop(message_size_stats);
        
        let mut bandwidth_stats = self.bandwidth_stats.write().await;
        bandwidth_stats.record_received(message_size);
        drop(bandwidth_stats);
        
        self.update_last_activity().await;
        self.update_uptime();
        self.increment_version();
    }
    
    pub async fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.update_quality_metrics().await;
        self.increment_version();
    }
    
    pub async fn record_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        self.update_quality_metrics().await;
        self.increment_version();
    }
    
    pub async fn record_error(&self) {
        self.errors_encountered.fetch_add(1, Ordering::Relaxed);
        self.update_quality_metrics().await;
        self.increment_version();
    }
    
    pub async fn record_retry(&self) {
        self.retries_attempted.fetch_add(1, Ordering::Relaxed);
        self.increment_version();
    }
    
    pub async fn record_latency(&self, latency: Duration) {
        let mut latency_stats = self.latency_stats.write().await;
        latency_stats.add(latency.as_secs_f64());
        drop(latency_stats);
        
        self.update_last_activity().await;
        self.update_quality_metrics().await;
        self.increment_version();
    }
    
    pub async fn record_rtt(&self, rtt: Duration) {
        let mut rtt_stats = self.rtt_stats.write().await;
        rtt_stats.add(rtt.as_secs_f64());
        drop(rtt_stats);
        
        self.increment_version();
    }
    
    pub async fn update_physics_metrics(&self, potential: f64, kinetic: f64, entropy: f64, force: f64) {
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
        {
            let mut force_magnitude = self.force_magnitude.write().await;
            *force_magnitude = force;
        }
        
        self.increment_version();
    }
    
    pub async fn update_resource_usage(&self, cpu: f64, memory: u64, buffer: u64) {
        {
            let mut cpu_usage = self.cpu_usage.write().await;
            *cpu_usage = cpu;
        }
        {
            let mut memory_usage = self.memory_usage.write().await;
            *memory_usage = memory;
        }
        {
            let mut buffer_usage = self.buffer_usage.write().await;
            *buffer_usage = buffer;
        }
        
        self.increment_version();
    }
    
    pub async fn get_metrics_snapshot(&self) -> ConnectionMetricsSnapshot {
        let bytes_sent = self.bytes_sent.load(Ordering::Relaxed);
        let bytes_received = self.bytes_received.load(Ordering::Relaxed);
        let messages_sent = self.messages_sent.load(Ordering::Relaxed);
        let messages_received = self.messages_received.load(Ordering::Relaxed);
        let errors_encountered = self.errors_encountered.load(Ordering::Relaxed);
        let retries_attempted = self.retries_attempted.load(Ordering::Relaxed);
        let success_count = self.success_count.load(Ordering::Relaxed);
        let failure_count = self.failure_count.load(Ordering::Relaxed);
        let timeout_count = self.timeout_count.load(Ordering::Relaxed);
        
        let latency_stats = self.latency_stats.read().await;
        let rtt_stats = self.rtt_stats.read().await;
        let message_size_stats = self.message_size_stats.read().await;
        let bandwidth_stats = self.bandwidth_stats.read().await;
        
        let quality_score = *self.quality_score.read().await;
        let stability_score = *self.stability_score.read().await;
        let reliability_score = *self.reliability_score.read().await;
        
        let potential_energy = *self.potential_energy.read().await;
        let kinetic_energy = *self.kinetic_energy.read().await;
        let entropy_contribution = *self.entropy_contribution.read().await;
        let force_magnitude = *self.force_magnitude.read().await;
        
        let cpu_usage = *self.cpu_usage.read().await;
        let memory_usage = *self.memory_usage.read().await;
        let buffer_usage = *self.buffer_usage.read().await;
        
        let version = self.version.load(Ordering::Relaxed);
        
        // Calculate latency statistics
        let average_latency = Duration::from_secs_f64(latency_stats.mean());
        let latency_variance = latency_stats.variance();
        let min_latency = Duration::from_secs_f64(latency_stats.min());
        let max_latency = Duration::from_secs_f64(latency_stats.max());
        let latency_percentiles = latency_stats.percentiles(&[0.5, 0.95, 0.99]);
        
        // Calculate RTT statistics
        let average_rtt = Duration::from_secs_f64(rtt_stats.mean());
        let rtt_variance = rtt_stats.variance();
        
        // Calculate message size statistics
        let average_message_size = message_size_stats.mean();
        let message_size_variance = message_size_stats.variance();
        
        // Calculate bandwidth
        let (bytes_per_second_sent, bytes_per_second_received) = bandwidth_stats.get_current_bandwidth();
        
        // Calculate rates
        let total_operations = success_count + failure_count;
        let success_rate = if total_operations > 0 {
            success_count as f64 / total_operations as f64
        } else {
            0.0
        };
        
        let total_messages = messages_sent + messages_received;
        let messages_per_second = if self.uptime.as_secs() > 0 {
            total_messages as f64 / self.uptime.as_secs_f64()
        } else {
            0.0
        };
        
        ConnectionMetricsSnapshot {
            connection_id: self.connection_id.clone(),
            peer_id: self.peer_id.clone(),
            protocol: self.protocol.clone(),
            established_at: self.established_at,
            last_activity: self.last_activity,
            uptime: self.uptime,
            
            bytes_sent,
            bytes_received,
            messages_sent,
            messages_received,
            errors_encountered,
            retries_attempted,
            success_count,
            failure_count,
            timeout_count,
            
            quality_score,
            stability_score,
            reliability_score,
            
            potential_energy,
            kinetic_energy,
            entropy_contribution,
            force_magnitude,
            
            average_latency,
            latency_variance,
            min_latency,
            max_latency,
            latency_percentiles,
            
            average_rtt,
            rtt_variance,
            
            average_message_size,
            message_size_variance,
            
            bytes_per_second_sent,
            bytes_per_second_received,
            
            success_rate,
            messages_per_second,
            
            cpu_usage,
            memory_usage,
            buffer_usage,
            
            timestamp: Instant::now(),
            version,
        }
    }
    
    pub async fn calculate_quality_score(&self) -> f64 {
        let snapshot = self.get_metrics_snapshot().await;
        
        // Base quality factors
        let latency_factor = if snapshot.average_latency > Duration::ZERO {
            1.0 / (1.0 + snapshot.average_latency.as_secs_f64())
        } else {
            1.0
        };
        
        let success_factor = snapshot.success_rate;
        let stability_factor = self.calculate_stability_factor(&snapshot).await;
        
        // Bandwidth efficiency
        let bandwidth_efficiency = if snapshot.bytes_per_second_sent > 0.0 && snapshot.bytes_per_second_received > 0.0 {
            let total_bytes = snapshot.bytes_per_second_sent + snapshot.bytes_per_second_received;
            if total_bytes > 1_000_000.0 { // 1 MB/s threshold
                0.8
            } else {
                1.0
            }
        } else {
            1.0
        };
        
        // Physics factors
        let physics_factor = (snapshot.potential_energy + (1.0 - snapshot.entropy_contribution)) / 2.0;
        
        // Weighted combination
        let quality = (latency_factor * 0.25) +
                     (success_factor * 0.20) +
                     (stability_factor * 0.15) +
                     (bandwidth_efficiency * 0.15) +
                     (physics_factor * 0.25);
        
        quality.clamp(0.0, 1.0)
    }
    
    async fn calculate_stability_factor(&self, snapshot: &ConnectionMetricsSnapshot) -> f64 {
        // Low latency variance indicates stability
        let latency_stability = if snapshot.latency_variance > 0.0 {
            1.0 / (1.0 + snapshot.latency_variance.sqrt())
        } else {
            1.0
        };
        
        // Low error rate indicates stability
        let total_operations = snapshot.success_count + snapshot.failure_count + snapshot.errors_encountered;
        let error_stability = if total_operations > 0 {
            let error_rate = (snapshot.failure_count + snapshot.errors_encountered) as f64 / total_operations as f64;
            1.0 - error_rate
        } else {
            1.0
        };
        
        // Uptime contributes to stability
        let uptime_stability = (snapshot.uptime.as_secs_f64() / (24.0 * 3600.0)).min(1.0); // Normalize to 24 hours
        
        (latency_stability * 0.4) + (error_stability * 0.4) + (uptime_stability * 0.2)
    }
    
    async fn update_last_activity(&self) {
        // This would require interior mutability pattern
        // For now, we'll handle this through the methods that update activity
    }
    
    fn update_uptime(&self) {
        // Uptime is calculated on-demand in snapshot
    }
    
    async fn update_quality_metrics(&self) {
        let quality_score = self.calculate_quality_score().await;
        let snapshot = self.get_metrics_snapshot().await;
        let stability_score = self.calculate_stability_factor(&snapshot).await;
        let reliability_score = self.calculate_reliability_score(&snapshot).await;
        
        {
            let mut qs = self.quality_score.write().await;
            *qs = quality_score;
        }
        {
            let mut ss = self.stability_score.write().await;
            *ss = stability_score;
        }
        {
            let mut rs = self.reliability_score.write().await;
            *rs = reliability_score;
        }
    }
    
    fn calculate_reliability_score(&self, snapshot: &ConnectionMetricsSnapshot) -> f64 {
        let success_rate = snapshot.success_rate;
        let total_messages = snapshot.messages_sent + snapshot.messages_received;
        let error_rate = if total_messages > 0 {
            snapshot.errors_encountered as f64 / total_messages as f64
        } else {
            0.0
        };
        
        // Recent activity bonus
        let activity_bonus = if snapshot.timestamp.duration_since(snapshot.last_activity) < Duration::from_secs(300) {
            0.1
        } else {
            0.0
        };
        
        let reliability = (success_rate * 0.7) + ((1.0 - error_rate) * 0.2) + activity_bonus;
        
        reliability.clamp(0.0, 1.0)
    }
    
    pub async fn get_bandwidth_usage(&self) -> (f64, f64) {
        let bandwidth_stats = self.bandwidth_stats.read().await;
        bandwidth_stats.get_current_bandwidth()
    }
    
    pub async fn get_latency_stats(&self) -> (Duration, Duration, Duration, f64) {
        let latency_stats = self.latency_stats.read().await;
        (
            Duration::from_secs_f64(latency_stats.mean()),
            Duration::from_secs_f64(latency_stats.min()),
            Duration::from_secs_f64(latency_stats.max()),
            latency_stats.variance(),
        )
    }
    
    pub async fn reset(&self) {
        // Reset atomic counters
        self.bytes_sent.store(0, Ordering::Relaxed);
        self.bytes_received.store(0, Ordering::Relaxed);
        self.messages_sent.store(0, Ordering::Relaxed);
        self.messages_received.store(0, Ordering::Relaxed);
        self.errors_encountered.store(0, Ordering::Relaxed);
        self.retries_attempted.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Relaxed);
        self.timeout_count.store(0, Ordering::Relaxed);
        
        // Reset statistics
        {
            let mut latency_stats = self.latency_stats.write().await;
            latency_stats.reset();
        }
        {
            let mut rtt_stats = self.rtt_stats.write().await;
            rtt_stats.reset();
        }
        {
            let mut message_size_stats = self.message_size_stats.write().await;
            message_size_stats.reset();
        }
        {
            let mut bandwidth_stats = self.bandwidth_stats.write().await;
            bandwidth_stats.reset();
        }
        
        // Reset quality scores
        {
            let mut quality_score = self.quality_score.write().await;
            *quality_score = 1.0;
        }
        {
            let mut stability_score = self.stability_score.write().await;
            *stability_score = 1.0;
        }
        {
            let mut reliability_score = self.reliability_score.write().await;
            *reliability_score = 1.0;
        }
        
        // Reset physics metrics
        {
            let mut potential_energy = self.potential_energy.write().await;
            *potential_energy = 1.0;
        }
        {
            let mut kinetic_energy = self.kinetic_energy.write().await;
            *kinetic_energy = 0.0;
        }
        {
            let mut entropy_contribution = self.entropy_contribution.write().await;
            *entropy_contribution = 0.0;
        }
        {
            let mut force_magnitude = self.force_magnitude.write().await;
            *force_magnitude = 0.0;
        }
        
        self.increment_version();
    }
    
    fn increment_version(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }
    
    // Protocol-specific metric accessors
    pub async fn get_tcp_metrics(&self) -> TcpMetrics {
        self.tcp_metrics.read().await.clone()
    }
    
    pub async fn update_tcp_metrics<F>(&self, updater: F) 
    where
        F: FnOnce(&mut TcpMetrics),
    {
        let mut metrics = self.tcp_metrics.write().await;
        updater(&mut metrics);
        self.increment_version();
    }
    
    pub async fn get_udp_metrics(&self) -> UdpMetrics {
        self.udp_metrics.read().await.clone()
    }
    
    pub async fn update_udp_metrics<F>(&self, updater: F) 
    where
        F: FnOnce(&mut UdpMetrics),
    {
        let mut metrics = self.udp_metrics.write().await;
        updater(&mut metrics);
        self.increment_version();
    }
    
    pub async fn get_ws_metrics(&self) -> WebSocketMetrics {
        self.ws_metrics.read().await.clone()
    }
    
    pub async fn update_ws_metrics<F>(&self, updater: F) 
    where
        F: FnOnce(&mut WebSocketMetrics),
    {
        let mut metrics = self.ws_metrics.write().await;
        updater(&mut metrics);
        self.increment_version();
    }
    
    pub async fn get_security_metrics(&self) -> SecurityMetrics {
        self.security_metrics.read().await.clone()
    }
    
    pub async fn update_security_metrics<F>(&self, updater: F) 
    where
        F: FnOnce(&mut SecurityMetrics),
    {
        let mut metrics = self.security_metrics.write().await;
        updater(&mut metrics);
        self.increment_version();
    }
}

impl BandwidthCalculator {
    pub fn new(window_size: Duration, max_windows: usize) -> Self {
        let now = Instant::now();
        
        Self {
            windows: VecDeque::with_capacity(max_windows),
            current_window: BandwidthWindow {
                start_time: now,
                end_time: now + window_size,
                bytes_sent: 0,
                bytes_received: 0,
                message_count: 0,
            },
            window_size,
            max_windows,
        }
    }
    
    pub fn record_sent(&mut self, bytes: u64) {
        let now = Instant::now();
        self.ensure_current_window(now);
        self.current_window.bytes_sent += bytes;
        self.current_window.message_count += 1;
    }
    
    pub fn record_received(&mut self, bytes: u64) {
        let now = Instant::now();
        self.ensure_current_window(now);
        self.current_window.bytes_received += bytes;
        self.current_window.message_count += 1;
    }
    
    fn ensure_current_window(&mut self, now: Instant) {
        if now > self.current_window.end_time {
            // Move current window to history
            self.windows.push_back(self.current_window.clone());
            
            // Maintain window count limit
            if self.windows.len() > self.max_windows {
                self.windows.pop_front();
            }
            
            self.current_window = BandwidthWindow {
                start_time: now,
                end_time: now + self.window_size,
                bytes_sent: 0,
                bytes_received: 0,
                message_count: 0,
            };
        }
    }
    
    pub fn get_current_bandwidth(&self) -> (f64, f64) {
        let now = Instant::now();
        let window_duration = now.duration_since(self.current_window.start_time).as_secs_f64();
        
        if window_duration > 0.0 {
            let sent = self.current_window.bytes_sent as f64 / window_duration;
            let received = self.current_window.bytes_received as f64 / window_duration;
            (sent, received)
        } else {
            (0.0, 0.0)
        }
    }
    
    pub fn get_average_bandwidth(&self) -> (f64, f64) {
        if self.windows.is_empty() {
            return self.get_current_bandwidth();
        }
        
        let mut total_sent = 0u64;
        let mut total_received = 0u64;
        let mut total_duration = 0.0f64;
        
        for window in &self.windows {
            let window_duration = window.end_time.duration_since(window.start_time).as_secs_f64();
            total_sent += window.bytes_sent;
            total_received += window.bytes_received;
            total_duration += window_duration;
        }
        
        // Include current window
        let now = Instant::now();
        let current_duration = now.duration_since(self.current_window.start_time).as_secs_f64();
        total_sent += self.current_window.bytes_sent;
        total_received += self.current_window.bytes_received;
        total_duration += current_duration;
        
        if total_duration > 0.0 {
            let sent = total_sent as f64 / total_duration;
            let received = total_received as f64 / total_duration;
            (sent, received)
        } else {
            (0.0, 0.0)
        }
    }
    
    pub fn reset(&mut self) {
        self.windows.clear();
        let now = Instant::now();
        self.current_window = BandwidthWindow {
            start_time: now,
            end_time: now + self.window_size,
            bytes_sent: 0,
            bytes_received: 0,
            message_count: 0,
        };
    }
}

// Default implementations
impl Default for TcpMetrics {
    fn default() -> Self {
        Self {
            segments_sent: 0,
            segments_received: 0,
            retransmissions: 0,
            congestion_window: 0,
            rtt_variance: 0.0,
            ssthresh: 0,
            state: "established".to_string(),
        }
    }
}

impl Default for UdpMetrics {
    fn default() -> Self {
        Self {
            datagrams_sent: 0,
            datagrams_received: 0,
            datagrams_lost: 0,
            jitter: Duration::ZERO,
            reorder_buffer: 0,
        }
    }
}

impl Default for WebSocketMetrics {
    fn default() -> Self {
        Self {
            frames_sent: 0,
            frames_received: 0,
            ping_count: 0,
            pong_count: 0,
            close_count: 0,
            compression_ratio: 1.0,
        }
    }
}

impl Default for SecurityMetrics {
    fn default() -> Self {
        Self {
            handshakes_completed: 0,
            handshakes_failed: 0,
            auth_failures: 0,
            encryption_failures: 0,
            decryption_failures: 0,
            replay_attempts: 0,
            signature_failures: 0,
        }
    }
}

impl std::fmt::Display for ConnectionMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // For display purposes, we'll use a simple representation
        // In production, you might want to use the snapshot
        write!(
            f,
            "Metrics[Conn:{}, Peer:{}, Protocol:{}]",
            if self.connection_id.len() >= 8 { &self.connection_id[..8] } else { &self.connection_id },
            if self.peer_id.len() >= 8 { &self.peer_id[..8] } else { &self.peer_id },
            self.protocol
        )
    }
}