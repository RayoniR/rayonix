use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub messages_per_minute: u32,
    pub bandwidth_per_minute: u64, // bytes
    pub connection_attempts_per_minute: u32,
    pub burst_capacity: u32,
    pub enabled: bool,
    pub cleanup_interval_secs: u64,
    pub inactive_timeout_secs: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            messages_per_minute: 1000,
            bandwidth_per_minute: 1024 * 1024, // 1MB
            connection_attempts_per_minute: 10,
            burst_capacity: 100,
            enabled: true,
            cleanup_interval_secs: 300, // 5 minutes
            inactive_timeout_secs: 3600, // 1 hour
        }
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitData {
    pub incoming_message_count: u32,
    pub outgoing_message_count: u32,
    pub incoming_bytes: u64,
    pub outgoing_bytes: u64,
    pub connection_attempts: u32,
    pub last_reset: Instant,
    pub last_activity: Instant,
    pub limited_until: Option<Instant>,
}

impl Default for RateLimitData {
    fn default() -> Self {
        Self {
            incoming_message_count: 0,
            outgoing_message_count: 0,
            incoming_bytes: 0,
            outgoing_bytes: 0,
            connection_attempts: 0,
            last_reset: Instant::now(),
            last_activity: Instant::now(),
            limited_until: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitStats {
    pub total_checked: u64,
    pub total_limited: u64,
    pub current_active_limits: usize,
}

impl Default for RateLimitStats {
    fn default() -> Self {
        Self {
            total_checked: 0,
            total_limited: 0,
            current_active_limits: 0,
        }
    }
}

#[derive(Debug)]
pub enum RateLimitType {
    IncomingMessages,
    OutgoingMessages,
    IncomingBandwidth,
    OutgoingBandwidth,
    ConnectionRate,
}

#[derive(Debug)]
pub struct RateLimitResult {
    pub allowed: bool,
    pub limit_type: Option<String>,
    pub retry_after: Option<Duration>,
}

pub struct RateLimiter {
    config: RateLimitConfig,
    rate_limits: Arc<RwLock<HashMap<String, RateLimitData>>>,
    global_stats: Arc<RwLock<RateLimitStats>>,
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
            global_stats: Arc::new(RwLock::new(RateLimitStats::default())),
            cleanup_handle: None,
        }
    }

    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.config.enabled {
            info!("Rate limiter disabled");
            return Ok(());
        }

        let rate_limits = Arc::clone(&self.rate_limits);
        let config = self.config.clone();
        let stats = Arc::clone(&self.global_stats);

        let handle = tokio::spawn(async move {
            Self::cleanup_loop(rate_limits, config, stats).await;
        });

        self.cleanup_handle = Some(handle);
        info!("Rate limiter started");
        Ok(())
    }

    pub async fn stop(&mut self) {
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
            let _ = handle.await;
        }
        info!("Rate limiter stopped");
    }

    pub async fn check_rate_limit(
        &self, 
        connection_id: &str, 
        message_size: u64
    ) -> RateLimitResult {
        self.check_limit(
            connection_id,
            message_size,
            RateLimitType::IncomingMessages,
            RateLimitType::IncomingBandwidth,
        ).await
    }

    pub async fn check_outgoing_rate_limit(
        &self, 
        connection_id: &str, 
        message_size: u64
    ) -> RateLimitResult {
        self.check_limit(
            connection_id,
            message_size,
            RateLimitType::OutgoingMessages,
            RateLimitType::OutgoingBandwidth,
        ).await
    }

    pub async fn check_connection_rate_limit(&self, address: &str) -> RateLimitResult {
        if !self.config.enabled {
            return RateLimitResult {
                allowed: true,
                limit_type: None,
                retry_after: None,
            };
        }

        let mut stats = self.global_stats.write().await;
        stats.total_checked += 1;

        let mut rate_limits = self.rate_limits.write().await;

        // Check if currently limited
        if let Some(limit_data) = rate_limits.get(address) {
            if let Some(limited_until) = limit_data.limited_until {
                if Instant::now() < limited_until {
                    stats.total_limited += 1;
                    let retry_after = limited_until.duration_since(Instant::now());
                    warn!("Connection rate limit exceeded for {}", address);
                    return RateLimitResult {
                        allowed: false,
                        limit_type: Some("connection_rate".to_string()),
                        retry_after: Some(retry_after),
                    };
                }
            }
        }

        // Get or create limit data
        let limit_data = rate_limits.entry(address.to_string())
            .or_insert_with(RateLimitData::default);
        
        self.reset_if_needed(limit_data);

        // Check connection attempt limit
        if limit_data.connection_attempts >= self.config.connection_attempts_per_minute {
            // Apply temporary limit (1 minute)
            let limited_until = Instant::now() + Duration::from_secs(60);
            limit_data.limited_until = Some(limited_until);
            stats.total_limited += 1;
            
            let retry_after = limited_until.duration_since(Instant::now());
            warn!("Connection rate limit exceeded for {}", address);
            
            return RateLimitResult {
                allowed: false,
                limit_type: Some("connection_rate".to_string()),
                retry_after: Some(retry_after),
            };
        }

        // Update counters
        limit_data.connection_attempts += 1;
        limit_data.last_activity = Instant::now();

        RateLimitResult {
            allowed: true,
            limit_type: None,
            retry_after: None,
        }
    }

    async fn check_limit(
        &self,
        connection_id: &str,
        message_size: u64,
        message_limit_type: RateLimitType,
        bandwidth_limit_type: RateLimitType,
    ) -> RateLimitResult {
        if !self.config.enabled {
            return RateLimitResult {
                allowed: true,
                limit_type: None,
                retry_after: None,
            };
        }

        let mut stats = self.global_stats.write().await;
        stats.total_checked += 1;

        let mut rate_limits = self.rate_limits.write().await;

        // Check if currently limited
        if let Some(limit_data) = rate_limits.get(connection_id) {
            if let Some(limited_until) = limit_data.limited_until {
                if Instant::now() < limited_until {
                    stats.total_limited += 1;
                    let retry_after = limited_until.duration_since(Instant::now());
                    return RateLimitResult {
                        allowed: false,
                        limit_type: Some("temporary_limit".to_string()),
                        retry_after: Some(retry_after),
                    };
                }
            }
        }

        // Get or create limit data
        let limit_data = rate_limits.entry(connection_id.to_string())
            .or_insert_with(RateLimitData::default);
        
        self.reset_if_needed(limit_data);

        // Check message count limit with burst capacity
        let max_messages = self.config.messages_per_minute + self.config.burst_capacity;
        
        let message_count = match message_limit_type {
            RateLimitType::IncomingMessages => {
                let count = limit_data.incoming_message_count;
                if count >= max_messages {
                    return self.apply_limit(
                        connection_id, 
                        limit_data, 
                        &mut stats, 
                        "message_count"
                    );
                }
                limit_data.incoming_message_count += 1;
                count
            }
            RateLimitType::OutgoingMessages => {
                let count = limit_data.outgoing_message_count;
                if count >= max_messages {
                    return self.apply_limit(
                        connection_id, 
                        limit_data, 
                        &mut stats, 
                        "message_count"
                    );
                }
                limit_data.outgoing_message_count += 1;
                count
            }
            _ => 0,
        };

        // Check bandwidth limit
        let max_bandwidth = self.config.bandwidth_per_minute;
        
        match bandwidth_limit_type {
            RateLimitType::IncomingBandwidth => {
                if limit_data.incoming_bytes + message_size > max_bandwidth {
                    return self.apply_limit(
                        connection_id, 
                        limit_data, 
                        &mut stats, 
                        "bandwidth"
                    );
                }
                limit_data.incoming_bytes += message_size;
            }
            RateLimitType::OutgoingBandwidth => {
                if limit_data.outgoing_bytes + message_size > max_bandwidth {
                    return self.apply_limit(
                        connection_id, 
                        limit_data, 
                        &mut stats, 
                        "bandwidth"
                    );
                }
                limit_data.outgoing_bytes += message_size;
            }
            _ => {}
        }

        limit_data.last_activity = Instant::now();

        RateLimitResult {
            allowed: true,
            limit_type: None,
            retry_after: None,
        }
    }

    fn reset_if_needed(&self, limit_data: &mut RateLimitData) {
        let now = Instant::now();
        if now.duration_since(limit_data.last_reset) >= Duration::from_secs(60) {
            limit_data.incoming_message_count = 0;
            limit_data.outgoing_message_count = 0;
            limit_data.incoming_bytes = 0;
            limit_data.outgoing_bytes = 0;
            limit_data.connection_attempts = 0;
            limit_data.last_reset = now;
            limit_data.limited_until = None;
        }
    }

    fn apply_limit(
        &self,
        connection_id: &str,
        limit_data: &mut RateLimitData,
        stats: &mut RateLimitStats,
        limit_type: &str,
    ) -> RateLimitResult {
        // Set limited until time (30 seconds penalty)
        let limited_until = Instant::now() + Duration::from_secs(30);
        limit_data.limited_until = Some(limited_until);
        stats.total_limited += 1;

        let retry_after = limited_until.duration_since(Instant::now());
        
        warn!(
            "Rate limit exceeded for {}: {} limit. Limited for 30 seconds.",
            connection_id, limit_type
        );

        RateLimitResult {
            allowed: false,
            limit_type: Some(limit_type.to_string()),
            retry_after: Some(retry_after),
        }
    }

    pub async fn remove_connection(&self, connection_id: &str) {
        let mut rate_limits = self.rate_limits.write().await;
        if rate_limits.remove(connection_id).is_some() {
            debug!("Removed rate limit tracking for {}", connection_id);
        }
    }

    pub async fn get_connection_stats(&self, connection_id: &str) -> Option<RateLimitData> {
        let rate_limits = self.rate_limits.read().await;
        rate_limits.get(connection_id).cloned()
    }

    pub async fn get_global_stats(&self) -> RateLimitStats {
        let stats = self.global_stats.read().await;
        let rate_limits = self.rate_limits.read().await;
        
        RateLimitStats {
            total_checked: stats.total_checked,
            total_limited: stats.total_limited,
            current_active_limits: rate_limits.len(),
        }
    }

    pub async fn reset_connection_limits(&self, connection_id: &str) {
        let mut rate_limits = self.rate_limits.write().await;
        if rate_limits.contains_key(connection_id) {
            rate_limits.insert(connection_id.to_string(), RateLimitData::default());
            info!("Reset rate limits for {}", connection_id);
        }
    }

    pub fn is_connection_limited(&self, connection_id: &str) -> tokio::sync::RwLockReadGuard<'_, HashMap<String, RateLimitData>> {
        self.rate_limits.read()
    }

    async fn cleanup_loop(
        rate_limits: Arc<RwLock<HashMap<String, RateLimitData>>>,
        config: RateLimitConfig,
        stats: Arc<RwLock<RateLimitStats>>,
    ) {
        let mut interval = time::interval(Duration::from_secs(config.cleanup_interval_secs));
        
        loop {
            interval.tick().await;
            if let Err(e) = Self::cleanup_old_entries(&rate_limits, &config, &stats).await {
                error!("Cleanup error: {}", e);
            }
        }
    }

    async fn cleanup_old_entries(
        rate_limits: &Arc<RwLock<HashMap<String, RateLimitData>>>,
        config: &RateLimitConfig,
        stats: &Arc<RwLock<RateLimitStats>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut rate_limits_map = rate_limits.write().await;
        let now = Instant::now();
        let inactive_threshold = Duration::from_secs(config.inactive_timeout_secs);

        let initial_count = rate_limits_map.len();
        
        rate_limits_map.retain(|_, limit_data| {
            now.duration_since(limit_data.last_activity) <= inactive_threshold
        });

        let removed_count = initial_count - rate_limits_map.len();
        if removed_count > 0 {
            debug!("Cleaned up {} inactive rate limit entries", removed_count);
            
            // Update stats
            let mut stats_guard = stats.write().await;
            stats_guard.current_active_limits = rate_limits_map.len();
        }

        Ok(())
    }
}

impl Drop for RateLimiter {
    fn drop(&mut self) {
        if self.config.enabled {
            let _ = tokio::runtime::Handle::try_current().map(|handle| {
                handle.spawn(async move {
                    if let Some(handle) = self.cleanup_handle.take() {
                        handle.abort();
                    }
                });
            });
        }
    }
}