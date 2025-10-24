use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::time;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, error, instrument};
use async_trait::async_trait;

use crate::{
    network::Network,
    models::{NetworkMessage, PeerInfo},
    config::{MessageType, ProtocolType, ConnectionState},
    errors::MessageError,
    rate_limiter::{RateLimiter, RateLimitConfig},
};

#[derive(Debug, Clone, Serialize)]
pub struct SendResult {
    pub success: bool,
    pub duration: f64,
    pub bytes_sent: usize,
    pub retry_count: u32,
    pub error: Option<String>,
}

impl SendResult {
    pub fn new(success: bool, duration: f64, bytes_sent: usize) -> Self {
        Self {
            success,
            duration,
            bytes_sent,
            retry_count: 0,
            error: None,
        }
    }
    
    pub fn with_error(error: String, duration: f64) -> Self {
        Self {
            success: false,
            duration,
            bytes_sent: 0,
            retry_count: 0,
            error: Some(error),
        }
    }
}

#[async_trait]
pub trait MessageHandler: Send + Sync {
    async fn handle_message(&self, connection_id: &str, message: &NetworkMessage) -> Result<(), MessageError>;
}

pub struct MessageProcessor {
    network: Arc<Network>,
    handlers: Arc<RwLock<HashMap<MessageType, Vec<Arc<dyn MessageHandler>>>>>,
    rate_limiter: Arc<RateLimiter>,
    send_attempts: Arc<RwLock<HashMap<String, u32>>>, // Track send attempts per connection
    circuit_breakers: Arc<RwLock<HashMap<String, bool>>>, // Circuit breaker pattern
}

impl MessageProcessor {
    pub fn new(network: Arc<Network>) -> Self {
        let rate_config = RateLimitConfig {
            messages_per_minute: 1000,
            bandwidth_per_minute: 1024 * 1024, // 1MB
            ..Default::default()
        };
        
        Self {
            network,
            handlers: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(RateLimiter::new(rate_config)),
            send_attempts: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.rate_limiter.start().await?;
        info!("Message processor started");
        Ok(())
    }

    pub async fn stop(&self) {
        self.rate_limiter.stop().await;
        info!("Message processor stopped");
    }

    #[instrument(skip(self, message))]
    pub async fn process_message(&self, connection_id: &str, message: NetworkMessage) -> Result<(), MessageError> {
        let start_time = Instant::now();
        
        // Update connection activity
        if let Err(e) = self.network.connection_manager.update_connection_activity(connection_id).await {
            warn!("Failed to update connection activity for {}: {}", connection_id, e);
        }

        debug!("Processing {} from {}", message.message_type, connection_id);

        // Call registered handlers
        let handlers = self.handlers.read().await;
        if let Some(handler_list) = handlers.get(&message.message_type) {
            for handler in handler_list {
                if let Err(e) = handler.handle_message(connection_id, &message).await {
                    error!("Handler error for {}: {}", message.message_type, e);
                }
            }
        }

        // Handle specific message types
        match message.message_type {
            MessageType::Ping => self.handle_ping(connection_id, message).await?,
            MessageType::Pong => self.handle_pong(connection_id, message).await?,
            MessageType::PeerList => self.handle_peer_list(connection_id, message).await?,
            MessageType::Handshake => self.handle_handshake(connection_id, message).await?,
            _ => {} // Other message types handled by registered handlers
        }

        let duration = start_time.elapsed().as_secs_f64();
        debug!("Message processing completed in {:.3}s", duration);
        
        Ok(())
    }

    #[instrument(skip(self, message))]
    async fn handle_ping(&self, connection_id: &str, message: NetworkMessage) -> Result<(), MessageError> {
        let pong_message = NetworkMessage {
            message_id: format!("pong_{}", time::Instant::now().elapsed().as_secs_f64()),
            message_type: MessageType::Pong,
            payload: serde_json::json!({
                "original_timestamp": message.payload.get("timestamp")
            }),
            timestamp: time::Instant::now().elapsed().as_secs_f64(),
            ttl: 10,
            signature: None,
            source_node: Some(self.network.node_id.clone()),
            destination_node: None,
            priority: 0,
        };

        let result = self.send_message(connection_id, pong_message).await;
        if !result.success {
            error!("Failed to send pong response to {}: {:?}", connection_id, result.error);
        }

        Ok(())
    }

    #[instrument(skip(self, message))]
    async fn handle_pong(&self, connection_id: &str, message: NetworkMessage) -> Result<(), MessageError> {
        if let Some(original_timestamp) = message.payload.get("original_timestamp").and_then(|v| v.as_f64()) {
            let latency = time::Instant::now().elapsed().as_secs_f64() - original_timestamp;
            
            // Update connection metrics
            self.network.metrics_collector.update_connection_metrics(
                connection_id, 
                None, 
                None, 
                None, 
                None, 
                Some(latency)
            ).await;

            // Update peer info if available
            let mut connections = self.network.connections.write().await;
            if let Some(connection) = connections.get_mut(connection_id) {
                if let Some(peer_info) = &mut connection.peer_info {
                    peer_info.latency = latency;
                    peer_info.last_seen = time::Instant::now().elapsed().as_secs_f64();
                }
            }
        }

        Ok(())
    }

    #[instrument(skip(self, message))]
    async fn handle_peer_list(&self, connection_id: &str, message: NetworkMessage) -> Result<(), MessageError> {
        if let Some(request) = message.payload.get("request").and_then(|v| v.as_bool()) {
            if request {
                // Send our peer list
                let peers_to_share = self.network.peer_discovery.get_best_peers(10).await;
                let peer_data: Vec<serde_json::Value> = peers_to_share.iter().map(|peer| {
                    serde_json::json!({
                        "address": peer.address,
                        "port": peer.port,
                        "protocol": peer.protocol.to_string(),
                        "capabilities": peer.capabilities,
                        "reputation": peer.reputation
                    })
                }).collect();

                let response = NetworkMessage {
                    message_id: format!("peer_resp_{}", time::Instant::now().elapsed().as_secs_f64()),
                    message_type: MessageType::PeerList,
                    payload: serde_json::json!({ "peers": peer_data }),
                    timestamp: time::Instant::now().elapsed().as_secs_f64(),
                    ttl: 10,
                    signature: None,
                    source_node: Some(self.network.node_id.clone()),
                    destination_node: None,
                    priority: 0,
                };

                let result = self.send_message(connection_id, response).await;
                if !result.success {
                    error!("Failed to send peer list response to {}: {:?}", connection_id, result.error);
                }
            } else {
                // Process received peer list
                if let Some(peers_array) = message.payload.get("peers").and_then(|v| v.as_array()) {
                    for peer_data in peers_array {
                        if let Ok(peer_info) = self.parse_peer_info(peer_data) {
                            if let Err(e) = self.network.peer_discovery.add_peer(peer_info).await {
                                debug!("Invalid peer data: {}", e);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[instrument(skip(self, message))]
    async fn handle_handshake(&self, connection_id: &str, message: NetworkMessage) -> Result<(), MessageError> {
        let success = self.network.security_manager.process_handshake(connection_id, &message).await?;
        
        if success {
            let mut connections = self.network.connections.write().await;
            if let Some(connection) = connections.get_mut(connection_id) {
                if let Some(peer_info) = &mut connection.peer_info {
                    peer_info.state = ConnectionState::Ready;
                    peer_info.node_id = message.source_node.clone().unwrap_or_default();
                    peer_info.capabilities = message.payload.get("capabilities")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default();
                    peer_info.version = message.payload.get("version")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "1.0.0".to_string());
                }
            }
        }

        Ok(())
    }

    fn parse_peer_info(&self, peer_data: &serde_json::Value) -> Result<PeerInfo, MessageError> {
        let address = peer_data.get("address")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MessageError::InvalidData("Missing address".to_string()))?;
        
        let port = peer_data.get("port")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| MessageError::InvalidData("Missing port".to_string()))?;
        
        let protocol_str = peer_data.get("protocol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MessageError::InvalidData("Missing protocol".to_string()))?;
        
        let protocol = ProtocolType::from_str(protocol_str)
            .map_err(|_| MessageError::InvalidData(format!("Invalid protocol: {}", protocol_str)))?;

        let capabilities = peer_data.get("capabilities")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        let reputation = peer_data.get("reputation")
            .and_then(|v| v.as_i64())
            .unwrap_or(50) as i32;

        Ok(PeerInfo {
            node_id: "".to_string(), // Will be set during connection
            address: address.to_string(),
            port: port as u16,
            protocol,
            version: "1.0.0".to_string(),
            capabilities,
            last_seen: time::Instant::now().elapsed().as_secs_f64(),
            connection_count: 0,
            failed_attempts: 0,
            reputation,
            latency: 0.0,
            state: ConnectionState::Disconnected,
            public_key: None,
            user_agent: "".to_string(),
            services: 0,
            last_attempt: 0.0,
            next_attempt: 0.0,
            banned_until: None,
        })
    }

    pub async fn register_handler(&self, message_type: MessageType, handler: Arc<dyn MessageHandler>) {
        let mut handlers = self.handlers.write().await;
        handlers.entry(message_type).or_insert_with(Vec::new).push(handler);
        debug!("Registered handler for {}", message_type);
    }

    pub async fn unregister_handler(&self, message_type: MessageType, handler: Arc<dyn MessageHandler>) {
        let mut handlers = self.handlers.write().await;
        if let Some(handler_list) = handlers.get_mut(&message_type) {
            handler_list.retain(|h| !Arc::ptr_eq(h, &handler));
            debug!("Unregistered handler for {}", message_type);
        }
    }

    #[instrument(skip(self, message))]
    pub async fn send_message(&self, connection_id: &str, message: NetworkMessage) -> SendResult {
        let start_time = Instant::now();
        let message_size = self.calculate_message_size(&message);

        // Input validation
        if let Err(validation_error) = self.validate_inputs(connection_id, &message) {
            return SendResult::with_error(validation_error, start_time.elapsed().as_secs_f64());
        }

        // Check circuit breaker
        if self.is_circuit_open(connection_id).await {
            return SendResult::with_error("Circuit breaker open".to_string(), start_time.elapsed().as_secs_f64());
        }

        // Get connection and validate
        let connections = self.network.connections.read().await;
        let connection = match connections.get(connection_id) {
            Some(conn) => conn,
            None => {
                return SendResult::with_error("Connection not found".to_string(), start_time.elapsed().as_secs_f64());
            }
        };

        // Protocol validation - TCP only for P2P
        if connection.protocol != ProtocolType::Tcp {
            return SendResult::with_error(
                format!("Protocol {:?} not supported for P2P", connection.protocol),
                start_time.elapsed().as_secs_f64()
            );
        }

        // Pre-send checks
        if let Err(health_error) = self.perform_pre_send_checks(connection_id, connection).await {
            return SendResult::with_error(health_error, start_time.elapsed().as_secs_f64());
        }

        // Send with retry logic
        let result = self.send_with_retry(connection_id, message, message_size).await;
        let duration = start_time.elapsed().as_secs_f64();

        // Update circuit breaker based on result
        self.update_circuit_breaker(connection_id, result.success).await;

        SendResult {
            duration,
            bytes_sent: if result.success { message_size } else { 0 },
            ..result
        }
    }

    async fn send_with_retry(&self, connection_id: &str, message: NetworkMessage, message_size: usize) -> SendResult {
        let max_retries = 3;
        let mut last_error = None;

        for attempt in 0..=max_retries {
            // Check rate limiting before each attempt
            let rate_limit_result = self.rate_limiter.check_outgoing_rate_limit(connection_id, message_size as u64).await;
            if !rate_limit_result.allowed {
                last_error = Some("Outgoing rate limit exceeded".to_string());
                if attempt == max_retries {
                    break;
                }
                self.wait_for_retry(attempt, connection_id).await;
                continue;
            }

            // Attempt to send
            match self.network.tcp_handler.send_message(connection_id, message.clone()).await {
                Ok(success) => {
                    if success {
                        // Update metrics on success
                        self.update_success_metrics(connection_id, message_size).await;
                        
                        if attempt > 0 {
                            info!("Message send succeeded on retry {} for {}", attempt, connection_id);
                        }
                        
                        return SendResult {
                            success: true,
                            duration: 0.0, // Will be set by caller
                            bytes_sent: message_size,
                            retry_count: attempt as u32,
                            error: None,
                        };
                    } else {
                        last_error = Some("TCP handler returned failure".to_string());
                        if attempt < max_retries {
                            self.wait_for_retry(attempt, connection_id).await;
                        }
                    }
                }
                Err(e) => {
                    last_error = Some(format!("Send error: {}", e));
                    if attempt < max_retries {
                        self.wait_for_retry(attempt, connection_id).await;
                    } else {
                        error!("Send failed after {} attempts for {}: {}", max_retries + 1, connection_id, e);
                    }
                }
            }
        }

        // All retries failed
        self.update_failure_metrics(connection_id, message_size).await;
        SendResult {
            success: false,
            duration: 0.0, // Will be set by caller
            bytes_sent: 0,
            retry_count: max_retries as u32,
            error: last_error,
        }
    }

    async fn wait_for_retry(&self, attempt: u32, connection_id: &str) {
        let delay = self.calculate_retry_delay(attempt);
        warn!(
            "Send failed for {}, retrying in {:.2}s (attempt {})",
            connection_id, delay, attempt + 1
        );
        time::sleep(Duration::from_secs_f64(delay)).await;
    }

    fn calculate_retry_delay(&self, attempt: u32) -> f64 {
        let base_delay = 0.5; // 500ms
        let max_delay = 10.0; // 10 seconds max
        let delay = (base_delay * (2.0f64).powi(attempt as i32)).min(max_delay);
        
        // Add jitter to avoid thundering herd
        let jitter = 0.8 + rand::random::<f64>() * 0.4; // 0.8 to 1.2
        delay * jitter
    }

    fn validate_inputs(&self, connection_id: &str, message: &NetworkMessage) -> Result<(), String> {
        if connection_id.is_empty() {
            return Err("Invalid connection_id".to_string());
        }
        
        if message.message_id.is_empty() {
            return Err("Message missing valid message_id".to_string());
        }
        
        // Check message size limits
        let message_size = self.calculate_message_size(message);
        if message_size > self.network.config.max_message_size {
            return Err(format!("Message too large: {} bytes", message_size));
        }
        
        // Check TTL for gossip messages
        if message.ttl <= 0 {
            return Err("Message TTL expired".to_string());
        }
        
        Ok(())
    }

    fn calculate_message_size(&self, message: &NetworkMessage) -> usize {
        message.message_id.len() 
            + serde_json::to_string(&message.payload).map(|s| s.len()).unwrap_or(0)
            + message.source_node.as_ref().map(|s| s.len()).unwrap_or(0)
            + message.destination_node.as_ref().map(|s| s.len()).unwrap_or(0)
    }

    async fn perform_pre_send_checks(&self, connection_id: &str, connection: &crate::models::Connection) -> Result<(), String> {
        // Check connection health
        if !self.is_connection_healthy(connection_id, connection).await {
            return Err("Connection not healthy".to_string());
        }
        
        // Check peer reputation
        if let Some(peer_info) = &connection.peer_info {
            if peer_info.reputation < -50 {
                return Err("Peer reputation too low".to_string());
            }
        }
        
        Ok(())
    }

    async fn is_connection_healthy(&self, connection_id: &str, connection: &crate::models::Connection) -> bool {
        // Check last activity timeout
        let last_activity = connection.last_activity;
        if Instant::now().duration_since(Instant::now() - Duration::from_secs_f64(last_activity)) 
            > Duration::from_secs(self.network.config.connection_timeout as u64) {
            warn!("Connection {} is stale", connection_id);
            return false;
        }
        
        // Check connection state
        if let Some(peer_info) = &connection.peer_info {
            if peer_info.state != ConnectionState::Ready {
                warn!("Connection {} not in READY state: {:?}", connection_id, peer_info.state);
                return false;
            }
        }
        
        // TCP-specific health check would go here
        true
    }

    async fn update_success_metrics(&self, connection_id: &str, message_size: usize) {
        if let Err(e) = self.network.metrics_collector.update_connection_metrics(
            connection_id, 
            Some(message_size), 
            None, 
            Some(1), 
            None, 
            None
        ).await {
            debug!("Metrics update error: {}", e);
        }
        
        // Reset circuit breaker on success
        let mut circuit_breakers = self.circuit_breakers.write().await;
        circuit_breakers.insert(connection_id.to_string(), false);
    }

    async fn update_failure_metrics(&self, connection_id: &str, message_size: usize) {
        if let Err(e) = self.network.metrics_collector.update_connection_metrics(
            connection_id, 
            None, 
            None, 
            None, 
            Some(1), 
            None
        ).await {
            debug!("Failure metrics error: {}", e);
        }
    }

    async fn is_circuit_open(&self, connection_id: &str) -> bool {
        let circuit_breakers = self.circuit_breakers.read().await;
        circuit_breakers.get(connection_id).copied().unwrap_or(false)
    }

    async fn update_circuit_breaker(&self, connection_id: &str, success: bool) {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        let mut send_attempts = self.send_attempts.write().await;
        
        if success {
            circuit_breakers.insert(connection_id.to_string(), false);
            send_attempts.remove(connection_id);
        } else {
            let fail_count = send_attempts.entry(connection_id.to_string()).or_insert(0);
            *fail_count += 1;
            
            if *fail_count >= 5 { // Open circuit after 5 consecutive failures
                circuit_breakers.insert(connection_id.to_string(), true);
                error!("Circuit breaker opened for {}", connection_id);
            }
        }
    }

    #[instrument(skip(self, message))]
    pub async fn broadcast_message(&self, message: NetworkMessage, exclude: Option<Vec<String>>) -> HashMap<String, SendResult> {
        let exclude = exclude.unwrap_or_default();
        let connections = self.network.connections.read().await;
        let connection_ids: Vec<String> = connections.keys()
            .filter(|id| !exclude.contains(id))
            .cloned()
            .collect();
        
        let mut results = HashMap::new();
        
        // Batch processing for performance
        let batch_size = 10;
        let semaphore = Arc::new(Semaphore::new(5)); // Limit concurrent sends
        
        for chunk in connection_ids.chunks(batch_size) {
            let mut tasks = Vec::new();
            
            for connection_id in chunk {
                let message_clone = message.clone();
                let processor = Arc::new(self); // Assuming self is Arc<MessageProcessor>
                let semaphore = Arc::clone(&semaphore);
                let connection_id = connection_id.clone();
                
                tasks.push(tokio::spawn(async move {
                    let _permit = semaphore.acquire().await;
                    let result = processor.send_message(&connection_id, message_clone).await;
                    (connection_id, result)
                }));
            }
            
            // Wait for batch to complete
            for task in tasks {
                match task.await {
                    Ok((conn_id, result)) => {
                        results.insert(conn_id, result);
                    }
                    Err(e) => {
                        error!("Broadcast task failed: {}", e);
                    }
                }
            }
            
            // Small delay between batches
            if chunk.len() == batch_size {
                time::sleep(Duration::from_millis(50)).await;
            }
        }
        
        results
    }
}

impl Drop for MessageProcessor {
    fn drop(&mut self) {
        // Ensure proper cleanup
        let processor = Arc::new(std::mem::replace(self, Self::new(Arc::clone(&self.network))));
        tokio::spawn(async move {
            processor.stop().await;
        });
    }
}

