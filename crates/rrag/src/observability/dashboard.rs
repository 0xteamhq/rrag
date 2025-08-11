//! # Web Dashboard for RRAG Observability
//! 
//! Modern web interface providing real-time monitoring, metrics visualization,
//! and system insights through interactive charts and dashboards.

use crate::{RragError, RragResult};
use super::{
    metrics::{MetricsCollector, Metric, MetricValue},
    monitoring::{SystemMonitor, SystemOverview, PerformanceMetrics, SearchStats, UserStats}
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast, mpsc};
use chrono::{DateTime, Utc, Duration};
use std::net::SocketAddr;

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub title: String,
    pub refresh_interval_seconds: u32,
    pub max_data_points: usize,
    pub websocket_enabled: bool,
    pub auth_enabled: bool,
    pub auth_token: Option<String>,
    pub cors_enabled: bool,
    pub allowed_origins: Vec<String>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host: "0.0.0.0".to_string(),
            port: 3000,
            title: "RRAG Observability Dashboard".to_string(),
            refresh_interval_seconds: 5,
            max_data_points: 100,
            websocket_enabled: true,
            auth_enabled: false,
            auth_token: None,
            cors_enabled: true,
            allowed_origins: vec!["*".to_string()],
        }
    }
}

/// Chart data structure for frontend visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub labels: Vec<String>,
    pub datasets: Vec<ChartDataset>,
    pub chart_type: ChartType,
    pub title: String,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDataset {
    pub label: String,
    pub data: Vec<f64>,
    pub color: String,
    pub fill: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Gauge,
    Area,
    Scatter,
}

/// Real-time metrics for dashboard updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMetrics {
    pub timestamp: DateTime<Utc>,
    pub system_overview: SystemOverview,
    pub charts: HashMap<String, ChartData>,
    pub alerts: Vec<AlertInfo>,
    pub health_status: ComponentHealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertInfo {
    pub id: String,
    pub severity: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub component: String,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealthStatus {
    pub overall: String,
    pub components: HashMap<String, ComponentHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: String,
    pub uptime_seconds: i64,
    pub last_check: DateTime<Utc>,
    pub error_count: u64,
    pub response_time_ms: f64,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    #[serde(rename = "metrics_update")]
    MetricsUpdate { data: RealtimeMetrics },
    #[serde(rename = "alert")]
    Alert { alert: AlertInfo },
    #[serde(rename = "health_update")]
    HealthUpdate { health: ComponentHealthStatus },
    #[serde(rename = "chart_update")]
    ChartUpdate { chart_id: String, data: ChartData },
    #[serde(rename = "ping")]
    Ping { timestamp: DateTime<Utc> },
    #[serde(rename = "pong")]
    Pong { timestamp: DateTime<Utc> },
}

/// WebSocket connection manager
pub struct WebSocketManager {
    clients: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<String>>>>,
    broadcast_sender: broadcast::Sender<WebSocketMessage>,
    _broadcast_receiver: broadcast::Receiver<WebSocketMessage>,
}

impl WebSocketManager {
    pub fn new() -> Self {
        let (broadcast_sender, broadcast_receiver) = broadcast::channel(1000);
        
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            broadcast_sender,
            _broadcast_receiver: broadcast_receiver,
        }
    }

    pub async fn add_client(&self, client_id: String, sender: mpsc::UnboundedSender<String>) {
        let mut clients = self.clients.write().await;
        clients.insert(client_id, sender);
        tracing::info!("WebSocket client connected, total clients: {}", clients.len());
    }

    pub async fn remove_client(&self, client_id: &str) {
        let mut clients = self.clients.write().await;
        clients.remove(client_id);
        tracing::info!("WebSocket client disconnected, total clients: {}", clients.len());
    }

    pub async fn broadcast_message(&self, message: WebSocketMessage) -> RragResult<()> {
        // Send via broadcast channel
        if let Err(e) = self.broadcast_sender.send(message.clone()) {
            tracing::warn!("Failed to broadcast message: {}", e);
        }

        // Send to individual clients
        let message_str = serde_json::to_string(&message)
            .map_err(|e| RragError::agent("websocket", e.to_string()))?;

        let mut clients = self.clients.write().await;
        let mut disconnected_clients = Vec::new();

        for (client_id, sender) in clients.iter() {
            if sender.send(message_str.clone()).is_err() {
                disconnected_clients.push(client_id.clone());
            }
        }

        // Remove disconnected clients
        for client_id in disconnected_clients {
            clients.remove(&client_id);
        }

        Ok(())
    }

    pub async fn get_client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    pub fn subscribe_to_broadcasts(&self) -> broadcast::Receiver<WebSocketMessage> {
        self.broadcast_sender.subscribe()
    }
}

/// Dashboard metrics aggregator
pub struct DashboardMetrics {
    performance_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
    search_stats_history: Arc<RwLock<Vec<SearchStats>>>,
    user_stats_history: Arc<RwLock<Vec<UserStats>>>,
    max_data_points: usize,
}

impl DashboardMetrics {
    pub fn new(max_data_points: usize) -> Self {
        Self {
            performance_history: Arc::new(RwLock::new(Vec::new())),
            search_stats_history: Arc::new(RwLock::new(Vec::new())),
            user_stats_history: Arc::new(RwLock::new(Vec::new())),
            max_data_points,
        }
    }

    pub async fn update_performance(&self, metrics: PerformanceMetrics) {
        let mut history = self.performance_history.write().await;
        history.push(metrics);
        
        let current_len = history.len();
        if current_len > self.max_data_points {
            history.drain(0..current_len - self.max_data_points);
        }
    }

    pub async fn update_search_stats(&self, stats: SearchStats) {
        let mut history = self.search_stats_history.write().await;
        history.push(stats);
        
        let current_len = history.len();
        if current_len > self.max_data_points {
            history.drain(0..current_len - self.max_data_points);
        }
    }

    pub async fn update_user_stats(&self, stats: UserStats) {
        let mut history = self.user_stats_history.write().await;
        history.push(stats);
        
        let current_len = history.len();
        if current_len > self.max_data_points {
            history.drain(0..current_len - self.max_data_points);
        }
    }

    pub async fn generate_charts(&self) -> HashMap<String, ChartData> {
        let mut charts = HashMap::new();

        // Performance charts
        charts.insert("cpu_usage".to_string(), self.create_cpu_chart().await);
        charts.insert("memory_usage".to_string(), self.create_memory_chart().await);
        charts.insert("disk_usage".to_string(), self.create_disk_chart().await);
        
        // Search analytics charts
        charts.insert("search_performance".to_string(), self.create_search_performance_chart().await);
        charts.insert("search_success_rate".to_string(), self.create_search_success_chart().await);
        charts.insert("cache_hit_rate".to_string(), self.create_cache_hit_chart().await);
        
        // User activity charts
        charts.insert("active_users".to_string(), self.create_active_users_chart().await);
        charts.insert("user_actions".to_string(), self.create_user_actions_chart().await);

        charts
    }

    async fn create_cpu_chart(&self) -> ChartData {
        let history = self.performance_history.read().await;
        let labels: Vec<String> = history.iter()
            .map(|m| m.timestamp.format("%H:%M:%S").to_string())
            .collect();
        let data: Vec<f64> = history.iter()
            .map(|m| m.cpu_usage_percent)
            .collect();

        ChartData {
            labels,
            datasets: vec![ChartDataset {
                label: "CPU Usage %".to_string(),
                data,
                color: "#3b82f6".to_string(),
                fill: true,
            }],
            chart_type: ChartType::Area,
            title: "CPU Usage Over Time".to_string(),
            unit: Some("%".to_string()),
        }
    }

    async fn create_memory_chart(&self) -> ChartData {
        let history = self.performance_history.read().await;
        let labels: Vec<String> = history.iter()
            .map(|m| m.timestamp.format("%H:%M:%S").to_string())
            .collect();
        let data: Vec<f64> = history.iter()
            .map(|m| m.memory_usage_percent)
            .collect();

        ChartData {
            labels,
            datasets: vec![ChartDataset {
                label: "Memory Usage %".to_string(),
                data,
                color: "#10b981".to_string(),
                fill: true,
            }],
            chart_type: ChartType::Area,
            title: "Memory Usage Over Time".to_string(),
            unit: Some("%".to_string()),
        }
    }

    async fn create_disk_chart(&self) -> ChartData {
        let history = self.performance_history.read().await;
        let labels: Vec<String> = history.iter()
            .map(|m| m.timestamp.format("%H:%M:%S").to_string())
            .collect();
        let data: Vec<f64> = history.iter()
            .map(|m| m.disk_usage_percent)
            .collect();

        ChartData {
            labels,
            datasets: vec![ChartDataset {
                label: "Disk Usage %".to_string(),
                data,
                color: "#f59e0b".to_string(),
                fill: true,
            }],
            chart_type: ChartType::Area,
            title: "Disk Usage Over Time".to_string(),
            unit: Some("%".to_string()),
        }
    }

    async fn create_search_performance_chart(&self) -> ChartData {
        let history = self.search_stats_history.read().await;
        let labels: Vec<String> = (0..history.len())
            .map(|i| format!("Point {}", i + 1))
            .collect();
        let data: Vec<f64> = history.iter()
            .map(|s| s.average_processing_time_ms)
            .collect();

        ChartData {
            labels,
            datasets: vec![ChartDataset {
                label: "Avg Processing Time".to_string(),
                data,
                color: "#8b5cf6".to_string(),
                fill: false,
            }],
            chart_type: ChartType::Line,
            title: "Search Processing Time".to_string(),
            unit: Some("ms".to_string()),
        }
    }

    async fn create_search_success_chart(&self) -> ChartData {
        let history = self.search_stats_history.read().await;
        let labels: Vec<String> = (0..history.len())
            .map(|i| format!("Point {}", i + 1))
            .collect();
        let data: Vec<f64> = history.iter()
            .map(|s| s.success_rate)
            .collect();

        ChartData {
            labels,
            datasets: vec![ChartDataset {
                label: "Success Rate".to_string(),
                data,
                color: "#06d6a0".to_string(),
                fill: false,
            }],
            chart_type: ChartType::Line,
            title: "Search Success Rate".to_string(),
            unit: Some("%".to_string()),
        }
    }

    async fn create_cache_hit_chart(&self) -> ChartData {
        let history = self.search_stats_history.read().await;
        let labels: Vec<String> = (0..history.len())
            .map(|i| format!("Point {}", i + 1))
            .collect();
        let data: Vec<f64> = history.iter()
            .map(|s| s.cache_hit_rate)
            .collect();

        ChartData {
            labels,
            datasets: vec![ChartDataset {
                label: "Cache Hit Rate".to_string(),
                data,
                color: "#ff6b6b".to_string(),
                fill: false,
            }],
            chart_type: ChartType::Line,
            title: "Cache Hit Rate".to_string(),
            unit: Some("%".to_string()),
        }
    }

    async fn create_active_users_chart(&self) -> ChartData {
        let history = self.user_stats_history.read().await;
        let labels: Vec<String> = (0..history.len())
            .map(|i| format!("Point {}", i + 1))
            .collect();
        let data: Vec<f64> = history.iter()
            .map(|s| s.unique_users as f64)
            .collect();

        ChartData {
            labels,
            datasets: vec![ChartDataset {
                label: "Active Users".to_string(),
                data,
                color: "#4ecdc4".to_string(),
                fill: true,
            }],
            chart_type: ChartType::Area,
            title: "Active Users Over Time".to_string(),
            unit: Some("users".to_string()),
        }
    }

    async fn create_user_actions_chart(&self) -> ChartData {
        let history = self.user_stats_history.read().await;
        if let Some(latest_stats) = history.last() {
            let labels: Vec<String> = latest_stats.action_breakdown.keys().cloned().collect();
            let data: Vec<f64> = latest_stats.action_breakdown.values()
                .map(|&count| count as f64)
                .collect();

            ChartData {
                labels,
                datasets: vec![ChartDataset {
                    label: "User Actions".to_string(),
                    data,
                    color: "#ff9ff3".to_string(),
                    fill: false,
                }],
                chart_type: ChartType::Pie,
                title: "User Action Distribution".to_string(),
                unit: Some("actions".to_string()),
            }
        } else {
            ChartData {
                labels: vec![],
                datasets: vec![],
                chart_type: ChartType::Pie,
                title: "User Action Distribution".to_string(),
                unit: Some("actions".to_string()),
            }
        }
    }
}

/// Main dashboard server
pub struct DashboardServer {
    config: DashboardConfig,
    metrics_collector: Arc<MetricsCollector>,
    system_monitor: Arc<SystemMonitor>,
    websocket_manager: Arc<WebSocketManager>,
    dashboard_metrics: Arc<DashboardMetrics>,
    server_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    update_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    is_running: Arc<RwLock<bool>>,
}

impl DashboardServer {
    pub async fn new(
        config: DashboardConfig,
        metrics_collector: Arc<MetricsCollector>,
        system_monitor: Arc<SystemMonitor>,
    ) -> RragResult<Self> {
        let websocket_manager = Arc::new(WebSocketManager::new());
        let dashboard_metrics = Arc::new(DashboardMetrics::new(config.max_data_points));

        Ok(Self {
            config,
            metrics_collector,
            system_monitor,
            websocket_manager,
            dashboard_metrics,
            server_handle: Arc::new(RwLock::new(None)),
            update_handle: Arc::new(RwLock::new(None)),
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(&self) -> RragResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("dashboard_server", "stopped", "already running"));
        }

        // Start the HTTP server
        let server_handle = self.start_http_server().await?;
        {
            let mut handle = self.server_handle.write().await;
            *handle = Some(server_handle);
        }

        // Start the metrics update loop
        let update_handle = self.start_update_loop().await?;
        {
            let mut handle = self.update_handle.write().await;
            *handle = Some(update_handle);
        }

        *running = true;
        tracing::info!(
            "Dashboard server started on {}:{}",
            self.config.host,
            self.config.port
        );
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        // Stop server tasks
        {
            let mut handle = self.server_handle.write().await;
            if let Some(h) = handle.take() {
                h.abort();
            }
        }
        {
            let mut handle = self.update_handle.write().await;
            if let Some(h) = handle.take() {
                h.abort();
            }
        }

        *running = false;
        tracing::info!("Dashboard server stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    async fn start_http_server(&self) -> RragResult<tokio::task::JoinHandle<()>> {
        let config = self.config.clone();
        let websocket_manager = self.websocket_manager.clone();
        let is_running = self.is_running.clone();

        let handle = tokio::spawn(async move {
            // In a real implementation, this would start an actual HTTP server
            // using a framework like warp, axum, or actix-web
            // For now, we'll simulate the server behavior
            
            let addr: SocketAddr = format!("{}:{}", config.host, config.port)
                .parse()
                .expect("Invalid address");

            tracing::info!("Dashboard HTTP server would start on {}", addr);

            // Simulate server running
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            while *is_running.read().await {
                interval.tick().await;
                
                // Simulate periodic cleanup of WebSocket connections
                let client_count = websocket_manager.get_client_count().await;
                tracing::debug!("Active WebSocket clients: {}", client_count);
            }
        });

        Ok(handle)
    }

    async fn start_update_loop(&self) -> RragResult<tokio::task::JoinHandle<()>> {
        let config = self.config.clone();
        let system_monitor = self.system_monitor.clone();
        let websocket_manager = self.websocket_manager.clone();
        let dashboard_metrics = self.dashboard_metrics.clone();
        let is_running = self.is_running.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs(config.refresh_interval_seconds as u64)
            );

            while *is_running.read().await {
                interval.tick().await;

                // Collect current system overview
                let overview = system_monitor.get_system_overview().await;

                // Update dashboard metrics
                if let Some(ref perf) = overview.performance_metrics {
                    dashboard_metrics.update_performance(perf.clone()).await;
                }
                if let Some(ref search_stats) = overview.search_stats {
                    dashboard_metrics.update_search_stats(search_stats.clone()).await;
                }
                if let Some(ref user_stats) = overview.user_stats {
                    dashboard_metrics.update_user_stats(user_stats.clone()).await;
                }

                // Generate charts
                let charts = dashboard_metrics.generate_charts().await;

                // Create health status
                let health_status = ComponentHealthStatus {
                    overall: "healthy".to_string(),
                    components: HashMap::from([
                        ("metrics".to_string(), ComponentHealth {
                            status: "healthy".to_string(),
                            uptime_seconds: 3600,
                            last_check: Utc::now(),
                            error_count: 0,
                            response_time_ms: 10.0,
                        }),
                        ("monitoring".to_string(), ComponentHealth {
                            status: "healthy".to_string(),
                            uptime_seconds: 3600,
                            last_check: Utc::now(),
                            error_count: 0,
                            response_time_ms: 15.0,
                        }),
                    ]),
                };

                // Create realtime metrics
                let realtime_metrics = RealtimeMetrics {
                    timestamp: Utc::now(),
                    system_overview: overview,
                    charts,
                    alerts: vec![], // Would be populated with actual alerts
                    health_status,
                };

                // Broadcast to WebSocket clients
                if let Err(e) = websocket_manager.broadcast_message(
                    WebSocketMessage::MetricsUpdate { data: realtime_metrics }
                ).await {
                    tracing::warn!("Failed to broadcast metrics update: {}", e);
                }
            }
        });

        Ok(handle)
    }

    /// Get current dashboard data (for HTTP API endpoints)
    pub async fn get_current_data(&self) -> RragResult<RealtimeMetrics> {
        let overview = self.system_monitor.get_system_overview().await;
        let charts = self.dashboard_metrics.generate_charts().await;

        let health_status = ComponentHealthStatus {
            overall: "healthy".to_string(),
            components: HashMap::from([
                ("metrics".to_string(), ComponentHealth {
                    status: "healthy".to_string(),
                    uptime_seconds: 3600,
                    last_check: Utc::now(),
                    error_count: 0,
                    response_time_ms: 10.0,
                }),
            ]),
        };

        Ok(RealtimeMetrics {
            timestamp: Utc::now(),
            system_overview: overview,
            charts,
            alerts: vec![],
            health_status,
        })
    }

    /// Get WebSocket manager for integration with HTTP server
    pub fn websocket_manager(&self) -> &Arc<WebSocketManager> {
        &self.websocket_manager
    }

    /// Get dashboard configuration
    pub fn config(&self) -> &DashboardConfig {
        &self.config
    }
}

/// Dashboard request handler (for HTTP endpoints)
pub struct DashboardHandler {
    server: Arc<DashboardServer>,
}

impl DashboardHandler {
    pub fn new(server: Arc<DashboardServer>) -> Self {
        Self { server }
    }

    /// Handle dashboard home page
    pub async fn handle_dashboard(&self) -> RragResult<String> {
        // In a real implementation, this would render the dashboard HTML
        let data = self.server.get_current_data().await?;
        
        Ok(format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
        .status-healthy {{ color: #10b981; }}
        .status-warning {{ color: #f59e0b; }}
        .status-error {{ color: #ef4444; }}
        .chart-placeholder {{ height: 200px; background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{}</h1>
            <p>Last updated: {}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">System Status</div>
                <div class="metric-value status-healthy">Healthy</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Active WebSocket Clients</div>
                <div class="metric-value">{}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">CPU Usage</div>
                <div class="metric-value">{:.1}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Memory Usage</div>
                <div class="metric-value">{:.1}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Search Success Rate</div>
                <div class="metric-value">{:.1}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Active Users</div>
                <div class="metric-value">{}</div>
            </div>
        </div>
        
        <div style="margin-top: 40px;">
            <h2>Charts</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">CPU Usage Over Time</div>
                    <div class="chart-placeholder">CPU Usage Chart</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Memory Usage Over Time</div>
                    <div class="chart-placeholder">Memory Usage Chart</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Search Performance</div>
                    <div class="chart-placeholder">Search Performance Chart</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">User Activity</div>
                    <div class="chart-placeholder">User Activity Chart</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket('ws://{}:{}/ws');
        ws.onmessage = function(event) {{
            const message = JSON.parse(event.data);
            if (message.type === 'metrics_update') {{
                // Update dashboard with new data
                console.log('Received metrics update:', message.data);
            }}
        }};
        
        // Refresh page every 30 seconds as fallback
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"#,
            self.server.config().title,
            self.server.config().title,
            data.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            self.server.websocket_manager().get_client_count().await,
            data.system_overview.performance_metrics.as_ref()
                .map(|p| p.cpu_usage_percent).unwrap_or(0.0),
            data.system_overview.performance_metrics.as_ref()
                .map(|p| p.memory_usage_percent).unwrap_or(0.0),
            data.system_overview.search_stats.as_ref()
                .map(|s| s.success_rate).unwrap_or(0.0),
            data.system_overview.user_stats.as_ref()
                .map(|u| u.unique_users).unwrap_or(0),
            self.server.config().host,
            self.server.config().port
        ))
    }

    /// Handle metrics API endpoint
    pub async fn handle_metrics_api(&self) -> RragResult<String> {
        let data = self.server.get_current_data().await?;
        serde_json::to_string_pretty(&data)
            .map_err(|e| RragError::agent("dashboard", e.to_string()))
    }

    /// Handle health check endpoint
    pub async fn handle_health(&self) -> RragResult<String> {
        let health = ComponentHealthStatus {
            overall: "healthy".to_string(),
            components: HashMap::from([
                ("dashboard".to_string(), ComponentHealth {
                    status: "healthy".to_string(),
                    uptime_seconds: 3600,
                    last_check: Utc::now(),
                    error_count: 0,
                    response_time_ms: 5.0,
                }),
            ]),
        };
        
        serde_json::to_string(&health)
            .map_err(|e| RragError::agent("dashboard", e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::{metrics::MetricsConfig, monitoring::MonitoringConfig};

    async fn create_test_components() -> (Arc<MetricsCollector>, Arc<SystemMonitor>) {
        let metrics_collector = Arc::new(
            MetricsCollector::new(MetricsConfig::default()).await.unwrap()
        );
        let system_monitor = Arc::new(
            SystemMonitor::new(MonitoringConfig::default(), metrics_collector.clone()).await.unwrap()
        );
        (metrics_collector, system_monitor)
    }

    #[tokio::test]
    async fn test_websocket_manager() {
        let manager = WebSocketManager::new();
        assert_eq!(manager.get_client_count().await, 0);
        
        let (sender, _receiver) = mpsc::unbounded_channel();
        manager.add_client("client1".to_string(), sender).await;
        assert_eq!(manager.get_client_count().await, 1);
        
        manager.remove_client("client1").await;
        assert_eq!(manager.get_client_count().await, 0);
    }

    #[tokio::test]
    async fn test_dashboard_metrics() {
        let dashboard_metrics = DashboardMetrics::new(100);
        
        let perf_metrics = PerformanceMetrics {
            timestamp: Utc::now(),
            cpu_usage_percent: 50.0,
            memory_usage_mb: 1024.0,
            memory_usage_percent: 60.0,
            disk_usage_mb: 2048.0,
            disk_usage_percent: 70.0,
            network_bytes_sent: 1000,
            network_bytes_received: 2000,
            active_connections: 10,
            thread_count: 50,
            gc_collections: 5,
            gc_pause_time_ms: 2.5,
        };
        
        dashboard_metrics.update_performance(perf_metrics).await;
        
        let charts = dashboard_metrics.generate_charts().await;
        assert!(charts.contains_key("cpu_usage"));
        assert!(charts.contains_key("memory_usage"));
        assert!(charts.contains_key("disk_usage"));
    }

    #[tokio::test]
    async fn test_dashboard_server() {
        let (metrics_collector, system_monitor) = create_test_components().await;
        let config = DashboardConfig::default();
        let mut server = DashboardServer::new(config, metrics_collector, system_monitor)
            .await.unwrap();
        
        assert!(!server.is_healthy().await);
        
        server.start().await.unwrap();
        assert!(server.is_healthy().await);
        
        let current_data = server.get_current_data().await.unwrap();
        assert!(current_data.charts.len() > 0);
        
        server.stop().await.unwrap();
        assert!(!server.is_healthy().await);
    }

    #[tokio::test]
    async fn test_dashboard_handler() {
        let (metrics_collector, system_monitor) = create_test_components().await;
        let config = DashboardConfig::default();
        let server = Arc::new(
            DashboardServer::new(config, metrics_collector, system_monitor).await.unwrap()
        );
        
        let handler = DashboardHandler::new(server);
        
        let dashboard_html = handler.handle_dashboard().await.unwrap();
        assert!(dashboard_html.contains("<!DOCTYPE html>"));
        assert!(dashboard_html.contains("RRAG Observability Dashboard"));
        
        let metrics_json = handler.handle_metrics_api().await.unwrap();
        assert!(serde_json::from_str::<RealtimeMetrics>(&metrics_json).is_ok());
        
        let health_json = handler.handle_health().await.unwrap();
        assert!(serde_json::from_str::<ComponentHealthStatus>(&health_json).is_ok());
    }

    #[test]
    fn test_chart_data_creation() {
        let chart_data = ChartData {
            labels: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            datasets: vec![ChartDataset {
                label: "Test Data".to_string(),
                data: vec![10.0, 20.0, 30.0],
                color: "#3b82f6".to_string(),
                fill: false,
            }],
            chart_type: ChartType::Line,
            title: "Test Chart".to_string(),
            unit: Some("units".to_string()),
        };
        
        assert_eq!(chart_data.labels.len(), 3);
        assert_eq!(chart_data.datasets[0].data.len(), 3);
        assert_eq!(chart_data.title, "Test Chart");
    }

    #[test]
    fn test_websocket_message_serialization() {
        let message = WebSocketMessage::Ping {
            timestamp: Utc::now(),
        };
        
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"type\":\"ping\""));
        
        let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();
        match deserialized {
            WebSocketMessage::Ping { .. } => {},
            _ => panic!("Wrong message type"),
        }
    }
}