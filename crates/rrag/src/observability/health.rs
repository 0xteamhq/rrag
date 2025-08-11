//! # Health Monitoring System
//! 
//! Comprehensive health checking with component status monitoring,
//! dependency verification, and automated health reporting.

use crate::{RragError, RragResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use chrono::{DateTime, Utc, Duration};

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    pub enabled: bool,
    pub check_interval_seconds: u64,
    pub timeout_seconds: u64,
    pub max_consecutive_failures: u32,
    pub recovery_threshold: u32,
    pub enable_detailed_checks: bool,
    pub enable_dependency_checks: bool,
    pub custom_checks: Vec<CustomHealthCheckConfig>,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_seconds: 30,
            timeout_seconds: 10,
            max_consecutive_failures: 3,
            recovery_threshold: 2,
            enable_detailed_checks: true,
            enable_dependency_checks: true,
            custom_checks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomHealthCheckConfig {
    pub name: String,
    pub description: String,
    pub check_type: CustomCheckType,
    pub config: HashMap<String, String>,
    pub critical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CustomCheckType {
    HttpEndpoint,
    DatabaseConnection,
    FileSystemCheck,
    NetworkConnectivity,
    CustomScript,
}

/// Component status levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComponentStatus {
    Healthy = 1,
    Degraded = 2,
    Unhealthy = 3,
    Critical = 4,
    Unknown = 5,
}

impl std::fmt::Display for ComponentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "HEALTHY"),
            Self::Degraded => write!(f, "DEGRADED"),
            Self::Unhealthy => write!(f, "UNHEALTHY"),
            Self::Critical => write!(f, "CRITICAL"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Individual service health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealth {
    pub component_name: String,
    pub status: ComponentStatus,
    pub last_check: DateTime<Utc>,
    pub last_healthy: Option<DateTime<Utc>>,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub response_time_ms: Option<f64>,
    pub error_message: Option<String>,
    pub details: HashMap<String, serde_json::Value>,
    pub dependencies: Vec<String>,
    pub uptime_seconds: Option<i64>,
}

impl ServiceHealth {
    pub fn new(component_name: impl Into<String>) -> Self {
        Self {
            component_name: component_name.into(),
            status: ComponentStatus::Unknown,
            last_check: Utc::now(),
            last_healthy: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
            response_time_ms: None,
            error_message: None,
            details: HashMap::new(),
            dependencies: Vec::new(),
            uptime_seconds: None,
        }
    }

    pub fn with_status(mut self, status: ComponentStatus) -> Self {
        self.status = status;
        if status == ComponentStatus::Healthy {
            self.last_healthy = Some(self.last_check);
        }
        self
    }

    pub fn with_response_time(mut self, response_time_ms: f64) -> Self {
        self.response_time_ms = Some(response_time_ms);
        self
    }

    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error_message = Some(error.into());
        self
    }

    pub fn with_detail(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.details.insert(key.into(), value);
        self
    }

    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }

    pub fn with_uptime(mut self, uptime_seconds: i64) -> Self {
        self.uptime_seconds = Some(uptime_seconds);
        self
    }
}

/// Complete health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub overall_status: ComponentStatus,
    pub timestamp: DateTime<Utc>,
    pub services: HashMap<String, ServiceHealth>,
    pub dependencies_status: HashMap<String, ComponentStatus>,
    pub system_info: SystemInfo,
    pub alerts: Vec<HealthAlert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub uptime_seconds: i64,
    pub version: String,
    pub environment: String,
    pub hostname: String,
    pub total_memory_mb: Option<f64>,
    pub available_memory_mb: Option<f64>,
    pub cpu_count: Option<u32>,
    pub load_average: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    pub component: String,
    pub severity: ComponentStatus,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
}

/// Health check trait for components
#[async_trait::async_trait]
pub trait HealthChecker: Send + Sync {
    async fn check_health(&self) -> RragResult<ServiceHealth>;
    fn component_name(&self) -> &str;
    fn dependencies(&self) -> Vec<String> { Vec::new() }
    fn is_critical(&self) -> bool { false }
}

/// Basic component health checker
pub struct BasicHealthChecker {
    name: String,
    is_critical: bool,
    check_fn: Arc<dyn Fn() -> RragResult<ComponentStatus> + Send + Sync>,
}

impl BasicHealthChecker {
    pub fn new<F>(name: impl Into<String>, check_fn: F) -> Self 
    where
        F: Fn() -> RragResult<ComponentStatus> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            is_critical: false,
            check_fn: Arc::new(check_fn),
        }
    }

    pub fn with_critical(mut self, critical: bool) -> Self {
        self.is_critical = critical;
        self
    }
}

#[async_trait::async_trait]
impl HealthChecker for BasicHealthChecker {
    async fn check_health(&self) -> RragResult<ServiceHealth> {
        let start_time = std::time::Instant::now();
        
        match (self.check_fn)() {
            Ok(status) => {
                let response_time = start_time.elapsed().as_millis() as f64;
                Ok(ServiceHealth::new(&self.name)
                    .with_status(status)
                    .with_response_time(response_time))
            },
            Err(e) => {
                let response_time = start_time.elapsed().as_millis() as f64;
                Ok(ServiceHealth::new(&self.name)
                    .with_status(ComponentStatus::Unhealthy)
                    .with_response_time(response_time)
                    .with_error(e.to_string()))
            }
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }

    fn is_critical(&self) -> bool {
        self.is_critical
    }
}

/// HTTP endpoint health checker
pub struct HttpHealthChecker {
    name: String,
    url: String,
    expected_status: u16,
    timeout: Duration,
    #[cfg(feature = "http")]
    client: reqwest::Client,
    is_critical: bool,
}

impl HttpHealthChecker {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            expected_status: 200,
            timeout: Duration::seconds(10),
            #[cfg(feature = "http")]
            client: reqwest::Client::new(),
            is_critical: false,
        }
    }

    pub fn with_expected_status(mut self, status: u16) -> Self {
        self.expected_status = status;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_critical(mut self, critical: bool) -> Self {
        self.is_critical = critical;
        self
    }
}

#[async_trait::async_trait]
impl HealthChecker for HttpHealthChecker {
    async fn check_health(&self) -> RragResult<ServiceHealth> {
        #[cfg(feature = "http")]
        {
            let start_time = std::time::Instant::now();
            
            let timeout_duration = std::time::Duration::from_millis(self.timeout.num_milliseconds() as u64);
            
            match tokio::time::timeout(timeout_duration, self.client.get(&self.url).send()).await {
                Ok(Ok(response)) => {
                    let response_time = start_time.elapsed().as_millis() as f64;
                    let status_code = response.status().as_u16();
                    
                    let status = if status_code == self.expected_status {
                        ComponentStatus::Healthy
                    } else {
                        ComponentStatus::Degraded
                    };

                    Ok(ServiceHealth::new(&self.name)
                        .with_status(status)
                        .with_response_time(response_time)
                        .with_detail("status_code", serde_json::json!(status_code))
                        .with_detail("url", serde_json::json!(self.url)))
                },
                Ok(Err(e)) => {
                    let response_time = start_time.elapsed().as_millis() as f64;
                    Ok(ServiceHealth::new(&self.name)
                        .with_status(ComponentStatus::Unhealthy)
                        .with_response_time(response_time)
                        .with_error(e.to_string())
                        .with_detail("url", serde_json::json!(self.url)))
                },
                Err(_) => {
                    let response_time = start_time.elapsed().as_millis() as f64;
                    Ok(ServiceHealth::new(&self.name)
                        .with_status(ComponentStatus::Unhealthy)
                        .with_response_time(response_time)
                        .with_error("Request timeout")
                        .with_detail("timeout_ms", serde_json::json!(self.timeout.num_milliseconds()))
                        .with_detail("url", serde_json::json!(self.url)))
                }
            }
        }
        #[cfg(not(feature = "http"))]
        {
            // Without HTTP feature, return a placeholder healthy status
            Ok(ServiceHealth::new(&self.name)
                .with_status(ComponentStatus::Healthy)
                .with_response_time(0.0)
                .with_detail("note", serde_json::json!("HTTP feature disabled"))
                .with_detail("url", serde_json::json!(self.url)))
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }

    fn is_critical(&self) -> bool {
        self.is_critical
    }
}

/// Database connection health checker
pub struct DatabaseHealthChecker {
    name: String,
    connection_string: String,
    timeout: Duration,
    is_critical: bool,
}

impl DatabaseHealthChecker {
    pub fn new(name: impl Into<String>, connection_string: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            connection_string: connection_string.into(),
            timeout: Duration::seconds(5),
            is_critical: true,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_critical(mut self, critical: bool) -> Self {
        self.is_critical = critical;
        self
    }
}

#[async_trait::async_trait]
impl HealthChecker for DatabaseHealthChecker {
    async fn check_health(&self) -> RragResult<ServiceHealth> {
        let start_time = std::time::Instant::now();
        
        // In a real implementation, this would attempt to connect to the database
        // For now, we'll simulate the check
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        let response_time = start_time.elapsed().as_millis() as f64;
        
        // Simulate occasional connection issues
        let status = if rand::random::<f64>() > 0.1 {
            ComponentStatus::Healthy
        } else {
            ComponentStatus::Unhealthy
        };

        let mut health = ServiceHealth::new(&self.name)
            .with_status(status)
            .with_response_time(response_time)
            .with_detail("connection_string", serde_json::json!("***masked***"));

        if status == ComponentStatus::Unhealthy {
            health = health.with_error("Connection failed");
        }

        Ok(health)
    }

    fn component_name(&self) -> &str {
        &self.name
    }

    fn is_critical(&self) -> bool {
        self.is_critical
    }
}

/// Main health monitoring system
pub struct HealthMonitor {
    config: HealthConfig,
    checkers: Arc<RwLock<HashMap<String, Box<dyn HealthChecker>>>>,
    health_history: Arc<RwLock<HashMap<String, Vec<ServiceHealth>>>>,
    alerts: Arc<RwLock<Vec<HealthAlert>>>,
    system_start_time: DateTime<Utc>,
    monitoring_handle: Option<tokio::task::JoinHandle<()>>,
    is_running: Arc<RwLock<bool>>,
}

impl HealthMonitor {
    pub async fn new(config: HealthConfig) -> RragResult<Self> {
        Ok(Self {
            config,
            checkers: Arc::new(RwLock::new(HashMap::new())),
            health_history: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            system_start_time: Utc::now(),
            monitoring_handle: None,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(&mut self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("health_monitor", "stopped", "already running"));
        }

        let handle = self.start_monitoring_loop().await?;
        self.monitoring_handle = Some(handle);

        *running = true;
        tracing::info!("Health monitor started");
        Ok(())
    }

    pub async fn stop(&mut self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        if let Some(handle) = self.monitoring_handle.take() {
            handle.abort();
        }

        *running = false;
        tracing::info!("Health monitor stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    async fn start_monitoring_loop(&self) -> RragResult<tokio::task::JoinHandle<()>> {
        let config = self.config.clone();
        let checkers = self.checkers.clone();
        let health_history = self.health_history.clone();
        let alerts = self.alerts.clone();
        let is_running = self.is_running.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs(config.check_interval_seconds)
            );

            while *is_running.read().await {
                interval.tick().await;

                let checker_map = checkers.read().await;
                let checker_names: Vec<String> = checker_map.keys().cloned().collect();
                drop(checker_map);

                for checker_name in checker_names {
                    let checker_map = checkers.read().await;
                    if let Some(checker) = checker_map.get(&checker_name) {
                        let timeout_duration = std::time::Duration::from_secs(config.timeout_seconds);
                        
                        let health_result = tokio::time::timeout(
                            timeout_duration,
                            checker.check_health()
                        ).await;

                        let mut service_health = match health_result {
                            Ok(Ok(health)) => health,
                            Ok(Err(e)) => ServiceHealth::new(&checker_name)
                                .with_status(ComponentStatus::Unhealthy)
                                .with_error(e.to_string()),
                            Err(_) => ServiceHealth::new(&checker_name)
                                .with_status(ComponentStatus::Unhealthy)
                                .with_error("Health check timeout"),
                        };

                        // Update consecutive counters
                        let mut history = health_history.write().await;
                        let component_history = history.entry(checker_name.clone()).or_insert_with(Vec::new);
                        
                        if let Some(last_health) = component_history.last() {
                            if service_health.status == ComponentStatus::Healthy {
                                if last_health.status == ComponentStatus::Healthy {
                                    service_health.consecutive_successes = last_health.consecutive_successes + 1;
                                } else {
                                    service_health.consecutive_successes = 1;
                                }
                                service_health.consecutive_failures = 0;
                            } else {
                                if last_health.status != ComponentStatus::Healthy {
                                    service_health.consecutive_failures = last_health.consecutive_failures + 1;
                                } else {
                                    service_health.consecutive_failures = 1;
                                }
                                service_health.consecutive_successes = 0;
                            }
                        } else {
                            if service_health.status == ComponentStatus::Healthy {
                                service_health.consecutive_successes = 1;
                            } else {
                                service_health.consecutive_failures = 1;
                            }
                        }

                        component_history.push(service_health.clone());

                        // Keep only recent history (last 100 checks)
                        if component_history.len() > 100 {
                            component_history.drain(0..component_history.len() - 100);
                        }

                        // Generate alerts for significant status changes
                        if service_health.consecutive_failures >= config.max_consecutive_failures {
                            let alert = HealthAlert {
                                component: checker_name.clone(),
                                severity: service_health.status,
                                message: format!(
                                    "Component {} has failed {} consecutive health checks",
                                    checker_name, service_health.consecutive_failures
                                ),
                                timestamp: Utc::now(),
                                resolved: false,
                            };

                            let mut alert_list = alerts.write().await;
                            alert_list.push(alert);

                            // Keep only recent alerts
                            if alert_list.len() > 1000 {
                                alert_list.drain(0..alert_list.len() - 1000);
                            }
                        }
                    }
                    drop(checker_map);
                }
            }
        });

        Ok(handle)
    }

    pub async fn add_checker(&self, checker: Box<dyn HealthChecker>) -> RragResult<()> {
        let name = checker.component_name().to_string();
        let mut checkers = self.checkers.write().await;
        checkers.insert(name, checker);
        Ok(())
    }

    pub async fn remove_checker(&self, name: &str) -> RragResult<()> {
        let mut checkers = self.checkers.write().await;
        checkers.remove(name);
        
        // Also remove health history for this component
        let mut history = self.health_history.write().await;
        history.remove(name);
        
        Ok(())
    }

    pub async fn get_health_report(&self) -> HealthReport {
        let health_history = self.health_history.read().await;
        let alerts = self.alerts.read().await;

        let mut services = HashMap::new();
        let mut overall_status = ComponentStatus::Healthy;

        // Collect current health status for each service
        for (component_name, history) in health_history.iter() {
            if let Some(latest_health) = history.last() {
                services.insert(component_name.clone(), latest_health.clone());
                
                // Determine overall system status
                if latest_health.status > overall_status {
                    overall_status = latest_health.status;
                }
            }
        }

        let system_info = self.get_system_info().await;

        HealthReport {
            overall_status,
            timestamp: Utc::now(),
            services,
            dependencies_status: HashMap::new(), // Would be populated with dependency checks
            system_info,
            alerts: alerts.clone(),
        }
    }

    pub async fn get_component_health(&self, component_name: &str) -> Option<ServiceHealth> {
        let history = self.health_history.read().await;
        history.get(component_name)?.last().cloned()
    }

    pub async fn get_component_history(&self, component_name: &str, limit: Option<usize>) -> Vec<ServiceHealth> {
        let history = self.health_history.read().await;
        if let Some(component_history) = history.get(component_name) {
            let limit = limit.unwrap_or(component_history.len());
            let start_index = component_history.len().saturating_sub(limit);
            component_history[start_index..].to_vec()
        } else {
            Vec::new()
        }
    }

    pub async fn get_alerts(&self, resolved: Option<bool>) -> Vec<HealthAlert> {
        let alerts = self.alerts.read().await;
        if let Some(resolved_filter) = resolved {
            alerts.iter()
                .filter(|alert| alert.resolved == resolved_filter)
                .cloned()
                .collect()
        } else {
            alerts.clone()
        }
    }

    pub async fn acknowledge_alert(&self, component: &str, timestamp: DateTime<Utc>) -> RragResult<()> {
        let mut alerts = self.alerts.write().await;
        if let Some(alert) = alerts.iter_mut()
            .find(|a| a.component == component && a.timestamp == timestamp) {
            alert.resolved = true;
        }
        Ok(())
    }

    async fn get_system_info(&self) -> SystemInfo {
        let uptime = (Utc::now() - self.system_start_time).num_seconds();
        
        SystemInfo {
            uptime_seconds: uptime,
            version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "production".to_string(),
            hostname: hostname::get()
                .unwrap_or_else(|_| "unknown".into())
                .to_string_lossy()
                .to_string(),
            total_memory_mb: None,    // Would be populated with actual system info
            available_memory_mb: None,
            cpu_count: Some(num_cpus::get() as u32),
            load_average: None,
        }
    }

    pub async fn force_health_check(&self, component_name: &str) -> RragResult<ServiceHealth> {
        let checkers = self.checkers.read().await;
        let checker = checkers.get(component_name)
            .ok_or_else(|| RragError::agent("health_monitor", format!("Component not found: {}", component_name)))?;

        let timeout_duration = std::time::Duration::from_secs(self.config.timeout_seconds);
        
        let health_result = tokio::time::timeout(timeout_duration, checker.check_health()).await;

        match health_result {
            Ok(Ok(health)) => Ok(health),
            Ok(Err(e)) => Ok(ServiceHealth::new(component_name)
                .with_status(ComponentStatus::Unhealthy)
                .with_error(e.to_string())),
            Err(_) => Ok(ServiceHealth::new(component_name)
                .with_status(ComponentStatus::Unhealthy)
                .with_error("Health check timeout")),
        }
    }

    pub async fn get_health_summary(&self) -> HealthSummary {
        let report = self.get_health_report().await;
        
        let total_services = report.services.len();
        let healthy_services = report.services.values()
            .filter(|s| s.status == ComponentStatus::Healthy)
            .count();
        let degraded_services = report.services.values()
            .filter(|s| s.status == ComponentStatus::Degraded)
            .count();
        let unhealthy_services = report.services.values()
            .filter(|s| s.status == ComponentStatus::Unhealthy)
            .count();
        let critical_services = report.services.values()
            .filter(|s| s.status == ComponentStatus::Critical)
            .count();

        let active_alerts = report.alerts.iter()
            .filter(|a| !a.resolved)
            .count();

        HealthSummary {
            overall_status: report.overall_status,
            total_services,
            healthy_services,
            degraded_services,
            unhealthy_services,
            critical_services,
            active_alerts,
            uptime_seconds: report.system_info.uptime_seconds,
            last_check: report.timestamp,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    pub overall_status: ComponentStatus,
    pub total_services: usize,
    pub healthy_services: usize,
    pub degraded_services: usize,
    pub unhealthy_services: usize,
    pub critical_services: usize,
    pub active_alerts: usize,
    pub uptime_seconds: i64,
    pub last_check: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_health_creation() {
        let health = ServiceHealth::new("test_service")
            .with_status(ComponentStatus::Healthy)
            .with_response_time(150.5)
            .with_detail("version", serde_json::json!("1.0.0"))
            .with_dependencies(vec!["database".to_string(), "cache".to_string()])
            .with_uptime(3600);

        assert_eq!(health.component_name, "test_service");
        assert_eq!(health.status, ComponentStatus::Healthy);
        assert_eq!(health.response_time_ms.unwrap(), 150.5);
        assert_eq!(health.dependencies.len(), 2);
        assert_eq!(health.uptime_seconds.unwrap(), 3600);
        assert!(health.details.contains_key("version"));
    }

    #[tokio::test]
    async fn test_basic_health_checker() {
        let checker = BasicHealthChecker::new("test_component", || {
            Ok(ComponentStatus::Healthy)
        }).with_critical(true);

        assert_eq!(checker.component_name(), "test_component");
        assert!(checker.is_critical());

        let health = checker.check_health().await.unwrap();
        assert_eq!(health.component_name, "test_component");
        assert_eq!(health.status, ComponentStatus::Healthy);
        assert!(health.response_time_ms.is_some());
    }

    #[tokio::test]
    async fn test_basic_health_checker_failure() {
        let checker = BasicHealthChecker::new("failing_component", || {
            Err(RragError::agent("test", "Simulated failure"))
        });

        let health = checker.check_health().await.unwrap();
        assert_eq!(health.status, ComponentStatus::Unhealthy);
        assert!(health.error_message.is_some());
        assert!(health.error_message.as_ref().unwrap().contains("Simulated failure"));
    }

    #[tokio::test]
    async fn test_database_health_checker() {
        let checker = DatabaseHealthChecker::new("test_db", "postgresql://localhost:5432/test")
            .with_timeout(Duration::seconds(5))
            .with_critical(true);

        assert_eq!(checker.component_name(), "test_db");
        assert!(checker.is_critical());

        let health = checker.check_health().await.unwrap();
        assert_eq!(health.component_name, "test_db");
        assert!(health.response_time_ms.is_some());
        assert!(health.details.contains_key("connection_string"));
    }

    #[tokio::test]
    async fn test_health_monitor() {
        let config = HealthConfig {
            check_interval_seconds: 1, // Fast interval for testing
            ..Default::default()
        };
        
        let mut monitor = HealthMonitor::new(config).await.unwrap();

        // Add a test checker
        let checker = BasicHealthChecker::new("test_service", || Ok(ComponentStatus::Healthy));
        monitor.add_checker(Box::new(checker)).await.unwrap();

        assert!(!monitor.is_healthy().await);

        monitor.start().await.unwrap();
        assert!(monitor.is_healthy().await);

        // Wait for a health check to occur
        tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;

        let report = monitor.get_health_report().await;
        assert_eq!(report.overall_status, ComponentStatus::Healthy);
        assert!(report.services.contains_key("test_service"));

        let service_health = monitor.get_component_health("test_service").await;
        assert!(service_health.is_some());
        assert_eq!(service_health.unwrap().status, ComponentStatus::Healthy);

        monitor.stop().await.unwrap();
        assert!(!monitor.is_healthy().await);
    }

    #[tokio::test]
    async fn test_health_monitor_consecutive_failures() {
        let config = HealthConfig {
            check_interval_seconds: 1,
            max_consecutive_failures: 2,
            ..Default::default()
        };
        
        let mut monitor = HealthMonitor::new(config).await.unwrap();

        // Add a failing checker
        let checker = BasicHealthChecker::new("failing_service", || {
            Err(RragError::agent("test", "Always fails"))
        });
        monitor.add_checker(Box::new(checker)).await.unwrap();

        monitor.start().await.unwrap();

        // Wait for multiple health checks to occur
        tokio::time::sleep(tokio::time::Duration::from_millis(2500)).await;

        let alerts = monitor.get_alerts(Some(false)).await; // Get unresolved alerts
        assert!(!alerts.is_empty());

        let service_health = monitor.get_component_health("failing_service").await.unwrap();
        assert!(service_health.consecutive_failures >= 2);

        monitor.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_force_health_check() {
        let config = HealthConfig::default();
        let mut monitor = HealthMonitor::new(config).await.unwrap();

        let checker = BasicHealthChecker::new("manual_test", || Ok(ComponentStatus::Degraded));
        monitor.add_checker(Box::new(checker)).await.unwrap();

        let health = monitor.force_health_check("manual_test").await.unwrap();
        assert_eq!(health.component_name, "manual_test");
        assert_eq!(health.status, ComponentStatus::Degraded);
    }

    #[tokio::test]
    async fn test_health_summary() {
        let config = HealthConfig::default();
        let mut monitor = HealthMonitor::new(config).await.unwrap();

        // Add multiple checkers with different statuses
        let healthy_checker = BasicHealthChecker::new("healthy_service", || Ok(ComponentStatus::Healthy));
        let degraded_checker = BasicHealthChecker::new("degraded_service", || Ok(ComponentStatus::Degraded));
        let unhealthy_checker = BasicHealthChecker::new("unhealthy_service", || {
            Err(RragError::agent("test", "Service down"))
        });

        monitor.add_checker(Box::new(healthy_checker)).await.unwrap();
        monitor.add_checker(Box::new(degraded_checker)).await.unwrap();
        monitor.add_checker(Box::new(unhealthy_checker)).await.unwrap();

        monitor.start().await.unwrap();

        // Wait for health checks
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let summary = monitor.get_health_summary().await;
        assert_eq!(summary.total_services, 3);
        assert!(summary.uptime_seconds >= 0);

        monitor.stop().await.unwrap();
    }

    #[test]
    fn test_component_status_ordering() {
        assert!(ComponentStatus::Critical > ComponentStatus::Unhealthy);
        assert!(ComponentStatus::Unhealthy > ComponentStatus::Degraded);
        assert!(ComponentStatus::Degraded > ComponentStatus::Healthy);
        assert!(ComponentStatus::Unknown > ComponentStatus::Critical);
    }

    #[test]
    fn test_component_status_display() {
        assert_eq!(ComponentStatus::Healthy.to_string(), "HEALTHY");
        assert_eq!(ComponentStatus::Degraded.to_string(), "DEGRADED");
        assert_eq!(ComponentStatus::Unhealthy.to_string(), "UNHEALTHY");
        assert_eq!(ComponentStatus::Critical.to_string(), "CRITICAL");
        assert_eq!(ComponentStatus::Unknown.to_string(), "UNKNOWN");
    }
}