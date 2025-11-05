//! # Alerting System
//!
//! Intelligent alerting with threshold monitoring, notification channels,
//! and automated response capabilities for RRAG system health.

use super::metrics::{Metric, MetricValue, MetricsCollector};
use crate::{RragError, RragResult};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub enabled: bool,
    pub evaluation_interval_seconds: u64,
    pub alert_buffer_size: usize,
    pub notification_channels: Vec<NotificationChannelConfig>,
    pub default_severity: AlertSeverity,
    pub alert_grouping_enabled: bool,
    pub alert_grouping_window_minutes: u32,
    pub escalation_enabled: bool,
    pub escalation_delay_minutes: u32,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            evaluation_interval_seconds: 30,
            alert_buffer_size: 1000,
            notification_channels: vec![NotificationChannelConfig {
                name: "console".to_string(),
                channel_type: NotificationChannelType::Console,
                enabled: true,
                config: HashMap::new(),
            }],
            default_severity: AlertSeverity::Medium,
            alert_grouping_enabled: true,
            alert_grouping_window_minutes: 5,
            escalation_enabled: false,
            escalation_delay_minutes: 30,
        }
    }
}

/// Notification channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannelConfig {
    pub name: String,
    pub channel_type: NotificationChannelType,
    pub enabled: bool,
    pub config: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NotificationChannelType {
    Console,
    Email,
    Slack,
    Webhook,
    SMS,
    PagerDuty,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Alert conditions for triggering alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    Threshold {
        metric_name: String,
        operator: ComparisonOperator,
        value: f64,
        duration_minutes: u32,
    },
    RateOfChange {
        metric_name: String,
        operator: ComparisonOperator,
        rate_per_minute: f64,
        window_minutes: u32,
    },
    Anomaly {
        metric_name: String,
        sensitivity: f64,
        baseline_minutes: u32,
    },
    Composite {
        conditions: Vec<AlertCondition>,
        logic: LogicOperator,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicOperator {
    And,
    Or,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub notification_channels: Vec<String>,
    pub tags: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub cooldown_minutes: u32,
    pub auto_resolve: bool,
    pub auto_resolve_after_minutes: Option<u32>,
}

impl AlertRule {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        condition: AlertCondition,
        severity: AlertSeverity,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            condition,
            severity,
            enabled: true,
            notification_channels: vec!["console".to_string()],
            tags: HashMap::new(),
            created_at: now,
            updated_at: now,
            cooldown_minutes: 5,
            auto_resolve: true,
            auto_resolve_after_minutes: Some(30),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_channels(mut self, channels: Vec<String>) -> Self {
        self.notification_channels = channels;
        self
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    pub fn with_cooldown(mut self, minutes: u32) -> Self {
        self.cooldown_minutes = minutes;
        self
    }
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertNotification {
    pub id: String,
    pub rule_id: String,
    pub rule_name: String,
    pub severity: AlertSeverity,
    pub status: AlertStatus,
    pub message: String,
    pub details: HashMap<String, serde_json::Value>,
    pub triggered_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub acknowledged_at: Option<DateTime<Utc>>,
    pub acknowledged_by: Option<String>,
    pub notification_channels: Vec<String>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AlertStatus {
    Triggered,
    Acknowledged,
    Resolved,
    Suppressed,
}

/// Notification channel trait
#[async_trait::async_trait]
pub trait NotificationChannel: Send + Sync {
    async fn send_notification(&self, notification: &AlertNotification) -> RragResult<()>;
    fn channel_type(&self) -> NotificationChannelType;
    fn name(&self) -> &str;
    async fn is_healthy(&self) -> bool;
}

/// Console notification channel
pub struct ConsoleNotificationChannel {
    name: String,
}

impl ConsoleNotificationChannel {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait::async_trait]
impl NotificationChannel for ConsoleNotificationChannel {
    async fn send_notification(&self, notification: &AlertNotification) -> RragResult<()> {
        let status_symbol = match notification.status {
            AlertStatus::Triggered => "🚨",
            AlertStatus::Acknowledged => "✅",
            AlertStatus::Resolved => "✅",
            AlertStatus::Suppressed => "🔇",
        };

        let severity_color = match notification.severity {
            AlertSeverity::Critical => "\x1b[31m", // Red
            AlertSeverity::High => "\x1b[33m",     // Yellow
            AlertSeverity::Medium => "\x1b[36m",   // Cyan
            AlertSeverity::Low => "\x1b[32m",      // Green
        };

        tracing::debug!(
            "{} {}[{}]\x1b[0m {} - {} ({})",
            status_symbol,
            severity_color,
            notification.severity,
            notification.rule_name,
            notification.message,
            notification.triggered_at.format("%Y-%m-%d %H:%M:%S UTC")
        );

        if !notification.details.is_empty() {
            tracing::debug!("   Details: {:?}", notification.details);
        }

        Ok(())
    }

    fn channel_type(&self) -> NotificationChannelType {
        NotificationChannelType::Console
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn is_healthy(&self) -> bool {
        true // Console is always available
    }
}

/// Webhook notification channel
pub struct WebhookNotificationChannel {
    name: String,
    url: String,
    headers: HashMap<String, String>,
    #[cfg(feature = "http")]
    client: reqwest::Client,
}

impl WebhookNotificationChannel {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            headers: HashMap::new(),
            #[cfg(feature = "http")]
            client: reqwest::Client::new(),
        }
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

#[async_trait::async_trait]
impl NotificationChannel for WebhookNotificationChannel {
    async fn send_notification(&self, notification: &AlertNotification) -> RragResult<()> {
        #[cfg(feature = "http")]
        {
            let payload = serde_json::json!({
                "alert_id": notification.id,
                "rule_name": notification.rule_name,
                "severity": notification.severity,
                "status": notification.status,
                "message": notification.message,
                "details": notification.details,
                "triggered_at": notification.triggered_at,
                "tags": notification.tags
            });

            let mut request = self.client.post(&self.url).json(&payload);

            for (key, value) in &self.headers {
                request = request.header(key, value);
            }

            request
                .send()
                .await
                .map_err(|e| RragError::network("webhook_notification", Box::new(e)))?
                .error_for_status()
                .map_err(|e| RragError::network("webhook_notification", Box::new(e)))?;

            Ok(())
        }
        #[cfg(not(feature = "http"))]
        {
            tracing::warn!(
                "HTTP feature not enabled, webhook notification to {} skipped",
                self.url
            );
            Ok(())
        }
    }

    fn channel_type(&self) -> NotificationChannelType {
        NotificationChannelType::Webhook
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn is_healthy(&self) -> bool {
        #[cfg(feature = "http")]
        {
            // Simple health check - try to connect to the webhook URL
            self.client.head(&self.url).send().await.is_ok()
        }
        #[cfg(not(feature = "http"))]
        {
            // Without HTTP feature, assume healthy
            true
        }
    }
}

/// Alert condition evaluator
pub struct AlertEvaluator {
    metrics_history: Arc<RwLock<HashMap<String, Vec<(DateTime<Utc>, f64)>>>>,
    max_history_size: usize,
}

impl AlertEvaluator {
    pub fn new(max_history_size: usize) -> Self {
        Self {
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
            max_history_size,
        }
    }

    pub async fn update_metric(&self, metric_name: String, value: f64) {
        let mut history = self.metrics_history.write().await;
        let entry = history.entry(metric_name).or_insert_with(Vec::new);

        entry.push((Utc::now(), value));

        // Keep only recent data
        if entry.len() > self.max_history_size {
            entry.drain(0..entry.len() - self.max_history_size);
        }
    }

    pub async fn evaluate_condition(&self, condition: &AlertCondition) -> RragResult<bool> {
        match condition {
            AlertCondition::Threshold {
                metric_name,
                operator,
                value,
                duration_minutes,
            } => {
                self.evaluate_threshold(metric_name, operator, *value, *duration_minutes)
                    .await
            }
            AlertCondition::RateOfChange {
                metric_name,
                operator,
                rate_per_minute,
                window_minutes,
            } => {
                self.evaluate_rate_of_change(
                    metric_name,
                    operator,
                    *rate_per_minute,
                    *window_minutes,
                )
                .await
            }
            AlertCondition::Anomaly {
                metric_name,
                sensitivity,
                baseline_minutes,
            } => {
                self.evaluate_anomaly(metric_name, *sensitivity, *baseline_minutes)
                    .await
            }
            AlertCondition::Composite { conditions, logic } => {
                self.evaluate_composite(conditions, logic).await
            }
        }
    }

    async fn evaluate_threshold(
        &self,
        metric_name: &str,
        operator: &ComparisonOperator,
        threshold: f64,
        duration_minutes: u32,
    ) -> RragResult<bool> {
        let history = self.metrics_history.read().await;
        let values = history.get(metric_name).ok_or_else(|| {
            RragError::agent(
                "alert_evaluator",
                format!("Metric not found: {}", metric_name),
            )
        })?;

        if values.is_empty() {
            return Ok(false);
        }

        let cutoff_time = Utc::now() - Duration::minutes(duration_minutes as i64);
        let recent_values: Vec<_> = values
            .iter()
            .filter(|(timestamp, _)| *timestamp >= cutoff_time)
            .map(|(_, value)| *value)
            .collect();

        if recent_values.is_empty() {
            return Ok(false);
        }

        // All values in the duration must satisfy the condition
        Ok(recent_values.iter().all(|&value| match operator {
            ComparisonOperator::GreaterThan => value > threshold,
            ComparisonOperator::LessThan => value < threshold,
            ComparisonOperator::GreaterThanOrEqual => value >= threshold,
            ComparisonOperator::LessThanOrEqual => value <= threshold,
            ComparisonOperator::Equal => (value - threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (value - threshold).abs() >= f64::EPSILON,
        }))
    }

    async fn evaluate_rate_of_change(
        &self,
        metric_name: &str,
        operator: &ComparisonOperator,
        rate_threshold: f64,
        window_minutes: u32,
    ) -> RragResult<bool> {
        let history = self.metrics_history.read().await;
        let values = history.get(metric_name).ok_or_else(|| {
            RragError::agent(
                "alert_evaluator",
                format!("Metric not found: {}", metric_name),
            )
        })?;

        if values.len() < 2 {
            return Ok(false);
        }

        let cutoff_time = Utc::now() - Duration::minutes(window_minutes as i64);
        let recent_values: Vec<_> = values
            .iter()
            .filter(|(timestamp, _)| *timestamp >= cutoff_time)
            .collect();

        if recent_values.len() < 2 {
            return Ok(false);
        }

        let (earliest_time, earliest_value) = recent_values.first().unwrap();
        let (latest_time, latest_value) = recent_values.last().unwrap();

        let time_diff_minutes = (*latest_time - *earliest_time).num_minutes() as f64;
        if time_diff_minutes <= 0.0 {
            return Ok(false);
        }

        let rate_of_change = (latest_value - earliest_value) / time_diff_minutes;

        Ok(match operator {
            ComparisonOperator::GreaterThan => rate_of_change > rate_threshold,
            ComparisonOperator::LessThan => rate_of_change < rate_threshold,
            ComparisonOperator::GreaterThanOrEqual => rate_of_change >= rate_threshold,
            ComparisonOperator::LessThanOrEqual => rate_of_change <= rate_threshold,
            ComparisonOperator::Equal => (rate_of_change - rate_threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (rate_of_change - rate_threshold).abs() >= f64::EPSILON,
        })
    }

    async fn evaluate_anomaly(
        &self,
        metric_name: &str,
        sensitivity: f64,
        baseline_minutes: u32,
    ) -> RragResult<bool> {
        let history = self.metrics_history.read().await;
        let values = history.get(metric_name).ok_or_else(|| {
            RragError::agent(
                "alert_evaluator",
                format!("Metric not found: {}", metric_name),
            )
        })?;

        if values.len() < 10 {
            return Ok(false); // Need sufficient data for anomaly detection
        }

        let cutoff_time = Utc::now() - Duration::minutes(baseline_minutes as i64);
        let baseline_values: Vec<f64> = values
            .iter()
            .filter(|(timestamp, _)| *timestamp >= cutoff_time)
            .map(|(_, value)| *value)
            .collect();

        if baseline_values.len() < 5 {
            return Ok(false);
        }

        // Simple anomaly detection using standard deviation
        let mean = baseline_values.iter().sum::<f64>() / baseline_values.len() as f64;
        let variance = baseline_values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / baseline_values.len() as f64;
        let std_dev = variance.sqrt();

        let current_value = values.last().unwrap().1;
        let z_score = (current_value - mean) / std_dev;

        Ok(z_score.abs() > sensitivity)
    }

    async fn evaluate_composite(
        &self,
        conditions: &[AlertCondition],
        logic: &LogicOperator,
    ) -> RragResult<bool> {
        let mut results = Vec::new();
        for condition in conditions {
            let result = match condition {
                AlertCondition::Threshold {
                    metric_name,
                    operator,
                    value,
                    duration_minutes,
                } => {
                    self.evaluate_threshold(metric_name, operator, *value, *duration_minutes)
                        .await?
                }
                AlertCondition::RateOfChange {
                    metric_name,
                    operator,
                    rate_per_minute,
                    window_minutes,
                } => {
                    self.evaluate_rate_of_change(
                        metric_name,
                        operator,
                        *rate_per_minute,
                        *window_minutes,
                    )
                    .await?
                }
                AlertCondition::Anomaly {
                    metric_name,
                    sensitivity,
                    baseline_minutes,
                } => {
                    self.evaluate_anomaly(metric_name, *sensitivity, *baseline_minutes)
                        .await?
                }
                AlertCondition::Composite { .. } => {
                    // Prevent infinite recursion by limiting depth
                    return Err(RragError::config(
                        "alert_condition",
                        "non-nested composite",
                        "nested composite",
                    ));
                }
            };
            results.push(result);
        }

        Ok(match logic {
            LogicOperator::And => results.iter().all(|&result| result),
            LogicOperator::Or => results.iter().any(|&result| result),
        })
    }
}

/// Main alert manager
pub struct AlertManager {
    config: AlertConfig,
    metrics_collector: Arc<MetricsCollector>,
    alert_rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    active_alerts: Arc<RwLock<HashMap<String, AlertNotification>>>,
    notification_channels: Arc<RwLock<HashMap<String, Box<dyn NotificationChannel>>>>,
    evaluator: Arc<AlertEvaluator>,
    evaluation_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    is_running: Arc<RwLock<bool>>,
}

impl AlertManager {
    pub async fn new(
        config: AlertConfig,
        metrics_collector: Arc<MetricsCollector>,
    ) -> RragResult<Self> {
        let manager = Self {
            config: config.clone(),
            metrics_collector,
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            notification_channels: Arc::new(RwLock::new(HashMap::new())),
            evaluator: Arc::new(AlertEvaluator::new(1000)),
            evaluation_handle: Arc::new(RwLock::new(None)),
            is_running: Arc::new(RwLock::new(false)),
        };

        // Initialize default notification channels
        manager.setup_notification_channels().await?;

        // Add default alert rules
        manager.setup_default_rules().await?;

        Ok(manager)
    }

    async fn setup_notification_channels(&self) -> RragResult<()> {
        let mut channels = self.notification_channels.write().await;

        for channel_config in &self.config.notification_channels {
            if !channel_config.enabled {
                continue;
            }

            let channel: Box<dyn NotificationChannel> = match channel_config.channel_type {
                NotificationChannelType::Console => {
                    Box::new(ConsoleNotificationChannel::new(&channel_config.name))
                }
                NotificationChannelType::Webhook => {
                    if let Some(url) = channel_config.config.get("url") {
                        let mut webhook =
                            WebhookNotificationChannel::new(&channel_config.name, url);

                        // Add any custom headers
                        for (key, value) in &channel_config.config {
                            if key.starts_with("header_") {
                                let header_name = key.strip_prefix("header_").unwrap();
                                webhook = webhook.with_header(header_name, value);
                            }
                        }

                        Box::new(webhook)
                    } else {
                        return Err(RragError::config("webhook_channel", "url", "missing"));
                    }
                }
                _ => {
                    tracing::warn!(
                        "Notification channel type {:?} not yet implemented",
                        channel_config.channel_type
                    );
                    continue;
                }
            };

            channels.insert(channel_config.name.clone(), channel);
        }

        Ok(())
    }

    async fn setup_default_rules(&self) -> RragResult<()> {
        let mut rules = self.alert_rules.write().await;

        // High CPU usage alert
        let cpu_rule = AlertRule::new(
            "high_cpu_usage",
            "High CPU Usage",
            AlertCondition::Threshold {
                metric_name: "system_cpu_usage_percent".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: 80.0,
                duration_minutes: 5,
            },
            AlertSeverity::High,
        )
        .with_description("CPU usage is above 80% for more than 5 minutes");

        // High memory usage alert
        let memory_rule = AlertRule::new(
            "high_memory_usage",
            "High Memory Usage",
            AlertCondition::Threshold {
                metric_name: "system_memory_usage_percent".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: 85.0,
                duration_minutes: 5,
            },
            AlertSeverity::High,
        )
        .with_description("Memory usage is above 85% for more than 5 minutes");

        // High error rate alert
        let error_rate_rule = AlertRule::new(
            "high_error_rate",
            "High Error Rate",
            AlertCondition::RateOfChange {
                metric_name: "search_queries_failed".to_string(),
                operator: ComparisonOperator::GreaterThan,
                rate_per_minute: 10.0,
                window_minutes: 10,
            },
            AlertSeverity::Critical,
        )
        .with_description("Error rate is increasing rapidly");

        // Slow response time alert
        let slow_response_rule = AlertRule::new(
            "slow_response_time",
            "Slow Response Time",
            AlertCondition::Threshold {
                metric_name: "search_processing_time_ms".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: 1000.0,
                duration_minutes: 3,
            },
            AlertSeverity::Medium,
        )
        .with_description("Search response time is above 1 second");

        rules.insert("high_cpu_usage".to_string(), cpu_rule);
        rules.insert("high_memory_usage".to_string(), memory_rule);
        rules.insert("high_error_rate".to_string(), error_rate_rule);
        rules.insert("slow_response_time".to_string(), slow_response_rule);

        Ok(())
    }

    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config(
                "alert_manager",
                "stopped",
                "already running",
            ));
        }

        let handle = self.start_evaluation_loop().await?;
        {
            let mut eval_handle = self.evaluation_handle.write().await;
            *eval_handle = Some(handle);
        }

        *running = true;
        tracing::info!("Alert manager started");
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        {
            let mut eval_handle = self.evaluation_handle.write().await;
            if let Some(handle) = eval_handle.take() {
                handle.abort();
            }
        }

        *running = false;
        tracing::info!("Alert manager stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    async fn start_evaluation_loop(&self) -> RragResult<tokio::task::JoinHandle<()>> {
        let config = self.config.clone();
        let alert_rules = self.alert_rules.clone();
        let active_alerts = self.active_alerts.clone();
        let notification_channels = self.notification_channels.clone();
        let evaluator = self.evaluator.clone();
        let metrics_collector = self.metrics_collector.clone();
        let is_running = self.is_running.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
                config.evaluation_interval_seconds,
            ));

            while *is_running.read().await {
                interval.tick().await;

                // Update metrics in evaluator
                let all_metrics = metrics_collector.get_all_metrics().await;
                for metric in all_metrics {
                    if let Some(value) = Self::extract_metric_value(&metric) {
                        evaluator.update_metric(metric.name, value).await;
                    }
                }

                // Evaluate all alert rules
                let rules = alert_rules.read().await;
                for (rule_id, rule) in rules.iter() {
                    if !rule.enabled {
                        continue;
                    }

                    match evaluator.evaluate_condition(&rule.condition).await {
                        Ok(triggered) => {
                            if triggered {
                                Self::handle_alert_triggered(
                                    rule_id,
                                    rule,
                                    &active_alerts,
                                    &notification_channels,
                                )
                                .await;
                            } else {
                                Self::handle_alert_resolved(
                                    rule_id,
                                    rule,
                                    &active_alerts,
                                    &notification_channels,
                                )
                                .await;
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to evaluate alert rule {}: {}", rule_id, e);
                        }
                    }
                }
            }
        });

        Ok(handle)
    }

    fn extract_metric_value(metric: &Metric) -> Option<f64> {
        match &metric.value {
            MetricValue::Counter(value) => Some(*value as f64),
            MetricValue::Gauge(value) => Some(*value),
            MetricValue::Timer { duration_ms, .. } => Some(*duration_ms),
            MetricValue::Histogram { sum, count, .. } => {
                if *count > 0 {
                    Some(sum / *count as f64)
                } else {
                    Some(0.0)
                }
            }
            MetricValue::Summary { sum, count, .. } => {
                if *count > 0 {
                    Some(sum / *count as f64)
                } else {
                    Some(0.0)
                }
            }
        }
    }

    async fn handle_alert_triggered(
        rule_id: &str,
        rule: &AlertRule,
        active_alerts: &Arc<RwLock<HashMap<String, AlertNotification>>>,
        notification_channels: &Arc<RwLock<HashMap<String, Box<dyn NotificationChannel>>>>,
    ) {
        let mut alerts = active_alerts.write().await;

        // Check if alert is already active (within cooldown)
        if let Some(existing_alert) = alerts.get(rule_id) {
            let cooldown_duration = Duration::minutes(rule.cooldown_minutes as i64);
            if existing_alert.triggered_at + cooldown_duration > Utc::now() {
                return; // Still in cooldown period
            }
        }

        let alert_notification = AlertNotification {
            id: uuid::Uuid::new_v4().to_string(),
            rule_id: rule_id.to_string(),
            rule_name: rule.name.clone(),
            severity: rule.severity,
            status: AlertStatus::Triggered,
            message: format!("Alert triggered: {}", rule.description),
            details: HashMap::new(),
            triggered_at: Utc::now(),
            resolved_at: None,
            acknowledged_at: None,
            acknowledged_by: None,
            notification_channels: rule.notification_channels.clone(),
            tags: rule.tags.clone(),
        };

        alerts.insert(rule_id.to_string(), alert_notification.clone());
        drop(alerts);

        // Send notifications
        let channels = notification_channels.read().await;
        for channel_name in &rule.notification_channels {
            if let Some(channel) = channels.get(channel_name) {
                if let Err(e) = channel.send_notification(&alert_notification).await {
                    tracing::error!("Failed to send notification via {}: {}", channel_name, e);
                }
            }
        }
    }

    async fn handle_alert_resolved(
        rule_id: &str,
        rule: &AlertRule,
        active_alerts: &Arc<RwLock<HashMap<String, AlertNotification>>>,
        notification_channels: &Arc<RwLock<HashMap<String, Box<dyn NotificationChannel>>>>,
    ) {
        let mut alerts = active_alerts.write().await;

        if let Some(mut alert) = alerts.remove(rule_id) {
            if rule.auto_resolve && alert.status == AlertStatus::Triggered {
                alert.status = AlertStatus::Resolved;
                alert.resolved_at = Some(Utc::now());
                alert.message = format!("Alert resolved: {}", rule.description);

                drop(alerts);

                // Send resolution notifications
                let channels = notification_channels.read().await;
                for channel_name in &rule.notification_channels {
                    if let Some(channel) = channels.get(channel_name) {
                        if let Err(e) = channel.send_notification(&alert).await {
                            tracing::error!(
                                "Failed to send resolution notification via {}: {}",
                                channel_name,
                                e
                            );
                        }
                    }
                }
            }
        }
    }

    pub async fn add_alert_rule(&self, rule: AlertRule) -> RragResult<()> {
        let mut rules = self.alert_rules.write().await;
        rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    pub async fn remove_alert_rule(&self, rule_id: &str) -> RragResult<()> {
        let mut rules = self.alert_rules.write().await;
        rules.remove(rule_id);

        // Also remove any active alerts for this rule
        let mut alerts = self.active_alerts.write().await;
        alerts.remove(rule_id);

        Ok(())
    }

    pub async fn acknowledge_alert(
        &self,
        rule_id: &str,
        acknowledged_by: impl Into<String>,
    ) -> RragResult<()> {
        let mut alerts = self.active_alerts.write().await;
        if let Some(alert) = alerts.get_mut(rule_id) {
            alert.status = AlertStatus::Acknowledged;
            alert.acknowledged_at = Some(Utc::now());
            alert.acknowledged_by = Some(acknowledged_by.into());

            tracing::info!("Alert {} acknowledged", rule_id);
        }
        Ok(())
    }

    pub async fn get_active_alerts(&self) -> Vec<AlertNotification> {
        let alerts = self.active_alerts.read().await;
        alerts.values().cloned().collect()
    }

    pub async fn get_alert_rules(&self) -> Vec<AlertRule> {
        let rules = self.alert_rules.read().await;
        rules.values().cloned().collect()
    }

    pub async fn get_alert_stats(&self) -> AlertStats {
        let alerts = self.active_alerts.read().await;
        let rules = self.alert_rules.read().await;

        let total_alerts = alerts.len();
        let by_severity = alerts.values().fold(HashMap::new(), |mut acc, alert| {
            *acc.entry(alert.severity).or_insert(0) += 1;
            acc
        });

        let by_status = alerts.values().fold(HashMap::new(), |mut acc, alert| {
            *acc.entry(alert.status.clone()).or_insert(0) += 1;
            acc
        });

        AlertStats {
            total_active_alerts: total_alerts,
            total_rules: rules.len(),
            alerts_by_severity: by_severity,
            alerts_by_status: by_status,
            last_evaluation: Utc::now(),
        }
    }
}

/// Alert statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStats {
    pub total_active_alerts: usize,
    pub total_rules: usize,
    pub alerts_by_severity: HashMap<AlertSeverity, usize>,
    pub alerts_by_status: HashMap<AlertStatus, usize>,
    pub last_evaluation: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::metrics::MetricsConfig;

    async fn create_test_metrics_collector() -> Arc<MetricsCollector> {
        Arc::new(
            MetricsCollector::new(MetricsConfig::default())
                .await
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn test_console_notification_channel() {
        let channel = ConsoleNotificationChannel::new("test_console");
        assert_eq!(channel.name(), "test_console");
        assert_eq!(channel.channel_type(), NotificationChannelType::Console);
        assert!(channel.is_healthy().await);

        let notification = AlertNotification {
            id: "alert123".to_string(),
            rule_id: "rule123".to_string(),
            rule_name: "Test Alert".to_string(),
            severity: AlertSeverity::High,
            status: AlertStatus::Triggered,
            message: "Test alert message".to_string(),
            details: HashMap::new(),
            triggered_at: Utc::now(),
            resolved_at: None,
            acknowledged_at: None,
            acknowledged_by: None,
            notification_channels: vec!["test_console".to_string()],
            tags: HashMap::new(),
        };

        // This should print to console
        channel.send_notification(&notification).await.unwrap();
    }

    #[tokio::test]
    async fn test_alert_evaluator() {
        let evaluator = AlertEvaluator::new(100);

        // Test threshold condition
        evaluator.update_metric("cpu_usage".to_string(), 50.0).await;
        evaluator.update_metric("cpu_usage".to_string(), 75.0).await;
        evaluator.update_metric("cpu_usage".to_string(), 85.0).await;

        let condition = AlertCondition::Threshold {
            metric_name: "cpu_usage".to_string(),
            operator: ComparisonOperator::GreaterThan,
            value: 80.0,
            duration_minutes: 1,
        };

        let result = evaluator.evaluate_condition(&condition).await.unwrap();
        assert!(result); // Latest value (85.0) is > 80.0

        // Test rate of change condition
        let rate_condition = AlertCondition::RateOfChange {
            metric_name: "cpu_usage".to_string(),
            operator: ComparisonOperator::GreaterThan,
            rate_per_minute: 10.0,
            window_minutes: 5,
        };

        let rate_result = evaluator.evaluate_condition(&rate_condition).await.unwrap();
        // Should detect increasing trend
        assert!(rate_result);
    }

    #[tokio::test]
    async fn test_alert_rule_creation() {
        let rule = AlertRule::new(
            "test_rule",
            "Test Alert Rule",
            AlertCondition::Threshold {
                metric_name: "test_metric".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: 100.0,
                duration_minutes: 5,
            },
            AlertSeverity::High,
        )
        .with_description("Test alert rule description")
        .with_tag("component", "test")
        .with_cooldown(10);

        assert_eq!(rule.id, "test_rule");
        assert_eq!(rule.name, "Test Alert Rule");
        assert_eq!(rule.severity, AlertSeverity::High);
        assert_eq!(rule.cooldown_minutes, 10);
        assert!(rule.tags.contains_key("component"));
        assert_eq!(rule.tags["component"], "test");
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let metrics_collector = create_test_metrics_collector().await;
        let config = AlertConfig::default();
        let mut manager = AlertManager::new(config, metrics_collector).await.unwrap();

        assert!(!manager.is_healthy().await);

        manager.start().await.unwrap();
        assert!(manager.is_healthy().await);

        // Test adding custom alert rule
        let custom_rule = AlertRule::new(
            "custom_rule",
            "Custom Test Rule",
            AlertCondition::Threshold {
                metric_name: "custom_metric".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: 50.0,
                duration_minutes: 1,
            },
            AlertSeverity::Medium,
        );

        manager.add_alert_rule(custom_rule).await.unwrap();

        let rules = manager.get_alert_rules().await;
        assert!(rules.iter().any(|r| r.id == "custom_rule"));

        let stats = manager.get_alert_stats().await;
        assert!(stats.total_rules > 0);

        manager.stop().await.unwrap();
        assert!(!manager.is_healthy().await);
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::High);
        assert!(AlertSeverity::High > AlertSeverity::Medium);
        assert!(AlertSeverity::Medium > AlertSeverity::Low);
    }

    #[test]
    fn test_comparison_operators() {
        assert_eq!(
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThan
        );
        assert_ne!(
            ComparisonOperator::GreaterThan,
            ComparisonOperator::LessThan
        );
    }

    #[tokio::test]
    async fn test_alert_acknowledgment() {
        let metrics_collector = create_test_metrics_collector().await;
        let config = AlertConfig::default();
        let manager = AlertManager::new(config, metrics_collector).await.unwrap();

        // Create a mock active alert
        let alert = AlertNotification {
            id: "test_alert".to_string(),
            rule_id: "test_rule".to_string(),
            rule_name: "Test Rule".to_string(),
            severity: AlertSeverity::High,
            status: AlertStatus::Triggered,
            message: "Test alert".to_string(),
            details: HashMap::new(),
            triggered_at: Utc::now(),
            resolved_at: None,
            acknowledged_at: None,
            acknowledged_by: None,
            notification_channels: vec![],
            tags: HashMap::new(),
        };

        {
            let mut alerts = manager.active_alerts.write().await;
            alerts.insert("test_rule".to_string(), alert);
        }

        manager
            .acknowledge_alert("test_rule", "test_user")
            .await
            .unwrap();

        let active_alerts = manager.get_active_alerts().await;
        let acknowledged_alert = active_alerts
            .iter()
            .find(|a| a.rule_id == "test_rule")
            .unwrap();
        assert_eq!(acknowledged_alert.status, AlertStatus::Acknowledged);
        assert!(acknowledged_alert.acknowledged_at.is_some());
        assert_eq!(
            acknowledged_alert.acknowledged_by.as_ref().unwrap(),
            "test_user"
        );
    }
}
