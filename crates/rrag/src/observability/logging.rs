//! # Log Aggregation and Search System
//!
//! Centralized logging with structured data, search capabilities,
//! and real-time log streaming for RRAG system operations.

use crate::{RragError, RragResult};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    pub enabled: bool,
    pub level: LogLevel,
    pub buffer_size: usize,
    pub flush_interval_seconds: u64,
    pub retention_days: u32,
    pub structured_logging: bool,
    pub include_stack_trace: bool,
    pub log_to_file: bool,
    pub log_file_path: Option<String>,
    pub log_rotation_size_mb: u64,
    pub max_log_files: u32,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: LogLevel::Info,
            buffer_size: 10000,
            flush_interval_seconds: 5,
            retention_days: 30,
            structured_logging: true,
            include_stack_trace: false,
            log_to_file: true,
            log_file_path: Some("rrag.log".to_string()),
            log_rotation_size_mb: 100,
            max_log_files: 10,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Fatal = 5,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => write!(f, "TRACE"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warn => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

impl From<&str> for LogLevel {
    fn from(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "TRACE" => LogLevel::Trace,
            "DEBUG" => LogLevel::Debug,
            "INFO" => LogLevel::Info,
            "WARN" | "WARNING" => LogLevel::Warn,
            "ERROR" => LogLevel::Error,
            "FATAL" | "CRITICAL" => LogLevel::Fatal,
            _ => LogLevel::Info,
        }
    }
}

/// Structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub component: String,
    pub operation: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub fields: HashMap<String, serde_json::Value>,
    pub stack_trace: Option<String>,
    pub source_file: Option<String>,
    pub source_line: Option<u32>,
    pub duration_ms: Option<f64>,
}

impl LogEntry {
    pub fn new(level: LogLevel, message: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            level,
            message: message.into(),
            component: component.into(),
            operation: None,
            user_id: None,
            session_id: None,
            trace_id: None,
            span_id: None,
            fields: HashMap::new(),
            stack_trace: None,
            source_file: None,
            source_line: None,
            duration_ms: None,
        }
    }

    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.operation = Some(operation.into());
        self
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_trace(mut self, trace_id: impl Into<String>, span_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self.span_id = Some(span_id.into());
        self
    }

    pub fn with_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.fields.insert(key.into(), value);
        self
    }

    pub fn with_duration(mut self, duration_ms: f64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn with_source(mut self, file: impl Into<String>, line: u32) -> Self {
        self.source_file = Some(file.into());
        self.source_line = Some(line);
        self
    }

    pub fn with_stack_trace(mut self, stack_trace: impl Into<String>) -> Self {
        self.stack_trace = Some(stack_trace.into());
        self
    }

    /// Format as JSON for structured logging
    pub fn to_json(&self) -> RragResult<String> {
        serde_json::to_string(self).map_err(|e| RragError::agent("log_formatter", e.to_string()))
    }

    /// Format as human-readable text
    pub fn to_text(&self) -> String {
        let timestamp = self.timestamp.format("%Y-%m-%d %H:%M:%S%.3f UTC");
        let level_str = format!("{:5}", self.level);

        let mut parts = vec![
            format!("[{}]", timestamp),
            format!("[{}]", level_str),
            format!("[{}]", self.component),
        ];

        if let Some(ref operation) = self.operation {
            parts.push(format!("[{}]", operation));
        }

        parts.push(self.message.clone());

        if let Some(duration) = self.duration_ms {
            parts.push(format!("({}ms)", duration));
        }

        if !self.fields.is_empty() {
            let fields_str = self
                .fields
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ");
            parts.push(format!("{{{}}}", fields_str));
        }

        parts.join(" ")
    }
}

/// Log query for searching through logs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogQuery {
    pub level_filter: Option<LogLevel>,
    pub component_filter: Option<String>,
    pub operation_filter: Option<String>,
    pub user_filter: Option<String>,
    pub session_filter: Option<String>,
    pub message_contains: Option<String>,
    pub time_range: Option<TimeRange>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub sort_order: SortOrder,
    pub field_filters: HashMap<String, FieldFilter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Ascending,
    Descending,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldFilter {
    Equals(serde_json::Value),
    Contains(String),
    GreaterThan(f64),
    LessThan(f64),
    Between(f64, f64),
}

impl Default for LogQuery {
    fn default() -> Self {
        Self {
            level_filter: None,
            component_filter: None,
            operation_filter: None,
            user_filter: None,
            session_filter: None,
            message_contains: None,
            time_range: None,
            limit: Some(100),
            offset: None,
            sort_order: SortOrder::Descending,
            field_filters: HashMap::new(),
        }
    }
}

/// Log filter for real-time streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogFilter {
    pub min_level: LogLevel,
    pub components: Vec<String>,
    pub operations: Vec<String>,
    pub include_fields: Vec<String>,
    pub exclude_patterns: Vec<String>,
}

impl Default for LogFilter {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Info,
            components: Vec::new(),
            operations: Vec::new(),
            include_fields: Vec::new(),
            exclude_patterns: Vec::new(),
        }
    }
}

/// Log search engine for querying log entries
pub struct LogSearchEngine {
    logs: Arc<RwLock<Vec<LogEntry>>>,
    max_size: usize,
}

impl LogSearchEngine {
    pub fn new(max_size: usize) -> Self {
        Self {
            logs: Arc::new(RwLock::new(Vec::new())),
            max_size,
        }
    }

    pub async fn add_entry(&self, entry: LogEntry) {
        let mut logs = self.logs.write().await;
        logs.push(entry);

        // Keep only recent entries
        let logs_len = logs.len();
        if logs_len > self.max_size {
            logs.drain(0..logs_len - self.max_size);
        }
    }

    pub async fn search(&self, query: &LogQuery) -> Vec<LogEntry> {
        let logs = self.logs.read().await;
        let mut results: Vec<_> = logs
            .iter()
            .filter(|entry| self.matches_query(entry, query))
            .cloned()
            .collect();

        // Sort results
        match query.sort_order {
            SortOrder::Ascending => results.sort_by_key(|e| e.timestamp),
            SortOrder::Descending => results.sort_by_key(|e| std::cmp::Reverse(e.timestamp)),
        }

        // Apply pagination
        let start = query.offset.unwrap_or(0);
        let end = if let Some(limit) = query.limit {
            std::cmp::min(start + limit, results.len())
        } else {
            results.len()
        };

        if start < results.len() {
            results[start..end].to_vec()
        } else {
            Vec::new()
        }
    }

    fn matches_query(&self, entry: &LogEntry, query: &LogQuery) -> bool {
        // Level filter
        if let Some(min_level) = query.level_filter {
            if entry.level < min_level {
                return false;
            }
        }

        // Component filter
        if let Some(ref component) = query.component_filter {
            if entry.component != *component {
                return false;
            }
        }

        // Operation filter
        if let Some(ref operation) = query.operation_filter {
            if entry.operation.as_ref() != Some(operation) {
                return false;
            }
        }

        // User filter
        if let Some(ref user) = query.user_filter {
            if entry.user_id.as_ref() != Some(user) {
                return false;
            }
        }

        // Session filter
        if let Some(ref session) = query.session_filter {
            if entry.session_id.as_ref() != Some(session) {
                return false;
            }
        }

        // Message contains filter
        if let Some(ref text) = query.message_contains {
            if !entry.message.to_lowercase().contains(&text.to_lowercase()) {
                return false;
            }
        }

        // Time range filter
        if let Some(ref range) = query.time_range {
            if entry.timestamp < range.start || entry.timestamp > range.end {
                return false;
            }
        }

        // Field filters
        for (field_name, field_filter) in &query.field_filters {
            if let Some(field_value) = entry.fields.get(field_name) {
                if !self.matches_field_filter(field_value, field_filter) {
                    return false;
                }
            } else {
                return false; // Field doesn't exist
            }
        }

        true
    }

    fn matches_field_filter(&self, value: &serde_json::Value, filter: &FieldFilter) -> bool {
        match filter {
            FieldFilter::Equals(expected) => value == expected,
            FieldFilter::Contains(text) => {
                if let Some(s) = value.as_str() {
                    s.to_lowercase().contains(&text.to_lowercase())
                } else {
                    false
                }
            }
            FieldFilter::GreaterThan(threshold) => {
                if let Some(num) = value.as_f64() {
                    num > *threshold
                } else {
                    false
                }
            }
            FieldFilter::LessThan(threshold) => {
                if let Some(num) = value.as_f64() {
                    num < *threshold
                } else {
                    false
                }
            }
            FieldFilter::Between(min, max) => {
                if let Some(num) = value.as_f64() {
                    num >= *min && num <= *max
                } else {
                    false
                }
            }
        }
    }

    pub async fn get_log_stats(&self) -> LogStats {
        let logs = self.logs.read().await;

        let mut level_counts = HashMap::new();
        let mut component_counts = HashMap::new();
        let total_entries = logs.len();

        for entry in logs.iter() {
            *level_counts.entry(entry.level).or_insert(0) += 1;
            *component_counts.entry(entry.component.clone()).or_insert(0) += 1;
        }

        let recent_errors = logs
            .iter()
            .filter(|e| e.level >= LogLevel::Error)
            .filter(|e| e.timestamp > Utc::now() - Duration::hours(1))
            .count();

        LogStats {
            total_entries,
            entries_by_level: level_counts,
            entries_by_component: component_counts,
            recent_errors_count: recent_errors,
            oldest_entry: logs.first().map(|e| e.timestamp),
            newest_entry: logs.last().map(|e| e.timestamp),
        }
    }
}

/// Log statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStats {
    pub total_entries: usize,
    pub entries_by_level: HashMap<LogLevel, usize>,
    pub entries_by_component: HashMap<String, usize>,
    pub recent_errors_count: usize,
    pub oldest_entry: Option<DateTime<Utc>>,
    pub newest_entry: Option<DateTime<Utc>>,
}

/// Structured logger implementation
pub struct StructuredLogger {
    config: LogConfig,
    sender: mpsc::UnboundedSender<LogEntry>,
}

impl StructuredLogger {
    pub fn new(config: LogConfig, sender: mpsc::UnboundedSender<LogEntry>) -> Self {
        Self { config, sender }
    }

    pub fn trace(&self, message: impl Into<String>, component: impl Into<String>) -> LogBuilder {
        self.log(LogLevel::Trace, message, component)
    }

    pub fn debug(&self, message: impl Into<String>, component: impl Into<String>) -> LogBuilder {
        self.log(LogLevel::Debug, message, component)
    }

    pub fn info(&self, message: impl Into<String>, component: impl Into<String>) -> LogBuilder {
        self.log(LogLevel::Info, message, component)
    }

    pub fn warn(&self, message: impl Into<String>, component: impl Into<String>) -> LogBuilder {
        self.log(LogLevel::Warn, message, component)
    }

    pub fn error(&self, message: impl Into<String>, component: impl Into<String>) -> LogBuilder {
        self.log(LogLevel::Error, message, component)
    }

    pub fn fatal(&self, message: impl Into<String>, component: impl Into<String>) -> LogBuilder {
        self.log(LogLevel::Fatal, message, component)
    }

    fn log(
        &self,
        level: LogLevel,
        message: impl Into<String>,
        component: impl Into<String>,
    ) -> LogBuilder {
        LogBuilder::new(level, message, component, self.sender.clone())
    }
}

/// Fluent builder for log entries
pub struct LogBuilder {
    entry: LogEntry,
    sender: mpsc::UnboundedSender<LogEntry>,
}

impl LogBuilder {
    fn new(
        level: LogLevel,
        message: impl Into<String>,
        component: impl Into<String>,
        sender: mpsc::UnboundedSender<LogEntry>,
    ) -> Self {
        Self {
            entry: LogEntry::new(level, message, component),
            sender,
        }
    }

    pub fn operation(mut self, operation: impl Into<String>) -> Self {
        self.entry = self.entry.with_operation(operation);
        self
    }

    pub fn user(mut self, user_id: impl Into<String>) -> Self {
        self.entry = self.entry.with_user(user_id);
        self
    }

    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.entry = self.entry.with_session(session_id);
        self
    }

    pub fn trace(mut self, trace_id: impl Into<String>, span_id: impl Into<String>) -> Self {
        self.entry = self.entry.with_trace(trace_id, span_id);
        self
    }

    pub fn field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.entry = self.entry.with_field(key, value);
        self
    }

    pub fn duration(mut self, duration_ms: f64) -> Self {
        self.entry = self.entry.with_duration(duration_ms);
        self
    }

    pub fn source(mut self, file: impl Into<String>, line: u32) -> Self {
        self.entry = self.entry.with_source(file, line);
        self
    }

    pub fn send(self) {
        let _ = self.sender.send(self.entry);
    }
}

/// Main log aggregator
pub struct LogAggregator {
    config: LogConfig,
    search_engine: Arc<LogSearchEngine>,
    logger: Arc<StructuredLogger>,
    log_sender: mpsc::UnboundedSender<LogEntry>,
    log_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<LogEntry>>>>,
    file_writer: Option<Arc<RwLock<std::fs::File>>>,
    stream_sender: broadcast::Sender<LogEntry>,
    _stream_receiver: broadcast::Receiver<LogEntry>,
    processing_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    is_running: Arc<RwLock<bool>>,
}

impl LogAggregator {
    pub async fn new(config: LogConfig) -> RragResult<Self> {
        let search_engine = Arc::new(LogSearchEngine::new(config.buffer_size));
        let (log_sender, log_receiver) = mpsc::unbounded_channel();
        let logger = Arc::new(StructuredLogger::new(config.clone(), log_sender.clone()));
        let (stream_sender, stream_receiver) = broadcast::channel(1000);

        // Initialize file writer if needed
        let file_writer = if config.log_to_file {
            if let Some(ref path) = config.log_file_path {
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|e| RragError::storage("log_file_create", e))?;
                Some(Arc::new(RwLock::new(file)))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            search_engine,
            logger,
            log_sender,
            log_receiver: Arc::new(RwLock::new(Some(log_receiver))),
            file_writer,
            stream_sender,
            _stream_receiver: stream_receiver,
            processing_handle: Arc::new(RwLock::new(None)),
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config(
                "log_aggregator",
                "stopped",
                "already running",
            ));
        }

        {
            let mut receiver_guard = self.log_receiver.write().await;
            if let Some(receiver) = receiver_guard.take() {
                let handle = self.start_processing_loop(receiver).await?;
                let mut handle_guard = self.processing_handle.write().await;
                *handle_guard = Some(handle);
            }
        }

        *running = true;
        tracing::info!("Log aggregator started");
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        {
            let mut handle_guard = self.processing_handle.write().await;
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }

        // Flush any remaining logs
        if let Some(ref file_writer) = self.file_writer {
            let mut file = file_writer.write().await;
            let _ = file.flush();
        }

        *running = false;
        tracing::info!("Log aggregator stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    async fn start_processing_loop(
        &self,
        mut receiver: mpsc::UnboundedReceiver<LogEntry>,
    ) -> RragResult<tokio::task::JoinHandle<()>> {
        let search_engine = self.search_engine.clone();
        let file_writer = self.file_writer.clone();
        let stream_sender = self.stream_sender.clone();
        let config = self.config.clone();
        let is_running = self.is_running.clone();

        let handle = tokio::spawn(async move {
            let mut flush_interval = tokio::time::interval(tokio::time::Duration::from_secs(
                config.flush_interval_seconds,
            ));

            while *is_running.read().await {
                tokio::select! {
                    Some(entry) = receiver.recv() => {
                        // Check if entry meets minimum level requirement
                        if entry.level >= config.level {
                            // Add to search engine
                            search_engine.add_entry(entry.clone()).await;

                            // Write to file if configured
                            if let Some(ref writer) = file_writer {
                                let log_line = if config.structured_logging {
                                    entry.to_json().unwrap_or_else(|_| entry.to_text())
                                } else {
                                    entry.to_text()
                                };

                                let mut file = writer.write().await;
                                if writeln!(file, "{}", log_line).is_err() {
                                    tracing::debug!("Failed to write to log file");
                                }
                            }

                            // Send to live stream subscribers
                            let _ = stream_sender.send(entry);
                        }
                    }
                    _ = flush_interval.tick() => {
                        // Periodic flush
                        if let Some(ref writer) = file_writer {
                            let mut file = writer.write().await;
                            let _ = file.flush();
                        }
                    }
                }
            }
        });

        Ok(handle)
    }

    pub fn logger(&self) -> &Arc<StructuredLogger> {
        &self.logger
    }

    pub async fn search_logs(&self, query: &LogQuery) -> Vec<LogEntry> {
        self.search_engine.search(query).await
    }

    pub async fn get_stats(&self) -> LogStats {
        self.search_engine.get_log_stats().await
    }

    pub fn subscribe_to_stream(&self) -> broadcast::Receiver<LogEntry> {
        self.stream_sender.subscribe()
    }

    pub async fn add_log_entry(&self, entry: LogEntry) -> RragResult<()> {
        self.log_sender
            .send(entry)
            .map_err(|e| RragError::agent("log_aggregator", e.to_string()))?;
        Ok(())
    }

    /// Convenience method for creating a log entry and adding it
    pub async fn log(
        &self,
        level: LogLevel,
        message: impl Into<String>,
        component: impl Into<String>,
    ) -> RragResult<()> {
        let entry = LogEntry::new(level, message, component);
        self.add_log_entry(entry).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_log_entry_creation() {
        let entry = LogEntry::new(LogLevel::Info, "Test message", "test_component")
            .with_operation("test_operation")
            .with_user("user123")
            .with_session("session456")
            .with_field("custom_field", serde_json::json!("custom_value"))
            .with_duration(150.5);

        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.message, "Test message");
        assert_eq!(entry.component, "test_component");
        assert_eq!(entry.operation.as_ref().unwrap(), "test_operation");
        assert_eq!(entry.user_id.as_ref().unwrap(), "user123");
        assert_eq!(entry.session_id.as_ref().unwrap(), "session456");
        assert_eq!(entry.duration_ms.unwrap(), 150.5);
        assert!(entry.fields.contains_key("custom_field"));

        // Test formatting
        let json_str = entry.to_json().unwrap();
        assert!(json_str.contains("Test message"));
        assert!(json_str.contains("INFO"));

        let text_str = entry.to_text();
        assert!(text_str.contains("Test message"));
        assert!(text_str.contains("INFO"));
        assert!(text_str.contains("test_component"));
    }

    #[tokio::test]
    async fn test_log_search_engine() {
        let engine = LogSearchEngine::new(1000);

        // Add test log entries
        let entries = vec![
            LogEntry::new(LogLevel::Info, "Info message", "component1"),
            LogEntry::new(LogLevel::Error, "Error message", "component1"),
            LogEntry::new(LogLevel::Warn, "Warning message", "component2").with_user("user123"),
            LogEntry::new(LogLevel::Debug, "Debug message", "component2"),
        ];

        for entry in entries {
            engine.add_entry(entry).await;
        }

        // Test level filter
        let query = LogQuery {
            level_filter: Some(LogLevel::Warn),
            ..Default::default()
        };
        let results = engine.search(&query).await;
        assert_eq!(results.len(), 2); // Warn and Error (Error >= Warn)

        // Test component filter
        let query = LogQuery {
            component_filter: Some("component1".to_string()),
            ..Default::default()
        };
        let results = engine.search(&query).await;
        assert_eq!(results.len(), 2);

        // Test user filter
        let query = LogQuery {
            user_filter: Some("user123".to_string()),
            ..Default::default()
        };
        let results = engine.search(&query).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].level, LogLevel::Warn);

        // Test message contains
        let query = LogQuery {
            message_contains: Some("Error".to_string()),
            ..Default::default()
        };
        let results = engine.search(&query).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].level, LogLevel::Error);

        // Test stats
        let stats = engine.get_log_stats().await;
        assert_eq!(stats.total_entries, 4);
        assert_eq!(stats.entries_by_level[&LogLevel::Info], 1);
        assert_eq!(stats.entries_by_level[&LogLevel::Error], 1);
        assert_eq!(stats.entries_by_component["component1"], 2);
        assert_eq!(stats.entries_by_component["component2"], 2);
    }

    #[tokio::test]
    async fn test_log_aggregator() {
        let config = LogConfig {
            log_to_file: false, // Don't write to file in tests
            ..Default::default()
        };

        let mut aggregator = LogAggregator::new(config).await.unwrap();
        assert!(!aggregator.is_healthy().await);

        aggregator.start().await.unwrap();
        assert!(aggregator.is_healthy().await);

        // Test logging through the structured logger
        let logger = aggregator.logger();
        logger
            .info("Test info message", "test_component")
            .user("user123")
            .operation("test_operation")
            .field("test_field", serde_json::json!("test_value"))
            .send();

        logger
            .error("Test error message", "test_component")
            .session("session456")
            .duration(200.0)
            .send();

        // Give some time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Test search
        let query = LogQuery {
            level_filter: Some(LogLevel::Info),
            ..Default::default()
        };
        let results = aggregator.search_logs(&query).await;
        assert!(results.len() >= 2);

        let stats = aggregator.get_stats().await;
        assert!(stats.total_entries >= 2);

        aggregator.stop().await.unwrap();
        assert!(!aggregator.is_healthy().await);
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Fatal > LogLevel::Error);
        assert!(LogLevel::Error > LogLevel::Warn);
        assert!(LogLevel::Warn > LogLevel::Info);
        assert!(LogLevel::Info > LogLevel::Debug);
        assert!(LogLevel::Debug > LogLevel::Trace);
    }

    #[test]
    fn test_log_level_from_string() {
        assert_eq!(LogLevel::from("INFO"), LogLevel::Info);
        assert_eq!(LogLevel::from("info"), LogLevel::Info);
        assert_eq!(LogLevel::from("ERROR"), LogLevel::Error);
        assert_eq!(LogLevel::from("WARN"), LogLevel::Warn);
        assert_eq!(LogLevel::from("WARNING"), LogLevel::Warn);
        assert_eq!(LogLevel::from("FATAL"), LogLevel::Fatal);
        assert_eq!(LogLevel::from("unknown"), LogLevel::Info); // Default
    }

    #[tokio::test]
    async fn test_field_filters() {
        let engine = LogSearchEngine::new(1000);

        let entry = LogEntry::new(LogLevel::Info, "Test message", "component")
            .with_field("number", serde_json::json!(42))
            .with_field("text", serde_json::json!("hello world"))
            .with_field("decimal", serde_json::json!(3.14));

        engine.add_entry(entry).await;

        // Test equals filter
        let mut query = LogQuery::default();
        query.field_filters.insert(
            "number".to_string(),
            FieldFilter::Equals(serde_json::json!(42)),
        );
        let results = engine.search(&query).await;
        assert_eq!(results.len(), 1);

        // Test contains filter
        let mut query = LogQuery::default();
        query.field_filters.insert(
            "text".to_string(),
            FieldFilter::Contains("hello".to_string()),
        );
        let results = engine.search(&query).await;
        assert_eq!(results.len(), 1);

        // Test greater than filter
        let mut query = LogQuery::default();
        query
            .field_filters
            .insert("number".to_string(), FieldFilter::GreaterThan(40.0));
        let results = engine.search(&query).await;
        assert_eq!(results.len(), 1);

        // Test between filter
        let mut query = LogQuery::default();
        query
            .field_filters
            .insert("decimal".to_string(), FieldFilter::Between(3.0, 4.0));
        let results = engine.search(&query).await;
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_log_streaming() {
        let config = LogConfig {
            log_to_file: false,
            ..Default::default()
        };

        let mut aggregator = LogAggregator::new(config).await.unwrap();
        aggregator.start().await.unwrap();

        let mut stream = aggregator.subscribe_to_stream();

        // Send a log entry
        let entry = LogEntry::new(LogLevel::Info, "Stream test", "test_component");
        aggregator.add_log_entry(entry.clone()).await.unwrap();

        // Receive the streamed entry
        let received = tokio::time::timeout(tokio::time::Duration::from_millis(100), stream.recv())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(received.message, "Stream test");
        assert_eq!(received.component, "test_component");

        aggregator.stop().await.unwrap();
    }
}
