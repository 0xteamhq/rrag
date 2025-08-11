//! # RRAG Observability Dashboard Demo
//! 
//! Comprehensive demonstration of the RRAG observability system featuring:
//! - Real-time metrics collection and monitoring
//! - Web dashboard with interactive charts and visualizations
//! - Intelligent alerting with multiple notification channels
//! - Log aggregation with structured search capabilities
//! - Health monitoring with component status tracking
//! - Performance profiling and bottleneck analysis
//! - Data export and reporting functionality
//! - Automated data retention and archiving

use rrag::prelude::*;
use rrag::observability::{
    logging::{LogLevel, LogAggregator, LogQuery, LogConfig}, 
    monitoring::{SystemMonitor, MonitoringConfig},
    profiling::{PerformanceProfiler, ProfilingConfig},
    export::{ExportFormat, ReportConfig, ExportConfig},
    health::HealthConfig,
    retention::RetentionConfig,
    dashboard::DashboardConfig,
    alerting::AlertConfig,
    ObservabilityConfig,
};
use rrag::{AlertCondition, AlertSeverity, AlertRule, ComponentStatus};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use chrono::Utc;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("🚀 Starting RRAG Observability Dashboard Demo");
    println!("===============================================");

    // Create observability system with comprehensive configuration
    let observability_config = ObservabilityConfig {
        system_id: "rrag-demo-system".to_string(),
        environment: "demo".to_string(),
        
        // Enable all components
        metrics: rrag::observability::metrics::MetricsConfig {
            enabled: true,
            collection_interval_seconds: 5,
            buffer_size: 5000,
            export_interval_seconds: 30,
            retention_days: 30,
            labels: HashMap::from([
                ("service".to_string(), "rrag-demo".to_string()),
                ("version".to_string(), "1.0.0".to_string()),
            ]),
        },
        
        monitoring: MonitoringConfig {
            enabled: true,
            collection_interval_seconds: 10,
            performance_window_minutes: 5,
            search_analytics_enabled: true,
            user_tracking_enabled: true,
            resource_monitoring_enabled: true,
            alert_thresholds: rrag::observability::monitoring::AlertThresholds {
                cpu_usage_percent: 80.0,
                memory_usage_percent: 85.0,
                error_rate_percent: 5.0,
                response_time_ms: 1000.0,
                disk_usage_percent: 90.0,
                queue_size: 1000,
            },
        },
        
        alerting: AlertConfig {
            enabled: true,
            evaluation_interval_seconds: 15,
            alert_buffer_size: 500,
            notification_channels: vec![
                rrag::observability::alerting::NotificationChannelConfig {
                    name: "console".to_string(),
                    channel_type: rrag::observability::alerting::NotificationChannelType::Console,
                    enabled: true,
                    config: HashMap::new(),
                },
                rrag::observability::alerting::NotificationChannelConfig {
                    name: "webhook".to_string(),
                    channel_type: rrag::observability::alerting::NotificationChannelType::Webhook,
                    enabled: false, // Disabled for demo
                    config: HashMap::from([
                        ("url".to_string(), "http://localhost:9999/alerts".to_string()),
                    ]),
                },
            ],
            default_severity: AlertSeverity::Medium,
            alert_grouping_enabled: true,
            alert_grouping_window_minutes: 5,
            escalation_enabled: false,
            escalation_delay_minutes: 30,
        },
        
        dashboard: DashboardConfig {
            enabled: true,
            host: "127.0.0.1".to_string(),
            port: 3000,
            title: "RRAG Observability Dashboard - Demo".to_string(),
            refresh_interval_seconds: 5,
            max_data_points: 100,
            websocket_enabled: true,
            auth_enabled: false,
            auth_token: None,
            cors_enabled: true,
            allowed_origins: vec!["*".to_string()],
        },
        
        logging: LogConfig {
            enabled: true,
            level: LogLevel::Info,
            buffer_size: 1000,
            flush_interval_seconds: 5,
            retention_days: 30,
            structured_logging: true,
            include_stack_trace: false,
            log_to_file: true,
            log_file_path: Some("rrag_demo.log".to_string()),
            log_rotation_size_mb: 10,
            max_log_files: 5,
        },
        
        health: HealthConfig {
            enabled: true,
            check_interval_seconds: 20,
            timeout_seconds: 10,
            max_consecutive_failures: 3,
            recovery_threshold: 2,
            enable_detailed_checks: true,
            enable_dependency_checks: true,
            custom_checks: vec![],
        },
        
        profiling: ProfilingConfig {
            enabled: true,
            sample_rate: 0.3, // 30% sampling
            max_profiles: 1000,
            profile_duration_seconds: 60,
            enable_cpu_profiling: true,
            enable_memory_profiling: true,
            enable_io_profiling: true,
            enable_custom_metrics: true,
            bottleneck_threshold_ms: 100.0,
        },
        
        export: ExportConfig {
            enabled: true,
            default_format: ExportFormat::Json,
            output_directory: "./exports".to_string(),
            max_file_size_mb: 50,
            retention_days: 90,
            compression_enabled: true,
            scheduled_exports: vec![],
            destinations: vec![
                rrag::observability::export::ExportDestinationConfig {
                    name: "local_files".to_string(),
                    destination_type: rrag::observability::export::DestinationType::LocalFile,
                    config: HashMap::from([
                        ("path".to_string(), "./exports".to_string()),
                    ]),
                    enabled: true,
                },
            ],
        },
        
        retention: RetentionConfig {
            enabled: true,
            retention_days: 90,
            archive_enabled: true,
            archive_compression: true,
            cleanup_interval_hours: 24,
            policies: vec![
                rrag::observability::retention::RetentionPolicyConfig::default_metrics(),
                rrag::observability::retention::RetentionPolicyConfig::default_logs(),
                rrag::observability::retention::RetentionPolicyConfig::default_health(),
                rrag::observability::retention::RetentionPolicyConfig::default_profiles(),
            ],
            historical_analysis_enabled: true,
            trend_analysis_days: 30,
        },
        
        enabled: true,
        sample_rate: 1.0,
        batch_size: 100,
        flush_interval_seconds: 10,
    };

    // Initialize the observability system
    let observability = ObservabilitySystem::new(observability_config).await?;
    
    println!("📊 Starting observability system components...");
    observability.start().await?;
    
    // Get references to individual components
    let metrics = observability.metrics().clone();
    let monitoring = observability.monitoring().clone();
    let alerting = observability.alerting().clone();
    let _dashboard = observability.dashboard().clone();
    let logging = observability.logging().clone();
    let health = observability.health().clone();
    let profiling = observability.profiling().clone();
    let export = observability.export().clone();

    println!("✅ Observability system started successfully!");
    println!();
    println!("🌐 Dashboard available at: http://127.0.0.1:3000");
    println!("📈 Real-time metrics and charts will be displayed");
    println!("🚨 Alerts will be shown in the console and dashboard");
    println!("📝 Logs are being aggregated and searchable");
    println!("💾 Data export and retention policies are active");
    println!();

    // Set up health checkers for demo components
    println!("🏥 Setting up health checkers...");
    
    // Add various health checkers to demonstrate monitoring
    health.add_checker(Box::new(
        rrag::observability::health::BasicHealthChecker::new("search_engine", || {
            // Simulate occasional search engine issues
            if rand::random::<f64>() > 0.9 {
                Err(RragError::agent("search_engine", "Search index temporarily unavailable"))
            } else {
                Ok(ComponentStatus::Healthy)
            }
        }).with_critical(true)
    )).await?;

    health.add_checker(Box::new(
        rrag::observability::health::BasicHealthChecker::new("vector_store", || {
            // Simulate vector store with occasional degradation
            let rand_val = rand::random::<f64>();
            if rand_val > 0.95 {
                Ok(ComponentStatus::Unhealthy)
            } else if rand_val > 0.85 {
                Ok(ComponentStatus::Degraded)
            } else {
                Ok(ComponentStatus::Healthy)
            }
        })
    )).await?;

    health.add_checker(Box::new(
        rrag::observability::health::DatabaseHealthChecker::new("embedding_db", "postgresql://localhost:5432/embeddings")
            .with_critical(true)
    )).await?;

    // Add custom alert rules for demo
    println!("🚨 Setting up custom alert rules...");
    
    let custom_alert = AlertRule::new(
        "demo_high_search_latency",
        "High Search Latency - Demo",
        AlertCondition::Threshold {
            metric_name: "search_processing_time_ms".to_string(),
            operator: rrag::observability::alerting::ComparisonOperator::GreaterThan,
            value: 500.0,
            duration_minutes: 2,
        },
        AlertSeverity::High,
    ).with_description("Search queries are taking too long to complete")
     .with_channels(vec!["console".to_string()])
     .with_tag("demo", "true")
     .with_cooldown(5);

    alerting.add_alert_rule(custom_alert).await?;

    // Start simulating realistic system activity
    println!("🎭 Starting system activity simulation...");
    println!("   - Generating metrics, logs, and health data");
    println!("   - Simulating search queries and user activity");
    println!("   - Creating performance profiles");
    println!("   - Triggering occasional alerts");
    println!();

    let metrics_clone = metrics.clone();
    let monitoring_clone = monitoring.clone();
    let logging_clone = logging.clone();
    let profiling_clone = profiling.clone();
    
    let simulation_handle = tokio::spawn(async move {
        simulate_system_activity(
            metrics_clone,
            monitoring_clone,
            logging_clone,
            profiling_clone,
        ).await
    });

    // Wait for initial data to populate
    sleep(Duration::from_secs(5)).await;

    // Demonstrate dashboard features
    println!("📊 Dashboard Features Demo:");
    println!("=========================");

    // Show current system status
    let status = observability.status().await;
    println!("System Status: {:?}", status.running);
    println!("Component Health:");
    for (component, healthy) in &status.components {
        let status_icon = if *healthy { "✅" } else { "❌" };
        println!("  {} {}: {}", status_icon, component, if *healthy { "Healthy" } else { "Unhealthy" });
    }
    println!();

    // Demonstrate metrics export
    println!("💾 Exporting current metrics...");
    let all_metrics = observability.metrics().get_all_metrics().await;
    if !all_metrics.is_empty() {
        let export_result = export.export_metrics(
            all_metrics.clone(),
            ExportFormat::Json,
            vec!["local_files".to_string()],
            rrag::observability::export::ExportFilters::default(),
        ).await?;
        
        println!("   Exported {} metrics to: {:?}", 
                 export_result.record_count, 
                 export_result.file_path);
        println!("   Export status: {:?}", export_result.status);
    }
    println!();

    // Demonstrate profiling analysis
    println!("🔍 Analyzing performance bottlenecks...");
    let bottleneck_analysis = profiling.analyze_bottlenecks(5).await;
    if !bottleneck_analysis.bottlenecks.is_empty() {
        println!("   Detected {} bottlenecks:", bottleneck_analysis.bottlenecks.len());
        for bottleneck in &bottleneck_analysis.bottlenecks {
            println!("   - {}: {:.2}ms avg (Impact: {:.2})", 
                     bottleneck.operation, 
                     bottleneck.average_duration_ms,
                     bottleneck.impact_score);
        }
        
        if !bottleneck_analysis.recommendations.is_empty() {
            println!("   Recommendations:");
            for rec in &bottleneck_analysis.recommendations {
                println!("   * {} (Priority: {:?})", rec.recommendation, rec.priority);
            }
        }
    } else {
        println!("   No performance bottlenecks detected");
    }
    println!();

    // Show alert status
    println!("🚨 Current Alert Status:");
    let active_alerts = alerting.get_active_alerts().await;
    if active_alerts.is_empty() {
        println!("   No active alerts - System is healthy! ✅");
    } else {
        for alert in &active_alerts {
            println!("   {} {}: {} ({})", 
                     match alert.severity {
                         AlertSeverity::Critical => "🔴",
                         AlertSeverity::High => "🟠",
                         AlertSeverity::Medium => "🟡",
                         AlertSeverity::Low => "🟢",
                     },
                     alert.rule_name,
                     alert.message,
                     alert.triggered_at.format("%H:%M:%S"));
        }
    }
    println!();

    // Demonstrate log search
    println!("📝 Recent Log Entries:");
    let recent_logs = logging.search_logs(&LogQuery {
        level_filter: Some(LogLevel::Info),
        limit: Some(5),
        sort_order: rrag::observability::logging::SortOrder::Descending,
        ..Default::default()
    }).await;

    for log in recent_logs.iter().take(3) {
        println!("   [{}] {}: {}", 
                 log.timestamp.format("%H:%M:%S"),
                 log.component,
                 log.message);
    }
    println!();

    // Show health report
    println!("🏥 System Health Report:");
    let health_report = health.get_health_report().await;
    println!("   Overall Status: {}", health_report.overall_status);
    println!("   Components Monitored: {}", health_report.services.len());
    println!("   Active Alerts: {}", health_report.alerts.len());
    println!("   System Uptime: {} seconds", health_report.system_info.uptime_seconds);
    println!();

    // Keep the demo running
    println!("🔄 Demo is running...");
    println!("   Press Ctrl+C to stop");
    println!("   Visit http://127.0.0.1:3000 to see the dashboard");
    println!("   Monitor the console for real-time alerts and metrics");
    println!();

    // Run for demo duration
    let demo_duration = Duration::from_secs(300); // 5 minutes
    println!("⏰ Demo will run for {} seconds", demo_duration.as_secs());
    
    tokio::select! {
        _ = sleep(demo_duration) => {
            println!("\n⏰ Demo time completed!");
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\n⛔ Demo stopped by user");
        }
    }

    // Cleanup simulation
    simulation_handle.abort();

    // Generate final report
    println!("\n📋 Generating Final Demo Report...");
    let final_metrics = observability.metrics().get_all_metrics().await;
    let final_health = health.get_health_report().await;
    let alert_stats = alerting.get_alert_stats().await;
    let log_stats = logging.get_stats().await;

    println!("📊 Final Statistics:");
    println!("===================");
    println!("• Total Metrics Collected: {}", final_metrics.len());
    println!("• Total Alerts Generated: {}", alert_stats.total_active_alerts);
    println!("• Total Log Entries: {}", log_stats.total_entries);
    println!("• System Components: {}", final_health.services.len());
    println!("• Overall System Status: {}", final_health.overall_status);
    
    // Export final report
    let export_result = export.generate_and_export_report(
        ReportConfig {
            name: "RRAG Demo Final Report".to_string(),
            description: "Comprehensive report from the RRAG observability demo".to_string(),
            report_type: rrag::observability::export::ReportType::SystemHealth,
            template: None,
            parameters: HashMap::new(),
            output_format: ExportFormat::Json,
            include_charts: true,
            chart_config: rrag::observability::export::ChartConfig::default(),
        },
        rrag::observability::monitoring::SystemOverview {
            timestamp: Utc::now(),
            performance_metrics: Some(rrag::observability::monitoring::PerformanceMetrics {
                timestamp: Utc::now(),
                cpu_usage_percent: 45.0,
                memory_usage_mb: 512.0,
                memory_usage_percent: 60.0,
                disk_usage_mb: 2048.0,
                disk_usage_percent: 40.0,
                network_bytes_sent: 1000000,
                network_bytes_received: 2000000,
                active_connections: 25,
                thread_count: 50,
                gc_collections: 10,
                gc_pause_time_ms: 5.0,
            }),
            search_stats: None,
            user_stats: None,
            active_sessions: Some(15),
        },
        vec!["local_files".to_string()],
    ).await?;

    println!("📄 Final report exported: {:?}", export_result.file_path);

    // Stop observability system
    println!("\n🛑 Stopping observability system...");
    observability.stop().await?;
    
    println!("✅ RRAG Observability Dashboard Demo completed successfully!");
    println!("\n📁 Generated Files:");
    println!("   • Metrics exports in: ./exports/");
    println!("   • Log files: rrag_demo.log");
    println!("   • Archives in: ./archives/");
    
    Ok(())
}

/// Simulate realistic system activity for demonstration
async fn simulate_system_activity(
    metrics: Arc<rrag::observability::metrics::MetricsCollector>,
    monitoring: Arc<SystemMonitor>,
    logging: Arc<LogAggregator>,
    profiling: Arc<PerformanceProfiler>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    let mut request_counter = 0u64;
    let mut rng = StdRng::from_entropy();

    loop {
        interval.tick().await;
        request_counter += 1;

        // Simulate various system metrics
        let _ = metrics.inc_counter("requests_total").await;
        let _ = metrics.inc_counter_by("search_queries_total", rng.gen_range(1..5)).await;
        let _ = metrics.set_gauge("active_users", rng.gen_range(10.0..100.0)).await;
        let _ = metrics.set_gauge("system_cpu_usage_percent", rng.gen_range(20.0..80.0)).await;
        let _ = metrics.set_gauge("system_memory_usage_percent", rng.gen_range(30.0..70.0)).await;
        
        // Simulate search processing time (occasionally slow)
        let search_time = if rng.gen_bool(0.1) { 
            rng.gen_range(800.0..1500.0) // Slow query
        } else { 
            rng.gen_range(50.0..300.0) // Normal query
        };
        let _ = metrics.record_timer("search_processing_time_ms", search_time).await;
        
        // Simulate histogram metrics
        let _ = metrics.observe_histogram("response_time_ms", search_time, None).await;

        // Generate log entries
        let log_messages = vec![
            ("search_engine", "Processing search query", LogLevel::Info),
            ("vector_store", "Retrieving embeddings", LogLevel::Debug),
            ("reranker", "Applying reranking algorithm", LogLevel::Info),
            ("cache", "Cache miss for query", LogLevel::Warn),
            ("auth", "User authentication successful", LogLevel::Info),
        ];

        for (component, message, level) in &log_messages {
            if rng.gen_bool(0.3) { // 30% chance for each log
                let _ = logging.log(*level, *message, *component).await;
            }
        }

        // Occasionally generate errors
        if rng.gen_bool(0.05) { // 5% chance
            let error_messages = vec![
                ("search_engine", "Index temporarily unavailable"),
                ("vector_store", "Connection timeout"),
                ("cache", "Redis connection failed"),
                ("reranker", "Model inference timeout"),
            ];
            
            let (component, error_msg) = error_messages[rng.gen_range(0..error_messages.len())];
            let _ = logging.log(LogLevel::Error, error_msg, component).await;
            let _ = metrics.inc_counter("errors_total").await;
        }

        // Simulate user activity
        if rng.gen_bool(0.4) { // 40% chance
            let activity = rrag::observability::monitoring::UserActivity {
                timestamp: Utc::now(),
                user_id: format!("user_{}", rng.gen_range(1..100)),
                session_id: format!("session_{}", rng.gen_range(1..20)),
                action: match rng.gen_range(0..4) {
                    0 =>                     rrag::observability::monitoring::UserAction::Search,
                    1 =>                     rrag::observability::monitoring::UserAction::Chat,
                    2 =>                     rrag::observability::monitoring::UserAction::DocumentView,
                    _ =>                     rrag::observability::monitoring::UserAction::SystemHealth,
                },
                query: Some("sample query".to_string()),
                response_time_ms: search_time,
                success: rng.gen_bool(0.95), // 95% success rate
                ip_address: Some("127.0.0.1".to_string()),
                user_agent: Some("demo-client".to_string()),
            };

            let _ = monitoring.user_activity().track_activity(activity).await;
        }

        // Simulate search analytics
        if rng.gen_bool(0.6) { // 60% chance
            let search_analytics = rrag::observability::monitoring::SearchAnalytics {
                timestamp: Utc::now(),
                query: format!("demo query {}", request_counter),
                query_type: match rng.gen_range(0..4) {
                    0 =>                     rrag::observability::monitoring::QueryType::Factual,
                    1 =>                     rrag::observability::monitoring::QueryType::Conceptual,
                    2 =>                     rrag::observability::monitoring::QueryType::Procedural,
                    _ =>                     rrag::observability::monitoring::QueryType::Conversational,
                },
                results_count: rng.gen_range(1..20),
                processing_time_ms: search_time,
                success: rng.gen_bool(0.95),
                user_id: Some(format!("user_{}", rng.gen_range(1..50))),
                session_id: Some(format!("session_{}", rng.gen_range(1..10))),
                similarity_scores: vec![0.9, 0.8, 0.7, 0.6, 0.5],
                rerank_applied: rng.gen_bool(0.7),
                cache_hit: rng.gen_bool(0.3),
            };

            let _ = monitoring.search_analytics().record_search(search_analytics).await;
        }

        // Simulate performance profiling
        if rng.gen_bool(0.2) { // 20% chance
            let operation_id = format!("op_{}", request_counter);
            let _ = profiling.start_profile(&operation_id).await;
            
            // Simulate some processing time
            tokio::time::sleep(Duration::from_millis(rng.gen_range(10..200))).await;
            
            let _ = profiling.end_profile(&operation_id, "search_operation", "search_engine").await;
        }

        // Log periodic status
        if request_counter % 30 == 0 {
            let _ = logging.log(
                LogLevel::Info, 
                &format!("System status update - {} requests processed", request_counter), 
                "system"
            ).await;
        }
    }
}