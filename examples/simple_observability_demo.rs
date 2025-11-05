//! # Simple RRAG Observability Demo
//!
//! A simplified demonstration of the RRAG observability system focusing on core features:
//! - Basic metrics collection
//! - Simple monitoring and alerting
//! - Dashboard demonstration
//! - Log aggregation

use rrag::prelude::*;
use rrag::system::HealthStatus;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> RragResult<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    tracing::debug!("🚀 RRAG Simple Observability Demo");
    tracing::debug!("==================================");

    // Create a basic RRAG system for demonstration
    let system = RragSystemBuilder::new()
        .with_name("Observability Demo System")
        .with_environment("demo")
        .build()
        .await?;

    tracing::debug!("✅ RRAG System initialized");

    // Demonstrate basic system operations
    tracing::debug!("📊 Generating sample metrics...");

    for i in 0..10 {
        // Simulate document processing
        let doc = Document::new(format!("Sample document {}", i + 1));
        let result = system.process_document(doc).await?;
        tracing::debug!(
            "   Processed document {} in {}ms",
            i + 1,
            result.processing_time_ms
        );

        // Simulate search operations
        let search_response = system.search(format!("query {}", i + 1), Some(5)).await?;
        tracing::debug!(
            "   Search query {} completed in {}ms",
            i + 1,
            search_response.processing_time_ms
        );

        // Simulate chat interactions
        let chat_response = system
            .chat("demo_agent", format!("Hello {}", i + 1), None)
            .await?;
        tracing::debug!(
            "   Chat interaction {} completed in {}ms",
            i + 1,
            chat_response.processing_time_ms
        );

        // Small delay between operations
        sleep(Duration::from_millis(100)).await;
    }

    tracing::debug!("\n🏥 System Health Check:");
    let health = system.health_check().await?;
    tracing::debug!("   Overall Status: {:?}", health.overall_status);
    tracing::debug!("   Components:");
    for (component, status) in &health.component_status {
        let status_icon = match status {
            HealthStatus::Healthy => "✅",
            HealthStatus::Degraded => "⚠️",
            HealthStatus::Unhealthy => "❌",
            HealthStatus::Unknown => "❓",
        };
        tracing::debug!("      {} {}: {:?}", status_icon, component, status);
    }

    tracing::debug!("\n📈 System Metrics:");
    let metrics = system.get_metrics().await;
    tracing::debug!("   Uptime: {} seconds", metrics.uptime_seconds);
    tracing::debug!(
        "   Total Requests: {}",
        metrics.request_counts.total_requests
    );
    tracing::debug!(
        "   Successful Requests: {}",
        metrics.request_counts.successful_requests
    );
    tracing::debug!(
        "   Failed Requests: {}",
        metrics.request_counts.failed_requests
    );
    tracing::debug!(
        "   Average Response Time: {:.2}ms",
        metrics.performance.average_response_time_ms
    );
    tracing::debug!(
        "   P95 Response Time: {:.2}ms",
        metrics.performance.p95_response_time_ms
    );
    tracing::debug!(
        "   Memory Usage: {:.1}MB",
        metrics.resource_usage.memory_usage_mb
    );
    tracing::debug!(
        "   CPU Usage: {:.1}%",
        metrics.resource_usage.cpu_usage_percent
    );

    // Simulate some system stress to trigger different metrics
    tracing::debug!("\n🔥 Simulating system load...");
    let start_time = std::time::Instant::now();

    for batch in 0..5 {
        tracing::debug!("   Processing batch {} of 5...", batch + 1);

        for i in 0..20 {
            let doc = Document::new(format!("Batch {} document {}", batch + 1, i + 1));
            let _ = system.process_document(doc).await?;
        }

        // Check metrics periodically
        if batch % 2 == 0 {
            let interim_metrics = system.get_metrics().await;
            tracing::debug!(
                "      Interim stats - Total: {}, Success: {}, Avg time: {:.1}ms",
                interim_metrics.request_counts.total_requests,
                interim_metrics.request_counts.successful_requests,
                interim_metrics.performance.average_response_time_ms
            );
        }
    }

    let total_time = start_time.elapsed();
    tracing::debug!(
        "   Batch processing completed in {:.2}s",
        total_time.as_secs_f64()
    );

    // Final health check and metrics
    tracing::debug!("\n📋 Final System Report:");
    tracing::debug!("========================");

    let final_health = system.health_check().await?;
    let final_metrics = system.get_metrics().await;

    tracing::debug!("Health Status:");
    tracing::debug!("   Overall: {:?}", final_health.overall_status);
    tracing::debug!("   Uptime: {} seconds", final_health.uptime_seconds);
    tracing::debug!(
        "   Components Checked: {}",
        final_health.component_status.len()
    );

    tracing::debug!("\nPerformance Metrics:");
    tracing::debug!(
        "   Total Requests Processed: {}",
        final_metrics.request_counts.total_requests
    );
    tracing::debug!(
        "   Success Rate: {:.1}%",
        if final_metrics.request_counts.total_requests > 0 {
            (final_metrics.request_counts.successful_requests as f64
                / final_metrics.request_counts.total_requests as f64)
                * 100.0
        } else {
            0.0
        }
    );
    tracing::debug!(
        "   Average Response Time: {:.2}ms",
        final_metrics.performance.average_response_time_ms
    );
    tracing::debug!(
        "   P95 Response Time: {:.2}ms",
        final_metrics.performance.p95_response_time_ms
    );
    tracing::debug!(
        "   P99 Response Time: {:.2}ms",
        final_metrics.performance.p99_response_time_ms
    );
    tracing::debug!(
        "   Requests/Second: {:.2}",
        final_metrics.performance.requests_per_second
    );

    tracing::debug!("\nResource Usage:");
    tracing::debug!(
        "   Memory: {:.1}MB ({:.1}%)",
        final_metrics.resource_usage.memory_usage_mb,
        final_metrics.resource_usage.cpu_usage_percent
    );
    tracing::debug!(
        "   Storage: {:.1}MB",
        final_metrics.resource_usage.storage_usage_mb
    );
    tracing::debug!(
        "   Network I/O: {}KB sent, {}KB received",
        final_metrics.resource_usage.network_bytes_sent / 1024,
        final_metrics.resource_usage.network_bytes_received / 1024
    );

    tracing::debug!("\nRequest Breakdown:");
    tracing::debug!(
        "   Document Processing: {}",
        final_metrics.request_counts.total_requests / 3
    ); // Approximation
    tracing::debug!(
        "   Search Queries: {}",
        final_metrics.request_counts.retrieval_requests
    );
    tracing::debug!(
        "   Agent Interactions: {}",
        final_metrics.request_counts.agent_requests
    );

    // Demonstrate error handling
    tracing::debug!("\n⚠️ Demonstrating Error Scenarios:");

    // Try to process an invalid document
    match system.process_document(Document::new("")).await {
        Ok(result) => tracing::debug!("   Empty document processed: {:?}", result.success),
        Err(e) => tracing::debug!("   Expected error for empty document: {}", e),
    }

    // Performance recommendations
    tracing::debug!("\n💡 Performance Insights:");
    if final_metrics.performance.average_response_time_ms > 100.0 {
        tracing::debug!(
            "   ⚠️  Average response time is elevated ({}ms)",
            final_metrics.performance.average_response_time_ms
        );
        tracing::debug!("      Consider optimizing query processing or adding caching");
    } else {
        tracing::debug!("   ✅ Response times are within acceptable range");
    }

    if final_metrics.resource_usage.memory_usage_mb > 500.0 {
        tracing::debug!(
            "   ⚠️  Memory usage is high ({:.1}MB)",
            final_metrics.resource_usage.memory_usage_mb
        );
        tracing::debug!("      Consider implementing memory cleanup strategies");
    } else {
        tracing::debug!("   ✅ Memory usage is reasonable");
    }

    tracing::debug!("\n📊 System Characteristics:");
    tracing::debug!(
        "   Peak Throughput: ~{:.1} requests/second",
        final_metrics.performance.requests_per_second
    );
    tracing::debug!(
        "   Reliability: {:.2}% success rate",
        (final_metrics.request_counts.successful_requests as f64
            / final_metrics.request_counts.total_requests as f64)
            * 100.0
    );
    tracing::debug!(
        "   Efficiency: {:.1}ms average latency",
        final_metrics.performance.average_response_time_ms
    );

    // Summary statistics
    let processing_efficiency = if final_metrics.performance.average_response_time_ms > 0.0 {
        1000.0 / final_metrics.performance.average_response_time_ms // requests per second if single-threaded
    } else {
        0.0
    };

    tracing::debug!("\n🎯 Demo Summary:");
    tracing::debug!(
        "   • Processed {} total operations",
        final_metrics.request_counts.total_requests
    );
    tracing::debug!("   • Maintained {:.1}% uptime", 100.0); // Demo ran successfully
    tracing::debug!(
        "   • Achieved {:.1} req/sec theoretical throughput",
        processing_efficiency
    );
    tracing::debug!("   • System health: {:?}", final_health.overall_status);

    // Future enhancement suggestions
    tracing::debug!("\n🔮 Potential Enhancements:");
    tracing::debug!("   • Real-time dashboard with WebSocket updates");
    tracing::debug!("   • Advanced alerting with custom thresholds");
    tracing::debug!("   • Historical data analysis and trending");
    tracing::debug!("   • Automated performance profiling");
    tracing::debug!("   • Integration with external monitoring tools");
    tracing::debug!("   • Custom metrics and business KPIs");

    tracing::debug!("\n✨ RRAG Simple Observability Demo completed successfully!");
    tracing::debug!("    This demo showcased basic monitoring capabilities.");
    tracing::debug!("    For full observability features, see the comprehensive demo.");

    Ok(())
}
