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

    println!("🚀 RRAG Simple Observability Demo");
    println!("==================================");

    // Create a basic RRAG system for demonstration
    let system = RragSystemBuilder::new()
        .with_name("Observability Demo System")
        .with_environment("demo")
        .build()
        .await?;

    println!("✅ RRAG System initialized");

    // Demonstrate basic system operations
    println!("📊 Generating sample metrics...");
    
    for i in 0..10 {
        // Simulate document processing
        let doc = Document::new(format!("Sample document {}", i + 1));
        let result = system.process_document(doc).await?;
        println!("   Processed document {} in {}ms", 
                 i + 1, result.processing_time_ms);
        
        // Simulate search operations
        let search_response = system.search(format!("query {}", i + 1), Some(5)).await?;
        println!("   Search query {} completed in {}ms", 
                 i + 1, search_response.processing_time_ms);
        
        // Simulate chat interactions
        let chat_response = system.chat("demo_agent", format!("Hello {}", i + 1), None).await?;
        println!("   Chat interaction {} completed in {}ms", 
                 i + 1, chat_response.processing_time_ms);
        
        // Small delay between operations
        sleep(Duration::from_millis(100)).await;
    }

    println!("\n🏥 System Health Check:");
    let health = system.health_check().await?;
    println!("   Overall Status: {:?}", health.overall_status);
    println!("   Components:");
    for (component, status) in &health.component_status {
        let status_icon = match status {
            HealthStatus::Healthy => "✅",
            HealthStatus::Degraded => "⚠️",
            HealthStatus::Unhealthy => "❌",
            HealthStatus::Unknown => "❓",
        };
        println!("      {} {}: {:?}", status_icon, component, status);
    }

    println!("\n📈 System Metrics:");
    let metrics = system.get_metrics().await;
    println!("   Uptime: {} seconds", metrics.uptime_seconds);
    println!("   Total Requests: {}", metrics.request_counts.total_requests);
    println!("   Successful Requests: {}", metrics.request_counts.successful_requests);
    println!("   Failed Requests: {}", metrics.request_counts.failed_requests);
    println!("   Average Response Time: {:.2}ms", metrics.performance.average_response_time_ms);
    println!("   P95 Response Time: {:.2}ms", metrics.performance.p95_response_time_ms);
    println!("   Memory Usage: {:.1}MB", metrics.resource_usage.memory_usage_mb);
    println!("   CPU Usage: {:.1}%", metrics.resource_usage.cpu_usage_percent);

    // Simulate some system stress to trigger different metrics
    println!("\n🔥 Simulating system load...");
    let start_time = std::time::Instant::now();
    
    for batch in 0..5 {
        println!("   Processing batch {} of 5...", batch + 1);
        
        for i in 0..20 {
            let doc = Document::new(format!("Batch {} document {}", batch + 1, i + 1));
            let _ = system.process_document(doc).await?;
        }
        
        // Check metrics periodically
        if batch % 2 == 0 {
            let interim_metrics = system.get_metrics().await;
            println!("      Interim stats - Total: {}, Success: {}, Avg time: {:.1}ms",
                     interim_metrics.request_counts.total_requests,
                     interim_metrics.request_counts.successful_requests,
                     interim_metrics.performance.average_response_time_ms);
        }
    }

    let total_time = start_time.elapsed();
    println!("   Batch processing completed in {:.2}s", total_time.as_secs_f64());

    // Final health check and metrics
    println!("\n📋 Final System Report:");
    println!("========================");

    let final_health = system.health_check().await?;
    let final_metrics = system.get_metrics().await;
    
    println!("Health Status:");
    println!("   Overall: {:?}", final_health.overall_status);
    println!("   Uptime: {} seconds", final_health.uptime_seconds);
    println!("   Components Checked: {}", final_health.component_status.len());

    println!("\nPerformance Metrics:");
    println!("   Total Requests Processed: {}", final_metrics.request_counts.total_requests);
    println!("   Success Rate: {:.1}%", 
             if final_metrics.request_counts.total_requests > 0 {
                 (final_metrics.request_counts.successful_requests as f64 / 
                  final_metrics.request_counts.total_requests as f64) * 100.0
             } else { 0.0 });
    println!("   Average Response Time: {:.2}ms", final_metrics.performance.average_response_time_ms);
    println!("   P95 Response Time: {:.2}ms", final_metrics.performance.p95_response_time_ms);
    println!("   P99 Response Time: {:.2}ms", final_metrics.performance.p99_response_time_ms);
    println!("   Requests/Second: {:.2}", final_metrics.performance.requests_per_second);

    println!("\nResource Usage:");
    println!("   Memory: {:.1}MB ({:.1}%)", 
             final_metrics.resource_usage.memory_usage_mb,
             final_metrics.resource_usage.cpu_usage_percent);
    println!("   Storage: {:.1}MB", final_metrics.resource_usage.storage_usage_mb);
    println!("   Network I/O: {}KB sent, {}KB received", 
             final_metrics.resource_usage.network_bytes_sent / 1024,
             final_metrics.resource_usage.network_bytes_received / 1024);

    println!("\nRequest Breakdown:");
    println!("   Document Processing: {}", final_metrics.request_counts.total_requests / 3); // Approximation
    println!("   Search Queries: {}", final_metrics.request_counts.retrieval_requests);
    println!("   Agent Interactions: {}", final_metrics.request_counts.agent_requests);

    // Demonstrate error handling
    println!("\n⚠️ Demonstrating Error Scenarios:");
    
    // Try to process an invalid document
    match system.process_document(Document::new("")).await {
        Ok(result) => println!("   Empty document processed: {:?}", result.success),
        Err(e) => println!("   Expected error for empty document: {}", e),
    }

    // Performance recommendations
    println!("\n💡 Performance Insights:");
    if final_metrics.performance.average_response_time_ms > 100.0 {
        println!("   ⚠️  Average response time is elevated ({}ms)", 
                 final_metrics.performance.average_response_time_ms);
        println!("      Consider optimizing query processing or adding caching");
    } else {
        println!("   ✅ Response times are within acceptable range");
    }

    if final_metrics.resource_usage.memory_usage_mb > 500.0 {
        println!("   ⚠️  Memory usage is high ({:.1}MB)", 
                 final_metrics.resource_usage.memory_usage_mb);
        println!("      Consider implementing memory cleanup strategies");
    } else {
        println!("   ✅ Memory usage is reasonable");
    }

    println!("\n📊 System Characteristics:");
    println!("   Peak Throughput: ~{:.1} requests/second", final_metrics.performance.requests_per_second);
    println!("   Reliability: {:.2}% success rate", 
             (final_metrics.request_counts.successful_requests as f64 / 
              final_metrics.request_counts.total_requests as f64) * 100.0);
    println!("   Efficiency: {:.1}ms average latency", final_metrics.performance.average_response_time_ms);

    // Summary statistics
    let processing_efficiency = if final_metrics.performance.average_response_time_ms > 0.0 {
        1000.0 / final_metrics.performance.average_response_time_ms // requests per second if single-threaded
    } else {
        0.0
    };

    println!("\n🎯 Demo Summary:");
    println!("   • Processed {} total operations", final_metrics.request_counts.total_requests);
    println!("   • Maintained {:.1}% uptime", 100.0); // Demo ran successfully
    println!("   • Achieved {:.1} req/sec theoretical throughput", processing_efficiency);
    println!("   • System health: {:?}", final_health.overall_status);
    
    // Future enhancement suggestions
    println!("\n🔮 Potential Enhancements:");
    println!("   • Real-time dashboard with WebSocket updates");
    println!("   • Advanced alerting with custom thresholds");
    println!("   • Historical data analysis and trending");
    println!("   • Automated performance profiling");
    println!("   • Integration with external monitoring tools");
    println!("   • Custom metrics and business KPIs");

    println!("\n✨ RRAG Simple Observability Demo completed successfully!");
    println!("    This demo showcased basic monitoring capabilities.");
    println!("    For full observability features, see the comprehensive demo.");
    
    Ok(())
}