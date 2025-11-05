//! # Storage Demo
//!
//! Demonstrates the new Memory trait-based storage system in RRAG.
//!
//! This example shows:
//! - Using InMemoryStorage with the Memory trait
//! - Storing different types of values
//! - Using namespaces for organization
//! - Bulk operations (mset, mget, mdelete)
//! - Querying and filtering keys
//! - Getting storage statistics

use rrag::storage::{InMemoryStorage, Memory, MemoryQuery, MemoryValue};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RRAG Storage System Demo ===\n");

    // Create in-memory storage
    let storage = InMemoryStorage::new();
    println!("Created {} storage\n", storage.backend_name());

    // === Basic Operations ===
    println!("--- Basic Operations ---");

    // Store different types of values
    storage
        .set("user:name", MemoryValue::from("Alice"))
        .await?;
    storage.set("user:age", MemoryValue::from(30i64)).await?;
    storage
        .set("user:premium", MemoryValue::from(true))
        .await?;
    storage.set("user:score", MemoryValue::from(95.5f64)).await?;

    println!("✓ Stored 4 user values");

    // Retrieve values
    if let Some(name) = storage.get("user:name").await? {
        println!("  Name: {}", name.as_string().unwrap());
    }

    if let Some(age) = storage.get("user:age").await? {
        println!("  Age: {}", age.as_integer().unwrap());
    }

    if let Some(premium) = storage.get("user:premium").await? {
        println!("  Premium: {}", premium.as_boolean().unwrap());
    }

    if let Some(score) = storage.get("user:score").await? {
        println!("  Score: {}\n", score.as_float().unwrap());
    }

    // === Namespace Operations ===
    println!("--- Namespace Operations ---");

    // Store values in different namespaces
    storage
        .set("session::abc123", MemoryValue::from("active"))
        .await?;
    storage
        .set("session::xyz789", MemoryValue::from("expired"))
        .await?;
    storage
        .set("cache::homepage", MemoryValue::from("<!DOCTYPE html>..."))
        .await?;
    storage
        .set("cache::about", MemoryValue::from("<!DOCTYPE html>..."))
        .await?;

    println!("✓ Stored values in 'session' and 'cache' namespaces");

    // Count by namespace
    let session_count = storage.count(Some("session")).await?;
    let cache_count = storage.count(Some("cache")).await?;
    println!("  Sessions: {}", session_count);
    println!("  Cache entries: {}\n", cache_count);

    // === Bulk Operations ===
    println!("--- Bulk Operations ---");

    // Bulk set (mset)
    let products = vec![
        ("product:1".to_string(), MemoryValue::from("Laptop")),
        ("product:2".to_string(), MemoryValue::from("Mouse")),
        ("product:3".to_string(), MemoryValue::from("Keyboard")),
    ];
    storage.mset(&products).await?;
    println!("✓ Bulk stored 3 products");

    // Bulk get (mget)
    let product_keys = vec![
        "product:1".to_string(),
        "product:2".to_string(),
        "product:3".to_string(),
    ];
    let values = storage.mget(&product_keys).await?;
    println!("✓ Bulk retrieved {} products:", values.len());
    for (i, val) in values.iter().enumerate() {
        if let Some(v) = val {
            println!("  Product {}: {}", i + 1, v.as_string().unwrap());
        }
    }
    println!();

    // === Querying Keys ===
    println!("--- Querying Keys ---");

    // Query with pattern
    let query = MemoryQuery::new().with_pattern("user:");
    let user_keys = storage.keys(&query).await?;
    println!("✓ Found {} keys matching 'user:' pattern", user_keys.len());
    for key in &user_keys {
        println!("  - {}", key);
    }
    println!();

    // Query with namespace
    let query = MemoryQuery::new().with_namespace("session");
    let session_keys = storage.keys(&query).await?;
    println!(
        "✓ Found {} keys in 'session' namespace",
        session_keys.len()
    );
    for key in &session_keys {
        println!("  - {}", key);
    }
    println!();

    // Query with limit
    let query = MemoryQuery::new().with_limit(3);
    let limited_keys = storage.keys(&query).await?;
    println!("✓ Query with limit returned {} keys", limited_keys.len());
    println!();

    // === Storage Statistics ===
    println!("--- Storage Statistics ---");

    let stats = storage.stats().await?;
    println!("Backend: {}", stats.backend_type);
    println!("Total keys: {}", stats.total_keys);
    println!("Memory usage: {} bytes", stats.memory_bytes);
    println!("Namespaces: {}", stats.namespace_count);
    println!("Last updated: {}\n", stats.last_updated);

    // === Cleanup Operations ===
    println!("--- Cleanup Operations ---");

    // Delete a single key
    let deleted = storage.delete("user:score").await?;
    println!("✓ Deleted 'user:score': {}", deleted);

    // Bulk delete
    let to_delete = vec!["product:1".to_string(), "product:2".to_string()];
    let deleted_count = storage.mdelete(&to_delete).await?;
    println!("✓ Bulk deleted {} products", deleted_count);

    // Clear a namespace
    storage.clear(Some("session")).await?;
    let session_count_after = storage.count(Some("session")).await?;
    println!(
        "✓ Cleared 'session' namespace (now has {} keys)",
        session_count_after
    );
    println!();

    // === Final Stats ===
    println!("--- Final Statistics ---");
    let final_stats = storage.stats().await?;
    println!("Total keys remaining: {}", final_stats.total_keys);
    println!(
        "Memory usage: {} bytes ({} KB)",
        final_stats.memory_bytes,
        final_stats.memory_bytes / 1024
    );
    println!();

    // === Health Check ===
    let healthy = storage.health_check().await?;
    println!("Storage health check: {}", if healthy { "✓ OK" } else { "✗ FAIL" });

    println!("\n=== Demo Complete ===");

    Ok(())
}
