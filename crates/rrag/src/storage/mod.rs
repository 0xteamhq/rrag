//! # RRAG Storage Module
//!
//! This module provides a unified storage abstraction for RRAG with multiple backend implementations.
//!
//! ## Architecture
//!
//! - **Memory Trait**: Abstract interface for all storage backends
//! - **InMemoryStorage**: Fast, thread-safe in-memory implementation
//! - **DatabaseStorage**: Persistent storage using Toasty ORM (requires `database` feature)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use rrag::storage::{Memory, InMemoryStorage, MemoryValue};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create in-memory storage
//! let storage = InMemoryStorage::new();
//!
//! // Store a value
//! storage.set("user:123", MemoryValue::from("John Doe")).await?;
//!
//! // Retrieve a value
//! let value = storage.get("user:123").await?;
//! println!("User: {:?}", value);
//!
//! // Use namespaces
//! storage.set("session::abc123", MemoryValue::from(true)).await?;
//! let count = storage.count(Some("session")).await?;
//! println!("Active sessions: {}", count);
//! # Ok(())
//! # }
//! ```
//!
//! ## Database Backend - ⚠️ EXPERIMENTAL
//!
//! Enable the `database` feature to use database storage (currently experimental):
//!
//! ```toml
//! rrag = { version = "0.1", features = ["database"] }
//! ```
//!
//! ```rust,no_run
//! # #[cfg(feature = "database")]
//! use rrag::storage::{DatabaseStorage, DatabaseConfig};
//!
//! # #[cfg(feature = "database")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = DatabaseConfig {
//!     connection_string: "sqlite:memory.db".to_string(),
//!     ..Default::default()
//! };
//!
//! let storage = DatabaseStorage::with_config(config).await?;
//! # Ok(())
//! # }
//! ```
//!
//! **⚠️ WARNING**: `DatabaseStorage` currently uses in-memory fallback because:
//! - Toasty ORM (v0.1.1) is in early incubation and not production-ready
//! - Data is NOT persisted to disk despite the configuration
//! - For production persistence, use `sqlx` or `diesel` directly
//!
//! See [`database`](database/index.html) module for full details and migration path.

pub mod memory;
pub use memory::{Memory, MemoryQuery, MemoryStats, MemoryValue, SortOrder};

pub mod in_memory;
pub use in_memory::{InMemoryConfig, InMemoryStorage};

pub mod database;
#[cfg(feature = "database")]
pub use database::{DatabaseConfig, DatabaseStorage};

// Re-export the original storage types for backward compatibility
pub use crate::storage_legacy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_value_conversions() {
        // Test string conversion
        let val = MemoryValue::from("test");
        assert_eq!(val.as_string().unwrap(), "test");

        // Test integer conversion
        let val = MemoryValue::from(42i64);
        assert_eq!(val.as_integer().unwrap(), 42);

        // Test float conversion
        let val = MemoryValue::from(3.14f64);
        assert_eq!(val.as_float().unwrap(), 3.14);

        // Test boolean conversion
        let val = MemoryValue::from(true);
        assert_eq!(val.as_boolean().unwrap(), true);
    }

    #[tokio::test]
    async fn test_in_memory_storage() {
        let storage = InMemoryStorage::new();

        // Basic operations
        storage
            .set("test", MemoryValue::from("value"))
            .await
            .unwrap();
        assert!(storage.exists("test").await.unwrap());

        let val = storage.get("test").await.unwrap();
        assert_eq!(val.unwrap().as_string().unwrap(), "value");

        // Delete
        assert!(storage.delete("test").await.unwrap());
        assert!(!storage.exists("test").await.unwrap());
    }

    #[tokio::test]
    async fn test_namespace_operations() {
        let storage = InMemoryStorage::new();

        // Add namespaced keys
        storage
            .set("users::alice", MemoryValue::from("Alice"))
            .await
            .unwrap();
        storage
            .set("users::bob", MemoryValue::from("Bob"))
            .await
            .unwrap();
        storage
            .set("sessions::abc", MemoryValue::from("active"))
            .await
            .unwrap();

        // Count by namespace
        assert_eq!(storage.count(Some("users")).await.unwrap(), 2);
        assert_eq!(storage.count(Some("sessions")).await.unwrap(), 1);

        // Clear namespace
        storage.clear(Some("users")).await.unwrap();
        assert_eq!(storage.count(Some("users")).await.unwrap(), 0);
        assert_eq!(storage.count(Some("sessions")).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_bulk_operations() {
        let storage = InMemoryStorage::new();

        // Bulk set
        let pairs = vec![
            ("key1".to_string(), MemoryValue::from(1i64)),
            ("key2".to_string(), MemoryValue::from(2i64)),
            ("key3".to_string(), MemoryValue::from(3i64)),
        ];
        storage.mset(&pairs).await.unwrap();

        // Bulk get
        let keys = vec!["key1".to_string(), "key2".to_string(), "key3".to_string()];
        let values = storage.mget(&keys).await.unwrap();
        assert_eq!(values.len(), 3);
        assert!(values.iter().all(|v| v.is_some()));

        // Bulk delete
        let deleted = storage.mdelete(&keys).await.unwrap();
        assert_eq!(deleted, 3);
    }

    #[tokio::test]
    async fn test_query_operations() {
        let storage = InMemoryStorage::new();

        // Add test data
        storage
            .set("user:1", MemoryValue::from("Alice"))
            .await
            .unwrap();
        storage
            .set("user:2", MemoryValue::from("Bob"))
            .await
            .unwrap();
        storage
            .set("post:1", MemoryValue::from("Post 1"))
            .await
            .unwrap();

        // Query with pattern
        let query = MemoryQuery::new().with_pattern("user:");
        let keys = storage.keys(&query).await.unwrap();
        assert_eq!(keys.len(), 2);

        // Query with limit
        let query = MemoryQuery::new().with_limit(1);
        let keys = storage.keys(&query).await.unwrap();
        assert_eq!(keys.len(), 1);
    }

    #[tokio::test]
    async fn test_stats() {
        let storage = InMemoryStorage::new();

        // Add some data
        storage
            .set("test1", MemoryValue::from("value1"))
            .await
            .unwrap();
        storage
            .set("test2", MemoryValue::from("value2"))
            .await
            .unwrap();

        let stats = storage.stats().await.unwrap();
        assert_eq!(stats.total_keys, 2);
        assert_eq!(stats.backend_type, "in_memory");
        assert!(stats.memory_bytes > 0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let storage = InMemoryStorage::new();
        assert!(storage.health_check().await.unwrap());
    }
}
