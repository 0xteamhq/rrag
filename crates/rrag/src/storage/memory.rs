//! # Memory Trait - Abstract memory interface for RRAG storage
//!
//! This module provides the core Memory trait that abstracts over different storage backends.
//! All storage implementations (in-memory, database, etc.) implement this trait.

use crate::{RragError, RragResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a value that can be stored in memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryValue {
    /// String value
    String(String),

    /// Integer value
    Integer(i64),

    /// Float value
    Float(f64),

    /// Boolean value
    Boolean(bool),

    /// JSON value
    Json(serde_json::Value),

    /// Binary data
    Bytes(Vec<u8>),

    /// List of values
    List(Vec<MemoryValue>),

    /// Map of values
    Map(HashMap<String, MemoryValue>),
}

impl MemoryValue {
    /// Convert to string if possible
    pub fn as_string(&self) -> Option<&str> {
        match self {
            MemoryValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Convert to integer if possible
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            MemoryValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Convert to float if possible
    pub fn as_float(&self) -> Option<f64> {
        match self {
            MemoryValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Convert to boolean if possible
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            MemoryValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Convert to JSON if possible
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            MemoryValue::Json(j) => Some(j),
            _ => None,
        }
    }

    /// Convert to bytes if possible
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            MemoryValue::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

impl From<String> for MemoryValue {
    fn from(s: String) -> Self {
        MemoryValue::String(s)
    }
}

impl From<&str> for MemoryValue {
    fn from(s: &str) -> Self {
        MemoryValue::String(s.to_string())
    }
}

impl From<i64> for MemoryValue {
    fn from(i: i64) -> Self {
        MemoryValue::Integer(i)
    }
}

impl From<f64> for MemoryValue {
    fn from(f: f64) -> Self {
        MemoryValue::Float(f)
    }
}

impl From<bool> for MemoryValue {
    fn from(b: bool) -> Self {
        MemoryValue::Boolean(b)
    }
}

impl From<serde_json::Value> for MemoryValue {
    fn from(j: serde_json::Value) -> Self {
        MemoryValue::Json(j)
    }
}

impl From<Vec<u8>> for MemoryValue {
    fn from(b: Vec<u8>) -> Self {
        MemoryValue::Bytes(b)
    }
}

/// Query options for memory operations
#[derive(Debug, Clone, Default)]
pub struct MemoryQuery {
    /// Key pattern/prefix to match
    pub key_pattern: Option<String>,

    /// Namespace/collection filter
    pub namespace: Option<String>,

    /// Maximum results
    pub limit: Option<usize>,

    /// Skip first N results
    pub offset: Option<usize>,

    /// Sort order
    pub sort_order: Option<SortOrder>,
}

#[derive(Debug, Clone)]
pub enum SortOrder {
    /// Sort by key ascending
    KeyAsc,
    /// Sort by key descending
    KeyDesc,
    /// Sort by creation time ascending
    CreatedAsc,
    /// Sort by creation time descending
    CreatedDesc,
}

impl MemoryQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.key_pattern = Some(pattern.into());
        self
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// Core Memory trait - abstract interface for all storage backends
#[async_trait]
pub trait Memory: Send + Sync {
    /// Get the name of this memory backend
    fn backend_name(&self) -> &str;

    /// Set a value in memory
    async fn set(&self, key: &str, value: MemoryValue) -> RragResult<()>;

    /// Get a value from memory
    async fn get(&self, key: &str) -> RragResult<Option<MemoryValue>>;

    /// Delete a value from memory
    async fn delete(&self, key: &str) -> RragResult<bool>;

    /// Check if a key exists
    async fn exists(&self, key: &str) -> RragResult<bool>;

    /// List all keys matching a query
    async fn keys(&self, query: &MemoryQuery) -> RragResult<Vec<String>>;

    /// Get multiple values at once
    async fn mget(&self, keys: &[String]) -> RragResult<Vec<Option<MemoryValue>>>;

    /// Set multiple values at once
    async fn mset(&self, pairs: &[(String, MemoryValue)]) -> RragResult<()>;

    /// Delete multiple keys at once
    async fn mdelete(&self, keys: &[String]) -> RragResult<usize>;

    /// Clear all data (with optional namespace)
    async fn clear(&self, namespace: Option<&str>) -> RragResult<()>;

    /// Get count of keys
    async fn count(&self, namespace: Option<&str>) -> RragResult<usize>;

    /// Check if memory backend is healthy
    async fn health_check(&self) -> RragResult<bool>;

    /// Get memory statistics
    async fn stats(&self) -> RragResult<MemoryStats>;
}

/// Memory backend statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of keys
    pub total_keys: usize,

    /// Estimated memory usage in bytes
    pub memory_bytes: u64,

    /// Backend type
    pub backend_type: String,

    /// Number of namespaces
    pub namespace_count: usize,

    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,

    /// Additional backend-specific stats
    pub extra: HashMap<String, serde_json::Value>,
}
