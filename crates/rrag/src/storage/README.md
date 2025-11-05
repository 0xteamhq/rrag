# RRAG Storage Module

Unified memory abstraction for RRAG with multiple backend implementations.

## Overview

The storage module provides a trait-based abstraction (`Memory`) that allows different storage backends to be used interchangeably throughout RRAG.

## Available Backends

### ✅ InMemoryStorage (Production Ready)

Fast, thread-safe in-memory storage using `HashMap` with `RwLock`.

**Features**:
- Thread-safe concurrent access
- Configurable limits (max keys, max memory)
- Namespace support
- Bulk operations
- Memory usage tracking
- **Production ready and recommended**

**Usage**:
```rust
use rrag::storage::{InMemoryStorage, Memory, MemoryValue};

let storage = InMemoryStorage::new();
storage.set("key", MemoryValue::from("value")).await?;
```

### ⚠️ DatabaseStorage (Experimental - NOT Production Ready)

Database-backed persistent storage using Toasty ORM.

**Current Status**: **USES IN-MEMORY FALLBACK**

**Why Experimental**:
- Toasty ORM v0.1.1 is in early incubation (stated by Tokio team)
- API is unstable and subject to change
- Limited documentation and examples
- Not recommended for production use
- Data is NOT actually persisted despite configuration

**What Works**:
- ✅ Memory trait interface
- ✅ All operations (via in-memory fallback)
- ✅ Configuration API

**What Doesn't Work**:
- ❌ Actual database persistence
- ❌ Data survival across restarts
- ❌ Multi-process data sharing

**When to Use**:
- Experimentation only
- Preparing for future Toasty stability
- Testing storage abstraction

**Production Alternatives**:
1. **Use `InMemoryStorage`** for development/testing
2. **Integrate `sqlx`** for async PostgreSQL/MySQL/SQLite
3. **Integrate `diesel`** for sync database access
4. **Wait for Toasty v1.0+** for stable ORM

**Usage** (for testing/experimentation only):
```rust
#[cfg(feature = "database")]
use rrag::storage::{DatabaseStorage, DatabaseConfig};

let config = DatabaseConfig {
    connection_string: "sqlite:memory.db".to_string(),
    ..Default::default()
};

// WARNING: This will log a warning and use in-memory fallback
let storage = DatabaseStorage::with_config(config).await?;
```

## Migration Path to Production Database

When you need actual database persistence, here are your options:

### Option 1: Implement Custom Storage with sqlx

```rust
use rrag::storage::{Memory, MemoryQuery, MemoryStats, MemoryValue};
use sqlx::{SqlitePool, Row};

pub struct SqlxStorage {
    pool: SqlitePool,
}

#[async_trait::async_trait]
impl Memory for SqlxStorage {
    fn backend_name(&self) -> &str { "sqlx_sqlite" }

    async fn set(&self, key: &str, value: MemoryValue) -> RragResult<()> {
        let json = serde_json::to_string(&value)?;
        sqlx::query("INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)")
            .bind(key)
            .bind(json)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ... implement other methods
}
```

### Option 2: Wait for Toasty Stable Release

Monitor Toasty development:
- GitHub: https://github.com/tokio-rs/toasty
- Crates.io: https://crates.io/crates/toasty
- Tokio Blog: https://tokio.rs/blog/

When Toasty reaches v1.0:
1. Update dependency: `toasty = "1.0"`
2. Implement proper models with `#[toasty::model]`
3. Remove in-memory fallback
4. Update `database.rs` with real implementation

### Option 3: Use Diesel ORM

```rust
use diesel::prelude::*;
use rrag::storage::{Memory, MemoryValue};

pub struct DieselStorage {
    conn: SqliteConnection,
}

// Similar implementation to sqlx option
```

## Testing

```bash
# Test in-memory storage (production ready)
cargo test -p rrag storage::in_memory --features rsllm-client

# Run storage demo
cargo run -p rrag --example storage_demo --features rsllm-client
```

## Future Roadmap

- [ ] Toasty integration when stable (v1.0+)
- [ ] Redis backend for distributed caching
- [ ] S3/Object storage backend for large values
- [ ] Encryption at rest support
- [ ] Compression for large values
- [ ] TTL (time-to-live) support
- [ ] Transactions support

## References

- [Memory Trait Documentation](memory/trait.Memory.html)
- [InMemoryStorage Documentation](in_memory/struct.InMemoryStorage.html)
- [DatabaseStorage Documentation](database/struct.DatabaseStorage.html)
- [Toasty ORM](https://github.com/tokio-rs/toasty)
- [Toasty Announcement](https://tokio.rs/blog/2024-10-23-announcing-toasty)
