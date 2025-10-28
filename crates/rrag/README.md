# RRAG - Enterprise Rust RAG Framework

[![Crates.io](https://img.shields.io/crates/v/rrag.svg)](https://crates.io/crates/rrag)
[![Documentation](https://docs.rs/rrag/badge.svg)](https://docs.rs/rrag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/crates/d/rrag.svg)](https://crates.io/crates/rrag)
[![Rust](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org)

**RRAG** (Rust RAG) is a high-performance, enterprise-ready framework for building Retrieval-Augmented Generation applications in Rust. Built from the ground up with safety, performance, and developer experience in mind.

## 🎯 Why RRAG?

- **🚀 Native Performance**: Zero-cost abstractions with compile-time optimizations
- **🛡️ Memory Safety**: Rust's ownership system prevents data races and memory leaks
- **⚡ Async First**: Built on Tokio for maximum concurrency
- **🎯 Type Safety**: Compile-time guarantees eliminate runtime errors
- **🔌 Modular Design**: Pluggable architecture with swappable components
- **📊 Production Ready**: Built-in observability, security, and monitoring

## 🏗️ Architecture Overview

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Processing    │───▶│   Vector Store  │
│   (Input)       │    │   Pipeline      │    │   (Storage)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Responses     │◀───│     Agent       │◀───│    Retriever    │
│   (Output)      │    │   (rsllm)       │    │   (Search)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ⚡ Quick Start

Add RRAG to your `Cargo.toml`:

```toml
[dependencies]
rrag = { version = "0.1", features = ["rsllm-client"] }
tokio = { version = "1.0", features = ["full"] }
```

### Basic RAG Application

```rust
use rrag::prelude::*;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create a RAG system
    let rag = RragSystem::builder()
        .with_rsllm_client("http://localhost:8080")
        .with_vector_store(InMemoryStorage::new())
        .with_chunk_size(512)
        .build()
        .await?;

    // Add documents
    rag.ingest_documents(vec![
        Document::new("Rust is a systems programming language..."),
        Document::new("RAG combines retrieval with generation..."),
    ]).await?;

    // Query the system
    let response = rag.query("What is Rust?").await?;
    tracing::debug!("Response: {}", response.text);

    Ok(())
}
```

## 🌟 Core Features

### 🔍 Advanced Retrieval

- **Hybrid Search**: Combines semantic and keyword search with multiple fusion strategies
- **Graph-Based Retrieval**: Knowledge graph construction with entity extraction
- **Multi-Modal Support**: Process text, images, tables, charts, and documents
- **Smart Reranking**: Cross-encoder models for precise result ranking

### 🧠 Intelligent Agents

- **Tool Integration**: Built-in calculator, HTTP client, and custom tool support
- **Memory Management**: Conversation buffers, token limits, and summarization
- **Streaming Responses**: Real-time token streaming with async iterators

### ⚡ Performance & Scalability

- **Intelligent Caching**: Multi-level caching with semantic similarity
- **Incremental Indexing**: Efficient document updates without full rebuilds
- **Batch Processing**: High-throughput document ingestion

### 📊 Production Features

- **Observability Dashboard**: Real-time monitoring with web UI and metrics
- **Security & Rate Limiting**: Authentication, authorization, and abuse prevention
- **Health Checks**: Component monitoring and dependency tracking

## 📖 Documentation

Visit [docs.rs/rrag](https://docs.rs/rrag) for complete API documentation and examples.

## 🔧 Feature Flags

```toml
[dependencies.rrag]
version = "0.1"
features = [
    "rsllm-client",      # rsllm integration
    "http",              # HTTP tools and clients
    "concurrent",        # Concurrent data structures
    "multimodal",        # Multi-modal processing
    "observability",     # Monitoring and metrics
    "security",          # Authentication and rate limiting
]
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions welcome! Please see our contributing guidelines for details.
