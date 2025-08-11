//! # Enhanced Retrieval Module
//!
//! Provides multiple retrieval strategies including hybrid search combining
//! semantic and keyword-based approaches for optimal performance.

pub mod bm25;
pub mod fusion;
pub mod hybrid;
pub mod semantic;

pub use bm25::{BM25Config, BM25Retriever, TokenizerType};
pub use fusion::{RankFusion, ReciprocalRankFusion, WeightedFusion};
pub use hybrid::{FusionStrategy, HybridConfig, HybridRetriever};
pub use semantic::{SemanticConfig, SemanticRetriever};

// Re-export core retrieval types from parent module
pub use crate::retrieval_core::{
    InMemoryRetriever, QueryType, RetrievalService, Retriever, SearchAlgorithm, SearchConfig,
    SearchQuery, SearchResult,
};
