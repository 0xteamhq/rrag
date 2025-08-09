//! # Enhanced Retrieval Module
//! 
//! Provides multiple retrieval strategies including hybrid search combining
//! semantic and keyword-based approaches for optimal performance.

pub mod hybrid;
pub mod bm25;
pub mod semantic;
pub mod fusion;

pub use hybrid::{HybridRetriever, HybridConfig, FusionStrategy};
pub use bm25::{BM25Retriever, BM25Config, TokenizerType};
pub use semantic::{SemanticRetriever, SemanticConfig};
pub use fusion::{RankFusion, ReciprocalRankFusion, WeightedFusion};

// Re-export core retrieval types from parent module
pub use crate::retrieval_core::{
    Retriever, RetrievalService, SearchResult, SearchQuery, 
    QueryType, SearchConfig, SearchAlgorithm, InMemoryRetriever
};