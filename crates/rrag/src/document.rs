//! # RRAG Document Types
//!
//! Core document handling with zero-copy optimizations and efficient processing.
//! Designed for Rust's ownership system and memory efficiency.

use crate::{RragError, RragResult};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use uuid::Uuid;

/// Document metadata using Cow for zero-copy string handling
pub type Metadata = HashMap<String, serde_json::Value>;

/// Core document type optimized for Rust patterns
///
/// Uses `Cow<str>` for flexible string handling:
/// - Borrowed strings when possible (zero-copy)
/// - Owned strings when necessary (after processing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: String,

    /// Document content - uses Cow for efficient string handling
    #[serde(with = "cow_str_serde")]
    pub content: Cow<'static, str>,

    /// Document metadata
    pub metadata: Metadata,

    /// Content hash for deduplication
    pub content_hash: Option<String>,

    /// Document creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Document {
    /// Create a new document with generated ID
    pub fn new(content: impl Into<Cow<'static, str>>) -> Self {
        let content = content.into();
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            metadata: HashMap::new(),
            content_hash: None,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create document with specific ID
    pub fn with_id(id: impl Into<String>, content: impl Into<Cow<'static, str>>) -> Self {
        let content = content.into();
        Self {
            id: id.into(),
            content,
            metadata: HashMap::new(),
            content_hash: None,
            created_at: chrono::Utc::now(),
        }
    }

    /// Add metadata using builder pattern
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Add multiple metadata entries
    pub fn with_metadata_map(mut self, metadata: Metadata) -> Self {
        self.metadata.extend(metadata);
        self
    }

    /// Generate content hash for deduplication
    pub fn with_content_hash(mut self) -> Self {
        self.content_hash = Some(Self::hash_content(&self.content));
        self
    }

    /// Get content as string slice
    pub fn content_str(&self) -> &str {
        &self.content
    }

    /// Get content length in characters
    pub fn content_length(&self) -> usize {
        self.content.chars().count()
    }

    /// Check if document is empty
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }

    /// Generate hash for content deduplication
    fn hash_content(content: &str) -> String {
        // Simple hash implementation - in production, use a proper hash function
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Document chunk for processing pipelines
///
/// Represents a portion of a document with positional information
/// and overlap handling for better context preservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Reference to parent document ID
    pub document_id: String,

    /// Chunk content
    pub content: String,

    /// Chunk index within the document
    pub chunk_index: usize,

    /// Character start position in original document
    pub start_position: usize,

    /// Character end position in original document
    pub end_position: usize,

    /// Overlap with previous chunk (characters)
    pub overlap_previous: usize,

    /// Overlap with next chunk (characters)
    pub overlap_next: usize,

    /// Chunk metadata (inherited from document + chunk-specific)
    pub metadata: Metadata,
}

impl DocumentChunk {
    /// Create a new document chunk
    pub fn new(
        document_id: impl Into<String>,
        content: impl Into<String>,
        chunk_index: usize,
        start_position: usize,
        end_position: usize,
    ) -> Self {
        Self {
            document_id: document_id.into(),
            content: content.into(),
            chunk_index,
            start_position,
            end_position,
            overlap_previous: 0,
            overlap_next: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set overlap information
    pub fn with_overlap(mut self, previous: usize, next: usize) -> Self {
        self.overlap_previous = previous;
        self.overlap_next = next;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Get chunk length
    pub fn length(&self) -> usize {
        self.content.len()
    }

    /// Check if chunk is empty
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }
}

/// Document chunking strategy
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    /// Fixed size chunking with overlap
    FixedSize { 
        /// Size of each chunk in characters
        size: usize, 
        /// Number of characters to overlap between chunks
        overlap: usize 
    },

    /// Sentence-based chunking
    Sentence {
        /// Maximum number of sentences per chunk
        max_sentences: usize,
        /// Number of sentences to overlap between chunks
        overlap_sentences: usize,
    },

    /// Paragraph-based chunking
    Paragraph { 
        /// Maximum number of paragraphs per chunk
        max_paragraphs: usize 
    },

    /// Semantic chunking (requires embeddings)
    Semantic { 
        /// Similarity threshold for semantic boundaries
        similarity_threshold: f32 
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::FixedSize {
            size: 512,
            overlap: 64,
        }
    }
}

/// Document chunker with various strategies
pub struct DocumentChunker {
    strategy: ChunkingStrategy,
}

impl DocumentChunker {
    /// Create chunker with default strategy
    pub fn new() -> Self {
        Self {
            strategy: ChunkingStrategy::default(),
        }
    }

    /// Create chunker with specific strategy
    pub fn with_strategy(strategy: ChunkingStrategy) -> Self {
        Self { strategy }
    }

    /// Chunk a document into smaller pieces
    pub fn chunk_document(&self, document: &Document) -> RragResult<Vec<DocumentChunk>> {
        let content = document.content_str();

        let chunks = match &self.strategy {
            ChunkingStrategy::FixedSize { size, overlap } => {
                self.chunk_fixed_size(content, *size, *overlap)
            }
            ChunkingStrategy::Sentence {
                max_sentences,
                overlap_sentences,
            } => self.chunk_by_sentences(content, *max_sentences, *overlap_sentences),
            ChunkingStrategy::Paragraph { max_paragraphs } => {
                self.chunk_by_paragraphs(content, *max_paragraphs)
            }
            ChunkingStrategy::Semantic { .. } => {
                // Placeholder for semantic chunking
                return Err(RragError::document_processing(
                    "Semantic chunking not yet implemented",
                ));
            }
        };

        // Convert to DocumentChunk structs
        let mut document_chunks = Vec::new();
        let mut current_position = 0;

        for (i, chunk_content) in chunks.iter().enumerate() {
            let start_pos = current_position;
            let end_pos = start_pos + chunk_content.len();

            let mut chunk = DocumentChunk::new(&document.id, chunk_content, i, start_pos, end_pos);

            // Inherit document metadata
            chunk.metadata = document.metadata.clone();

            // Add chunk-specific metadata
            chunk = chunk
                .with_metadata(
                    "chunk_total",
                    serde_json::Value::Number(chunks.len().into()),
                )
                .with_metadata(
                    "chunk_strategy",
                    serde_json::Value::String(
                        match &self.strategy {
                            ChunkingStrategy::FixedSize { .. } => "fixed_size",
                            ChunkingStrategy::Sentence { .. } => "sentence",
                            ChunkingStrategy::Paragraph { .. } => "paragraph",
                            ChunkingStrategy::Semantic { .. } => "semantic",
                        }
                        .to_string(),
                    ),
                );

            document_chunks.push(chunk);
            current_position = end_pos;
        }

        Ok(document_chunks)
    }

    /// Fixed size chunking implementation
    fn chunk_fixed_size(&self, content: &str, size: usize, overlap: usize) -> Vec<String> {
        if content.len() <= size {
            return vec![content.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < content.len() {
            let end = std::cmp::min(start + size, content.len());
            let chunk = &content[start..end];
            chunks.push(chunk.to_string());

            if end >= content.len() {
                break;
            }

            start = if overlap >= end { 0 } else { end - overlap };
        }

        chunks
    }

    /// Sentence-based chunking implementation
    fn chunk_by_sentences(
        &self,
        content: &str,
        max_sentences: usize,
        overlap_sentences: usize,
    ) -> Vec<String> {
        // Simple sentence splitting - in production, use a proper NLP library
        let sentences: Vec<&str> = content
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.len() <= max_sentences {
            return vec![content.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < sentences.len() {
            let end = std::cmp::min(start + max_sentences, sentences.len());
            let chunk_sentences = &sentences[start..end];
            let chunk = chunk_sentences.join(". ") + ".";
            chunks.push(chunk);

            if end >= sentences.len() {
                break;
            }

            start = if overlap_sentences >= end {
                0
            } else {
                end - overlap_sentences
            };
        }

        chunks
    }

    /// Paragraph-based chunking implementation
    fn chunk_by_paragraphs(&self, content: &str, max_paragraphs: usize) -> Vec<String> {
        let paragraphs: Vec<&str> = content
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();

        if paragraphs.len() <= max_paragraphs {
            return vec![content.to_string()];
        }

        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();

        for paragraph in paragraphs {
            current_chunk.push(paragraph);

            if current_chunk.len() >= max_paragraphs {
                chunks.push(current_chunk.join("\n\n"));
                current_chunk.clear();
            }
        }

        // Add remaining paragraphs
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.join("\n\n"));
        }

        chunks
    }
}

impl Default for DocumentChunker {
    fn default() -> Self {
        Self::new()
    }
}

/// Custom serde module for Cow<str> handling
mod cow_str_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::borrow::Cow;

    pub fn serialize<S>(cow: &Cow<'static, str>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        cow.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Cow<'static, str>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Cow::Owned(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Test content")
            .with_metadata("source", serde_json::Value::String("test".to_string()));

        assert_eq!(doc.content_str(), "Test content");
        assert!(!doc.id.is_empty());
        assert_eq!(
            doc.metadata.get("source").unwrap().as_str().unwrap(),
            "test"
        );
    }

    #[test]
    fn test_document_chunk() {
        let chunk = DocumentChunk::new("doc1", "chunk content", 0, 0, 13)
            .with_overlap(0, 5)
            .with_metadata("test", serde_json::Value::String("value".to_string()));

        assert_eq!(chunk.document_id, "doc1");
        assert_eq!(chunk.content, "chunk content");
        assert_eq!(chunk.length(), 13);
        assert_eq!(chunk.overlap_next, 5);
    }

    #[test]
    fn test_fixed_size_chunking() {
        let chunker = DocumentChunker::with_strategy(ChunkingStrategy::FixedSize {
            size: 10,
            overlap: 3,
        });

        let doc = Document::new("This is a test document for chunking");
        let chunks = chunker.chunk_document(&doc).unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks[0].content.len() <= 10);
    }

    #[test]
    fn test_sentence_chunking() {
        let chunker = DocumentChunker::with_strategy(ChunkingStrategy::Sentence {
            max_sentences: 2,
            overlap_sentences: 1,
        });

        let doc =
            Document::new("First sentence. Second sentence. Third sentence. Fourth sentence.");
        let chunks = chunker.chunk_document(&doc).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_document_hash() {
        let doc1 = Document::new("Same content").with_content_hash();
        let doc2 = Document::new("Same content").with_content_hash();
        let doc3 = Document::new("Different content").with_content_hash();

        assert_eq!(doc1.content_hash, doc2.content_hash);
        assert_ne!(doc1.content_hash, doc3.content_hash);
    }

    #[test]
    fn test_empty_document() {
        let doc = Document::new("   ");
        assert!(doc.is_empty());

        let doc2 = Document::new("content");
        assert!(!doc2.is_empty());
    }
}
