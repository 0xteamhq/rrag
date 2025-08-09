//! # BM25 Keyword-based Retrieval
//! 
//! Implementation of the BM25 algorithm for keyword-based document retrieval.
//! BM25 is a probabilistic retrieval model that ranks documents based on term frequency
//! and inverse document frequency.

use crate::{RragResult, Document, SearchResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// BM25 retriever configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    /// k1 parameter: controls term frequency saturation (typically 1.2-2.0)
    pub k1: f32,
    
    /// b parameter: controls length normalization (typically 0.75)
    pub b: f32,
    
    /// Tokenizer type to use
    pub tokenizer: TokenizerType,
    
    /// Minimum token length to index
    pub min_token_length: usize,
    
    /// Maximum token length to index
    pub max_token_length: usize,
    
    /// Whether to use stemming
    pub use_stemming: bool,
    
    /// Whether to remove stop words
    pub remove_stopwords: bool,
    
    /// Custom stop words list
    pub custom_stopwords: Option<HashSet<String>>,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            tokenizer: TokenizerType::Standard,
            min_token_length: 2,
            max_token_length: 50,
            use_stemming: true,
            remove_stopwords: true,
            custom_stopwords: None,
        }
    }
}

/// Tokenizer types for text processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenizerType {
    /// Standard whitespace and punctuation tokenizer
    Standard,
    /// N-gram based tokenizer
    NGram(usize),
    /// Language-specific tokenizer
    Language(String),
}

/// BM25 index entry for a document
#[derive(Debug, Clone)]
struct BM25Document {
    /// Document ID
    id: String,
    
    /// Original content
    content: String,
    
    /// Tokenized terms with frequencies
    term_frequencies: HashMap<String, f32>,
    
    /// Document length (number of tokens)
    length: usize,
    
    /// Additional metadata
    metadata: HashMap<String, serde_json::Value>,
}

/// BM25 retriever implementation
pub struct BM25Retriever {
    /// Configuration
    config: BM25Config,
    
    /// Document storage
    documents: Arc<RwLock<HashMap<String, BM25Document>>>,
    
    /// Inverted index: term -> document IDs
    inverted_index: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    
    /// Document frequencies for each term
    document_frequencies: Arc<RwLock<HashMap<String, usize>>>,
    
    /// Average document length
    avg_doc_length: Arc<RwLock<f32>>,
    
    /// Total number of documents
    total_docs: Arc<RwLock<usize>>,
    
    /// Stop words set
    stop_words: HashSet<String>,
}

impl BM25Retriever {
    /// Create a new BM25 retriever with configuration
    pub fn new(config: BM25Config) -> Self {
        let stop_words = if config.remove_stopwords {
            Self::default_stop_words()
        } else {
            HashSet::new()
        };
        
        Self {
            config,
            documents: Arc::new(RwLock::new(HashMap::new())),
            inverted_index: Arc::new(RwLock::new(HashMap::new())),
            document_frequencies: Arc::new(RwLock::new(HashMap::new())),
            avg_doc_length: Arc::new(RwLock::new(0.0)),
            total_docs: Arc::new(RwLock::new(0)),
            stop_words,
        }
    }
    
    /// Index a document
    pub async fn index_document(&self, doc: &Document) -> RragResult<()> {
        let tokens = self.tokenize(&doc.content);
        let term_frequencies = self.calculate_term_frequencies(&tokens);
        
        let bm25_doc = BM25Document {
            id: doc.id.clone(),
            content: doc.content.to_string(),
            term_frequencies: term_frequencies.clone(),
            length: tokens.len(),
            metadata: doc.metadata.clone(),
        };
        
        // Update document storage
        let mut documents = self.documents.write().await;
        documents.insert(doc.id.clone(), bm25_doc);
        
        // Update inverted index
        let mut inverted_index = self.inverted_index.write().await;
        let mut doc_frequencies = self.document_frequencies.write().await;
        
        for term in term_frequencies.keys() {
            inverted_index
                .entry(term.clone())
                .or_insert_with(HashSet::new)
                .insert(doc.id.clone());
            
            *doc_frequencies.entry(term.clone()).or_insert(0) += 1;
        }
        
        // Update statistics
        let mut total_docs = self.total_docs.write().await;
        *total_docs += 1;
        
        let mut avg_length = self.avg_doc_length.write().await;
        *avg_length = (*avg_length * (*total_docs - 1) as f32 + tokens.len() as f32) / *total_docs as f32;
        
        Ok(())
    }
    
    /// Index multiple documents in batch
    pub async fn index_batch(&self, documents: Vec<Document>) -> RragResult<()> {
        for doc in documents {
            self.index_document(&doc).await?;
        }
        Ok(())
    }
    
    /// Search using BM25 algorithm
    pub async fn search(&self, query: &str, limit: usize) -> RragResult<Vec<SearchResult>> {
        let query_tokens = self.tokenize(query);
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }
        
        let documents = self.documents.read().await;
        let inverted_index = self.inverted_index.read().await;
        let doc_frequencies = self.document_frequencies.read().await;
        let avg_length = *self.avg_doc_length.read().await;
        let total_docs = *self.total_docs.read().await;
        
        let mut scores: HashMap<String, f32> = HashMap::new();
        
        // Calculate BM25 scores for each document
        for term in &query_tokens {
            if let Some(doc_ids) = inverted_index.get(term) {
                let df = doc_frequencies.get(term).copied().unwrap_or(0) as f32;
                let idf = ((total_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
                
                for doc_id in doc_ids {
                    if let Some(doc) = documents.get(doc_id) {
                        let tf = doc.term_frequencies.get(term).copied().unwrap_or(0.0);
                        let doc_length = doc.length as f32;
                        
                        // BM25 formula
                        let numerator = tf * (self.config.k1 + 1.0);
                        let denominator = tf + self.config.k1 * (1.0 - self.config.b + self.config.b * (doc_length / avg_length));
                        let score = idf * (numerator / denominator);
                        
                        *scores.entry(doc_id.clone()).or_insert(0.0) += score;
                    }
                }
            }
        }
        
        // Sort by score and return top results
        let mut results: Vec<_> = scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                documents.get(&doc_id).map(|doc| SearchResult {
                    id: doc_id.clone(),
                    content: doc.content.clone(),
                    score: score / query_tokens.len() as f32, // Normalize by query length
                    rank: 0,
                    metadata: doc.metadata.clone(),
                    embedding: None,
                })
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);
        
        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
        
        Ok(results)
    }
    
    /// Tokenize text into terms
    fn tokenize(&self, text: &str) -> Vec<String> {
        let lowercase = text.to_lowercase();
        let tokens: Vec<String> = match &self.config.tokenizer {
            TokenizerType::Standard => {
                lowercase
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|s| !s.is_empty())
                    .filter(|s| s.len() >= self.config.min_token_length)
                    .filter(|s| s.len() <= self.config.max_token_length)
                    .filter(|s| !self.stop_words.contains(*s))
                    .map(|s| {
                        if self.config.use_stemming {
                            Self::simple_stem(s)
                        } else {
                            s.to_string()
                        }
                    })
                    .collect()
            }
            TokenizerType::NGram(n) => {
                // N-gram tokenization
                let chars: Vec<char> = lowercase.chars().collect();
                let mut ngrams = Vec::new();
                for i in 0..chars.len().saturating_sub(n - 1) {
                    let ngram: String = chars[i..i + n].iter().collect();
                    if !ngram.trim().is_empty() {
                        ngrams.push(ngram);
                    }
                }
                ngrams
            }
            TokenizerType::Language(ref _lang) => {
                // For now, use standard tokenization
                // In production, integrate language-specific tokenizers
                lowercase
                    .split_whitespace()
                    .filter(|s| !self.stop_words.contains(*s))
                    .map(String::from)
                    .collect()
            }
        };
        
        tokens
    }
    
    /// Calculate term frequencies for a list of tokens
    fn calculate_term_frequencies(&self, tokens: &[String]) -> HashMap<String, f32> {
        let mut frequencies = HashMap::new();
        let total = tokens.len() as f32;
        
        for token in tokens {
            *frequencies.entry(token.clone()).or_insert(0.0) += 1.0;
        }
        
        // Normalize frequencies
        for freq in frequencies.values_mut() {
            *freq /= total;
        }
        
        frequencies
    }
    
    /// Simple stemming algorithm (Porter stemmer simplified)
    fn simple_stem(word: &str) -> String {
        let mut stem = word.to_string();
        
        // Remove common suffixes
        let suffixes = ["ing", "ed", "es", "s", "ly", "er", "est", "ness", "ment"];
        for suffix in &suffixes {
            if stem.len() > suffix.len() + 3 && stem.ends_with(suffix) {
                stem.truncate(stem.len() - suffix.len());
                break;
            }
        }
        
        stem
    }
    
    /// Default English stop words
    fn default_stop_words() -> HashSet<String> {
        let words = vec![
            "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", 
            "from", "has", "have", "he", "in", "is", "it", "its", "of", "on", 
            "that", "the", "to", "was", "will", "with", "the", "this", "these",
            "those", "i", "you", "we", "they", "them", "their", "what", "which",
            "who", "when", "where", "why", "how", "all", "would", "there", "could"
        ];
        
        words.into_iter().map(String::from).collect()
    }
    
    /// Clear the index
    pub async fn clear(&self) -> RragResult<()> {
        let mut documents = self.documents.write().await;
        let mut inverted_index = self.inverted_index.write().await;
        let mut doc_frequencies = self.document_frequencies.write().await;
        let mut avg_length = self.avg_doc_length.write().await;
        let mut total_docs = self.total_docs.write().await;
        
        documents.clear();
        inverted_index.clear();
        doc_frequencies.clear();
        *avg_length = 0.0;
        *total_docs = 0;
        
        Ok(())
    }
    
    /// Get index statistics
    pub async fn stats(&self) -> HashMap<String, serde_json::Value> {
        let documents = self.documents.read().await;
        let inverted_index = self.inverted_index.read().await;
        let total_docs = *self.total_docs.read().await;
        let avg_length = *self.avg_doc_length.read().await;
        
        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), total_docs.into());
        stats.insert("unique_terms".to_string(), inverted_index.len().into());
        stats.insert("average_document_length".to_string(), avg_length.into());
        stats.insert("index_size_bytes".to_string(), 
            (documents.len() * std::mem::size_of::<BM25Document>()).into());
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_bm25_indexing_and_search() {
        let retriever = BM25Retriever::new(BM25Config::default());
        
        let docs = vec![
            Document::new("1", "The quick brown fox jumps over the lazy dog"),
            Document::new("2", "A quick brown dog runs through the forest"),
            Document::new("3", "The lazy cat sleeps in the warm sunshine"),
        ];
        
        retriever.index_batch(docs).await.unwrap();
        
        let results = retriever.search("quick brown", 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].score > results[1].score);
    }
}