//! # RRAG Integration
//!
//! Integration layer between RGraph and RRAG for RAG-powered agent workflows.
//! This module provides nodes that can leverage RRAG's retrieval and generation capabilities.

use crate::core::{ExecutionContext, ExecutionResult, Node, NodeId};
use crate::state::{GraphState, StateValue};
use crate::{RGraphError, RGraphResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for RAG retrieval nodes
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RagRetrievalConfig {
    pub query_key: String,
    pub context_key: String,
    pub top_k: usize,
    pub similarity_threshold: Option<f32>,
    pub metadata_filters: Vec<(String, String)>,
}

impl Default for RagRetrievalConfig {
    fn default() -> Self {
        Self {
            query_key: "user_query".to_string(),
            context_key: "retrieval_context".to_string(),
            top_k: 5,
            similarity_threshold: Some(0.7),
            metadata_filters: Vec::new(),
        }
    }
}

/// Configuration for RAG generation nodes
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RagGenerationConfig {
    pub query_key: String,
    pub context_key: String,
    pub response_key: String,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

impl Default for RagGenerationConfig {
    fn default() -> Self {
        Self {
            query_key: "user_query".to_string(),
            context_key: "retrieval_context".to_string(),
            response_key: "rag_response".to_string(),
            system_prompt: Some(
                "You are a helpful AI assistant. Use the provided context to answer the user's question accurately and comprehensively.".to_string()
            ),
            max_tokens: Some(512),
            temperature: Some(0.7),
        }
    }
}

/// Configuration for context evaluation nodes
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ContextEvaluationConfig {
    pub context_key: String,
    pub query_key: String,
    pub relevance_key: String,
    pub min_relevance_score: f32,
}

impl Default for ContextEvaluationConfig {
    fn default() -> Self {
        Self {
            context_key: "retrieval_context".to_string(),
            query_key: "user_query".to_string(),
            relevance_key: "context_relevance".to_string(),
            min_relevance_score: 0.6,
        }
    }
}

/// A node that performs RAG retrieval (simplified mock implementation)
pub struct RagRetrievalNode {
    id: NodeId,
    name: String,
    config: RagRetrievalConfig,
}

impl RagRetrievalNode {
    pub fn new(id: impl Into<NodeId>, name: impl Into<String>, config: RagRetrievalConfig) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            config,
        }
    }
}

#[async_trait]
impl Node for RagRetrievalNode {
    async fn execute(
        &self,
        state: &mut GraphState,
        context: &ExecutionContext,
    ) -> RGraphResult<ExecutionResult> {
        // Get the query from state
        let query = state.get(&self.config.query_key)?;
        let query_text = query
            .as_string()
            .ok_or_else(|| RGraphError::node(self.id.as_str(), "Query must be a string"))?;

        // Simulate document retrieval (in real implementation, this would use RRAG)
        let mock_documents = vec![
            create_mock_document("Machine learning is a method of data analysis that automates analytical model building.", 0.85),
            create_mock_document("It is a branch of artificial intelligence based on the idea that systems can learn from data.", 0.78),
            create_mock_document("ML algorithms build a model based on training data to make predictions or decisions.", 0.72),
        ];

        // Filter based on similarity threshold if provided
        let filtered_docs: Vec<StateValue> =
            if let Some(threshold) = self.config.similarity_threshold {
                mock_documents
                    .into_iter()
                    .filter(|doc| {
                        if let Some(obj) = doc.as_object() {
                            if let Some(score_val) = obj.get("score") {
                                if let Some(score) = score_val.as_float() {
                                    return score >= threshold as f64;
                                }
                            }
                        }
                        false
                    })
                    .take(self.config.top_k)
                    .collect()
            } else {
                mock_documents.into_iter().take(self.config.top_k).collect()
            };

        // Store retrieval context
        state.set_with_context(
            context.current_node.as_str(),
            &self.config.context_key,
            StateValue::Array(filtered_docs.clone()),
        );

        // Store retrieval metadata
        state.set_with_context(
            context.current_node.as_str(),
            "retrieval_metadata",
            StateValue::from(serde_json::json!({
                "total_results": filtered_docs.len(),
                "retrieved_count": filtered_docs.len(),
                "query": query_text
            })),
        );

        Ok(ExecutionResult::Continue)
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn input_keys(&self) -> Vec<&str> {
        vec![&self.config.query_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.config.context_key, "retrieval_metadata"]
    }
}

/// A node that performs RAG generation (simplified mock implementation)
pub struct RagGenerationNode {
    id: NodeId,
    name: String,
    config: RagGenerationConfig,
}

impl RagGenerationNode {
    pub fn new(
        id: impl Into<NodeId>,
        name: impl Into<String>,
        config: RagGenerationConfig,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            config,
        }
    }
}

#[async_trait]
impl Node for RagGenerationNode {
    async fn execute(
        &self,
        state: &mut GraphState,
        context: &ExecutionContext,
    ) -> RGraphResult<ExecutionResult> {
        // Get query and context from state
        let query = state.get(&self.config.query_key)?;
        let query_text = query
            .as_string()
            .ok_or_else(|| RGraphError::node(self.id.as_str(), "Query must be a string"))?;

        let context_value = state.get(&self.config.context_key)?;
        let context_docs = if let Some(array) = context_value.as_array() {
            array
                .iter()
                .filter_map(|v| {
                    if let Some(obj) = v.as_object() {
                        if let Some(content) = obj.get("content") {
                            content.as_string()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<&str>>()
                .join("\n\n")
        } else {
            return Err(RGraphError::node(
                self.id.as_str(),
                "Context must be an array of documents",
            ));
        };

        // Simulate response generation (in real implementation, this would use RRAG's generation engine)
        let response = format!(
            "Based on the provided context, here's what I can tell you about {}: {}",
            query_text,
            if context_docs.is_empty() {
                "I don't have specific information available, but I can provide a general response."
            } else {
                "The context provides relevant information that I can use to answer your question comprehensively."
            }
        );

        // Calculate token estimate before moving response
        let token_estimate = response.len() / 4;

        // Store the generated response
        state.set_with_context(
            context.current_node.as_str(),
            &self.config.response_key,
            StateValue::String(response),
        );

        // Store generation metadata
        state.set_with_context(
            context.current_node.as_str(),
            "generation_metadata",
            StateValue::from(serde_json::json!({
                "tokens_used": token_estimate,
                "model": "mock-model",
                "finish_reason": "complete"
            })),
        );

        Ok(ExecutionResult::Continue)
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn input_keys(&self) -> Vec<&str> {
        vec![&self.config.query_key, &self.config.context_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.config.response_key, "generation_metadata"]
    }
}

/// A node that evaluates context relevance
pub struct ContextEvaluationNode {
    id: NodeId,
    name: String,
    config: ContextEvaluationConfig,
}

impl ContextEvaluationNode {
    pub fn new(
        id: impl Into<NodeId>,
        name: impl Into<String>,
        config: ContextEvaluationConfig,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            config,
        }
    }
}

#[async_trait]
impl Node for ContextEvaluationNode {
    async fn execute(
        &self,
        state: &mut GraphState,
        context: &ExecutionContext,
    ) -> RGraphResult<ExecutionResult> {
        // Get context and query from state
        let context_value = state.get(&self.config.context_key)?;
        let query_value = state.get(&self.config.query_key)?;

        let query_text = query_value
            .as_string()
            .ok_or_else(|| RGraphError::node(self.id.as_str(), "Query must be a string"))?;

        let context_docs = if let Some(array) = context_value.as_array() {
            array
        } else {
            return Err(RGraphError::node(
                self.id.as_str(),
                "Context must be an array of documents",
            ));
        };

        // Evaluate relevance for each document
        let mut relevant_docs = Vec::new();
        let mut total_score = 0.0;
        let mut evaluated_count = 0;

        for doc in context_docs {
            if let Some(obj) = doc.as_object() {
                if let Some(content_val) = obj.get("content") {
                    if let Some(content) = content_val.as_string() {
                        // Simple relevance scoring based on keyword overlap
                        let relevance_score = self.calculate_relevance_score(query_text, content);

                        if relevance_score >= self.config.min_relevance_score {
                            let mut relevant_doc_map = obj.clone();
                            relevant_doc_map.insert(
                                "relevance_score".to_string(),
                                StateValue::Float(relevance_score as f64),
                            );
                            let relevant_doc = StateValue::Object(relevant_doc_map);
                            relevant_docs.push(relevant_doc);
                        }

                        total_score += relevance_score;
                        evaluated_count += 1;
                    }
                }
            }
        }

        let average_relevance = if evaluated_count > 0 {
            total_score / evaluated_count as f32
        } else {
            0.0
        };

        // Store filtered relevant context
        state.set_with_context(
            context.current_node.as_str(),
            "filtered_context",
            StateValue::Array(relevant_docs.clone()),
        );

        // Store relevance metrics
        state.set_with_context(
            context.current_node.as_str(),
            &self.config.relevance_key,
            StateValue::from(serde_json::json!({
                "average_score": average_relevance,
                "relevant_docs_count": relevant_docs.len(),
                "total_docs_evaluated": evaluated_count,
                "min_threshold": self.config.min_relevance_score
            })),
        );

        Ok(ExecutionResult::Continue)
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn input_keys(&self) -> Vec<&str> {
        vec![&self.config.context_key, &self.config.query_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec!["filtered_context", &self.config.relevance_key]
    }
}

impl ContextEvaluationNode {
    /// Simple relevance scoring based on keyword overlap
    fn calculate_relevance_score(&self, query: &str, content: &str) -> f32 {
        let query_words: std::collections::HashSet<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect();

        let content_words: std::collections::HashSet<String> = content
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect();

        if query_words.is_empty() || content_words.is_empty() {
            return 0.0;
        }

        let intersection_count = query_words.intersection(&content_words).count();
        let union_count = query_words.union(&content_words).count();

        if union_count == 0 {
            0.0
        } else {
            intersection_count as f32 / union_count as f32
        }
    }
}

/// Builder for RAG-powered workflows
pub struct RagWorkflowBuilder;

impl RagWorkflowBuilder {
    pub fn new() -> Self {
        Self
    }

    /// Create a RAG retrieval node
    pub fn build_retrieval_node(
        &self,
        id: impl Into<NodeId>,
        name: impl Into<String>,
        config: RagRetrievalConfig,
    ) -> RGraphResult<Arc<RagRetrievalNode>> {
        Ok(Arc::new(RagRetrievalNode::new(id, name, config)))
    }

    /// Create a RAG generation node
    pub fn build_generation_node(
        &self,
        id: impl Into<NodeId>,
        name: impl Into<String>,
        config: RagGenerationConfig,
    ) -> RGraphResult<Arc<RagGenerationNode>> {
        Ok(Arc::new(RagGenerationNode::new(id, name, config)))
    }

    /// Create a context evaluation node
    pub fn build_evaluation_node(
        &self,
        id: impl Into<NodeId>,
        name: impl Into<String>,
        config: ContextEvaluationConfig,
    ) -> RGraphResult<Arc<ContextEvaluationNode>> {
        Ok(Arc::new(ContextEvaluationNode::new(id, name, config)))
    }
}

impl Default for RagWorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create mock documents for demonstration
fn create_mock_document(content: &str, score: f64) -> StateValue {
    let mut doc = HashMap::new();
    doc.insert(
        "content".to_string(),
        StateValue::String(content.to_string()),
    );
    doc.insert("score".to_string(), StateValue::Float(score));
    doc.insert("metadata".to_string(), StateValue::Object(HashMap::new()));
    StateValue::Object(doc)
}
