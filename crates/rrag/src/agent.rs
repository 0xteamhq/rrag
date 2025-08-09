//! # RRAG Agent System
//! 
//! Type-safe, async-first agent implementation with rsllm integration.
//! Focuses on Rust's strengths: ownership, concurrency, and zero-cost abstractions.

use crate::{
    RragError, RragResult, 
    Tool, ToolRegistry, ToolResult,
    Memory,
    StreamingResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Agent configuration using the builder pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent identifier
    pub id: String,
    
    /// Agent name for display
    pub name: String,
    
    /// Model configuration for rsllm
    pub model_config: ModelConfig,
    
    /// Maximum number of tool calls per conversation turn
    pub max_tool_calls: usize,
    
    /// Maximum thinking time before timeout
    pub max_thinking_time: Duration,
    
    /// Whether to enable verbose logging
    pub verbose: bool,
    
    /// System prompt for the agent
    pub system_prompt: Option<String>,
    
    /// Temperature for generation (0.0 to 2.0)
    pub temperature: f32,
    
    /// Whether to stream responses
    pub stream_responses: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "RRAG Agent".to_string(),
            model_config: ModelConfig::default(),
            max_tool_calls: 10,
            max_thinking_time: Duration::from_secs(30),
            verbose: false,
            system_prompt: None,
            temperature: 0.7,
            stream_responses: true,
        }
    }
}

/// Model configuration for rsllm client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model provider (openai, anthropic, etc.)
    pub provider: String,
    
    /// Model name/identifier
    pub model: String,
    
    /// API endpoint URL
    pub api_url: Option<String>,
    
    /// API key (should be in environment variable)
    pub api_key_env: String,
    
    /// Request timeout
    pub timeout: Duration,
    
    /// Maximum tokens in response
    pub max_tokens: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            api_url: None,
            api_key_env: "OPENAI_API_KEY".to_string(),
            timeout: Duration::from_secs(30),
            max_tokens: Some(2048),
        }
    }
}

/// Agent execution context for maintaining state during conversations
pub struct AgentContext {
    /// Current conversation ID
    pub conversation_id: String,
    
    /// Tool call history for this turn
    pub tool_calls: Vec<ToolCall>,
    
    /// Memory for persistent state
    pub memory: Option<Arc<dyn Memory>>,
    
    /// Execution start time
    pub start_time: Instant,
    
    /// Custom context variables
    pub variables: HashMap<String, serde_json::Value>,
}

impl AgentContext {
    pub fn new(conversation_id: impl Into<String>) -> Self {
        Self {
            conversation_id: conversation_id.into(),
            tool_calls: Vec::new(),
            memory: None,
            start_time: Instant::now(),
            variables: HashMap::new(),
        }
    }

    pub fn with_memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn add_tool_call(&mut self, tool_call: ToolCall) {
        self.tool_calls.push(tool_call);
    }

    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.variables.get(key)
    }

    pub fn set_variable(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.variables.insert(key.into(), value);
    }
}

/// Tool call record for tracking agent actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub tool_name: String,
    pub input: String,
    pub result: Option<ToolResult>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration_ms: Option<u64>,
}

impl ToolCall {
    pub fn new(tool_name: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            tool_name: tool_name.into(),
            input: input.into(),
            result: None,
            timestamp: chrono::Utc::now(),
            duration_ms: None,
        }
    }

    pub fn with_result(mut self, result: ToolResult, duration_ms: u64) -> Self {
        self.result = Some(result);
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Agent response containing output and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Generated response text
    pub text: String,
    
    /// Tool calls made during generation
    pub tool_calls: Vec<ToolCall>,
    
    /// Response metadata
    pub metadata: ResponseMetadata,
    
    /// Whether this is a final response or needs continuation
    pub is_final: bool,
}

/// Response metadata for tracking and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Total processing time
    pub duration_ms: u64,
    
    /// Number of model calls made
    pub model_calls: usize,
    
    /// Total tokens used (if available)
    pub total_tokens: Option<usize>,
    
    /// Conversation turn ID
    pub turn_id: String,
    
    /// Agent configuration snapshot
    pub agent_config: AgentConfig,
}

/// Main RRAG agent implementation
pub struct RragAgent {
    /// Agent configuration
    config: AgentConfig,
    
    /// Tool registry for available tools
    tools: Arc<ToolRegistry>,
    
    /// Memory for conversation state
    memory: Option<Arc<dyn Memory>>,
    
    /// rsllm client (when feature is enabled)
    #[cfg(feature = "rsllm-client")]
    llm_client: Option<Arc<rsllm::Client>>,
    
    /// Active contexts indexed by conversation ID
    contexts: Arc<RwLock<HashMap<String, AgentContext>>>,
}

impl RragAgent {
    /// Create a new agent with default configuration
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
            tools: Arc::new(ToolRegistry::new()),
            memory: None,
            #[cfg(feature = "rsllm-client")]
            llm_client: None,
            contexts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get agent builder for configuration
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Process a message and generate a response
    pub async fn process_message(
        &self,
        message: impl Into<String>,
        conversation_id: Option<String>,
    ) -> RragResult<AgentResponse> {
        let message = message.into();
        let conversation_id = conversation_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let turn_id = Uuid::new_v4().to_string();
        let start_time = Instant::now();

        // Get or create conversation context
        let mut contexts = self.contexts.write().await;
        let context = contexts
            .entry(conversation_id.clone())
            .or_insert_with(|| {
                let mut ctx = AgentContext::new(&conversation_id);
                if let Some(memory) = &self.memory {
                    ctx = ctx.with_memory(memory.clone());
                }
                ctx
            });

        // Load conversation history from memory if available
        let conversation_history = if let Some(memory) = &context.memory {
            memory.get_conversation_history(&conversation_id).await
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Process message with tool calling loop
        let response = self.process_with_tools(message, context, &conversation_history).await?;

        // Save to memory if available
        if let Some(memory) = &context.memory {
            memory.add_message(&conversation_id, "user", &response).await
                .map_err(|e| RragError::memory("add_message", e.to_string()))?;
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(AgentResponse {
            text: response,
            tool_calls: context.tool_calls.clone(),
            metadata: ResponseMetadata {
                duration_ms,
                model_calls: 1, // Simplified for now
                total_tokens: None,
                turn_id,
                agent_config: self.config.clone(),
            },
            is_final: true,
        })
    }

    /// Stream a response for real-time interaction
    pub async fn stream_message(
        &self,
        message: impl Into<String>,
        conversation_id: Option<String>,
    ) -> RragResult<StreamingResponse> {
        // For now, return a simple streaming implementation
        // In production, this would integrate with rsllm's streaming capabilities
        let response = self.process_message(message, conversation_id).await?;
        
        Ok(StreamingResponse::from_text(response.text))
    }

    /// Internal method for processing with tool calling loop
    async fn process_with_tools(
        &self,
        message: String,
        context: &mut AgentContext,
        _conversation_history: &[String], // Currently unused but available
    ) -> RragResult<String> {
        // Check for timeout
        if context.elapsed() > self.config.max_thinking_time {
            return Err(RragError::timeout("agent_processing", 
                self.config.max_thinking_time.as_millis() as u64));
        }

        // For now, implement a simple processing logic
        // In production, this would integrate with rsllm for actual LLM calls
        let response = self.mock_llm_processing(&message, context).await?;
        
        Ok(response)
    }

    /// Mock LLM processing for demonstration
    /// In production, this would be replaced with actual rsllm client calls
    async fn mock_llm_processing(
        &self,
        message: &str,
        context: &mut AgentContext,
    ) -> RragResult<String> {
        // Simple pattern matching for demonstration
        if message.to_lowercase().contains("calculate") || message.contains("+") || message.contains("-") {
            // Try to use calculator tool
            if let Some(calc_tool) = self.tools.get("calculator") {
                let input = extract_calculation(message);
                let start = Instant::now();
                
                match calc_tool.execute(&input).await {
                    Ok(result) => {
                        let duration = start.elapsed().as_millis() as u64;
                        let tool_call = ToolCall::new("calculator", input)
                            .with_result(result.clone(), duration);
                        context.add_tool_call(tool_call);
                        
                        if result.success {
                            return Ok(format!("I calculated that for you: {}", result.output));
                        } else {
                            return Ok(format!("I tried to calculate that but encountered an error: {}", result.output));
                        }
                    }
                    Err(e) => {
                        return Err(RragError::tool_execution("calculator", e.to_string()));
                    }
                }
            }
        }

        // Default response
        Ok(format!("I understand you said: '{}'. How can I help you further?", message))
    }

    /// Get agent configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get available tools
    pub fn tools(&self) -> Arc<ToolRegistry> {
        self.tools.clone()
    }
}

impl Default for RragAgent {
    fn default() -> Self {
        Self::new()
    }
}

/// Agent builder for fluent configuration
pub struct AgentBuilder {
    config: AgentConfig,
    tools: Vec<Arc<dyn Tool>>,
    memory: Option<Arc<dyn Memory>>,
    #[cfg(feature = "rsllm-client")]
    llm_client: Option<Arc<rsllm::Client>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
            tools: Vec::new(),
            memory: None,
            #[cfg(feature = "rsllm-client")]
            llm_client: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    pub fn with_model(mut self, provider: impl Into<String>, model: impl Into<String>) -> Self {
        self.config.model_config.provider = provider.into();
        self.config.model_config.model = model.into();
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature.clamp(0.0, 2.0);
        self
    }

    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn with_memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    pub fn with_max_tool_calls(mut self, max: usize) -> Self {
        self.config.max_tool_calls = max;
        self
    }

    #[cfg(feature = "rsllm-client")]
    pub fn with_rsllm_client(self, client: Arc<rsllm::Client>) -> Self {
        self.with_llm_client(client)
    }

    #[cfg(feature = "rsllm-client")]
    pub fn with_llm_client(mut self, client: Arc<rsllm::Client>) -> Self {
        self.llm_client = Some(client);
        self
    }

    pub fn build(self) -> RragResult<RragAgent> {
        // Validate configuration
        if self.config.temperature < 0.0 || self.config.temperature > 2.0 {
            return Err(RragError::validation(
                "temperature", 
                "0.0 to 2.0", 
                self.config.temperature.to_string()
            ));
        }

        let tools = if self.tools.is_empty() {
            Arc::new(ToolRegistry::new())
        } else {
            Arc::new(ToolRegistry::with_tools(self.tools))
        };

        Ok(RragAgent {
            config: self.config,
            tools,
            memory: self.memory,
            #[cfg(feature = "rsllm-client")]
            llm_client: self.llm_client,
            contexts: Arc::new(RwLock::new(HashMap::new())),
        })
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract calculation from message (simple implementation)
fn extract_calculation(message: &str) -> String {
    // Look for mathematical expressions
    let patterns = ["+", "-", "*", "/", "="];
    
    for line in message.lines() {
        for pattern in &patterns {
            if line.contains(pattern) {
                // Extract the mathematical part
                let parts: Vec<&str> = line.split_whitespace().collect();
                for window in parts.windows(3) {
                    if window.len() == 3 {
                        if let (Ok(_), Ok(_)) = (window[0].parse::<f64>(), window[2].parse::<f64>()) {
                            if patterns.contains(&window[1]) {
                                return format!("{}{}{}", window[0], window[1], window[2]);
                            }
                        }
                    }
                }
                
                // Fallback: return the line that contains math
                return line.trim().to_string();
            }
        }
    }
    
    // Fallback: extract any numbers and operators
    let math_chars: String = message.chars()
        .filter(|c| c.is_ascii_digit() || "+-*/=. ".contains(*c))
        .collect::<String>()
        .trim()
        .to_string();
        
    if !math_chars.is_empty() {
        math_chars
    } else {
        message.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Calculator;

    #[tokio::test]
    async fn test_agent_creation() {
        let agent = RragAgent::builder()
            .with_name("Test Agent")
            .with_temperature(0.5)
            .build()
            .unwrap();
        
        assert_eq!(agent.config().name, "Test Agent");
        assert_eq!(agent.config().temperature, 0.5);
    }

    #[tokio::test]
    async fn test_agent_with_tools() {
        let agent = RragAgent::builder()
            .with_tool(Arc::new(Calculator))
            .build()
            .unwrap();
        
        let response = agent.process_message("Calculate 2+2", None).await.unwrap();
        assert!(response.text.contains("calculated"));
        assert!(!response.tool_calls.is_empty());
    }

    #[test]
    fn test_extract_calculation() {
        assert_eq!(extract_calculation("What is 2 + 2?"), "2+2");
        assert_eq!(extract_calculation("Calculate 10 * 5"), "10*5");
        assert_eq!(extract_calculation("Can you compute 15 / 3"), "15/3");
    }

    #[test]
    fn test_agent_config_validation() {
        let result = RragAgent::builder()
            .with_temperature(3.0)  // Invalid temperature
            .build();
        
        assert!(result.is_err());
        if let Err(RragError::Validation { field, .. }) = result {
            assert_eq!(field, "temperature");
        }
    }

    #[tokio::test]
    async fn test_agent_context() {
        let mut context = AgentContext::new("test-conversation");
        
        let tool_call = ToolCall::new("test_tool", "test_input");
        context.add_tool_call(tool_call);
        
        assert_eq!(context.tool_calls.len(), 1);
        assert_eq!(context.conversation_id, "test-conversation");
    }
}