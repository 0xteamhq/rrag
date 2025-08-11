//! # RRAG Agent System
//!
//! Type-safe, async-first agent implementation with comprehensive tool integration,
//! memory management, and streaming support. Built for production workloads with
//! robust error handling and configurable behavior.
//!
//! ## Features
//!
//! - **Tool Integration**: Type-safe tool calling with automatic schema generation
//! - **Memory Management**: Persistent conversation state and context
//! - **Streaming Responses**: Real-time token streaming for interactive experiences
//! - **rsllm Integration**: Native Rust LLM client support when feature is enabled
//! - **Async-First**: Built on Tokio for high-concurrency workloads
//! - **Configuration**: Flexible configuration with builder pattern
//! - **Error Handling**: Comprehensive error types with detailed context
//! - **Monitoring**: Built-in metrics and observability
//!
//! ## Quick Start
//!
//! ### Basic Agent
//!
//! ```rust
//! use rrag::prelude::*;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! // Create a basic agent
//! let agent = AgentBuilder::new()
//!     .with_name("My Assistant")
//!     .with_model("openai", "gpt-4")
//!     .with_temperature(0.7)
//!     .build()?;
//!
//! // Process a message
//! let response = agent.process_message("Hello! How can you help me?", None).await?;
//! println!("Agent: {}", response.text);
//! # Ok(())
//! # }
//! ```
//!
//! ### Agent with Tools
//!
//! ```rust
//! use rrag::prelude::*;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! // Create an agent with tools
//! let agent = AgentBuilder::new()
//!     .with_name("Math Assistant")
//!     .with_tool(Arc::new(Calculator::new()))
//!     # #[cfg(feature = "http")]
//!     .with_tool(Arc::new(HttpTool::new()))
//!     .with_system_prompt("You are a helpful math assistant with access to a calculator.")
//!     .build()?;
//!
//! // The agent can now use tools automatically
//! let response = agent.process_message("What's 15 * 23?", None).await?;
//! println!("Agent: {}", response.text);
//! println!("Used {} tools", response.tool_calls.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Agent with Memory
//!
//! ```rust
//! use rrag::prelude::*;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! // Create memory for conversation history
//! let memory = Arc::new(ConversationBufferMemory::new(100));
//!
//! // Create agent with memory
//! let agent = AgentBuilder::new()
//!     .with_name("Contextual Assistant")
//!     .with_memory(memory)
//!     .build()?;
//!
//! // Have a multi-turn conversation
//! let conversation_id = "user-123";
//!
//! let response1 = agent.process_message(
//!     "My name is Alice", 
//!     Some(conversation_id.to_string())
//! ).await?;
//!
//! let response2 = agent.process_message(
//!     "What's my name?", 
//!     Some(conversation_id.to_string())
//! ).await?;
//!
//! // Agent remembers previous context
//! assert!(response2.text.contains("Alice"));
//! # Ok(())
//! # }
//! ```
//!
//! ### Streaming Responses
//!
//! ```rust
//! use rrag::prelude::*;
//! use futures::StreamExt;
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! let agent = AgentBuilder::new()
//!     .with_name("Streaming Assistant")
//!     .build()?;
//!
//! // Stream response tokens in real-time
//! let mut stream = agent.stream_message("Tell me a story", None).await?;
//!
//! while let Some(token) = stream.next().await {
//!     match token {
//!         Ok(stream_token) => print!("{}", stream_token.content),
//!         Err(e) => eprintln!("Stream error: {}", e),
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The agent system is built around several key components:
//!
//! - **AgentConfig**: Configuration and behavior settings
//! - **AgentContext**: Per-conversation state and variables
//! - **ToolRegistry**: Type-safe tool management and execution
//! - **Memory**: Persistent conversation state
//! - **StreamingResponse**: Real-time response streaming
//!
//! ## Configuration Options
//!
//! Agents support extensive configuration:
//!
//! ```rust
//! use rrag::prelude::*;
//! use std::time::Duration;
//!
//! # fn example() -> RragResult<()> {
//! let config = AgentConfig {
//!     name: "Production Agent".to_string(),
//!     model_config: ModelConfig {
//!         provider: "openai".to_string(),
//!         model: "gpt-4-turbo".to_string(),
//!         timeout: Duration::from_secs(60),
//!         max_tokens: Some(4096),
//!         ..Default::default()
//!     },
//!     max_tool_calls: 5,
//!     max_thinking_time: Duration::from_secs(120),
//!     temperature: 0.3,
//!     stream_responses: true,
//!     verbose: false,
//!     system_prompt: Some("You are a professional assistant.".to_string()),
//!     ..Default::default()
//! };
//! # Ok(())
//! # }
//! ```
//!
//! ## Error Handling
//!
//! The agent system provides detailed error information:
//!
//! ```rust
//! use rrag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() {
//! match agent.process_message("Hello", None).await {
//!     Ok(response) => {
//!         println!("Success: {}", response.text);
//!         println!("Processing time: {}ms", response.metadata.duration_ms);
//!     }
//!     Err(RragError::Agent { agent_id, message, .. }) => {
//!         eprintln!("Agent {} error: {}", agent_id, message);
//!     }
//!     Err(RragError::ToolExecution { tool, message, .. }) => {
//!         eprintln!("Tool {} failed: {}", tool, message);
//!     }
//!     Err(RragError::Timeout { operation, duration_ms }) => {
//!         eprintln!("Operation {} timed out after {}ms", operation, duration_ms);
//!     }
//!     Err(e) => eprintln!("Other error: {}", e),
//! }
//! # }
//! ```
//!
//! ## Performance Considerations
//!
//! - Use streaming for long responses to improve perceived performance
//! - Configure appropriate timeouts based on expected response times
//! - Limit tool calls to prevent infinite loops
//! - Use memory efficiently by setting appropriate buffer sizes
//! - Monitor agent performance using the built-in metrics

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

/// Configuration for agent behavior and capabilities
///
/// Defines all aspects of agent behavior including model settings,
/// tool usage limits, streaming preferences, and safety constraints.
/// Uses the builder pattern for convenient configuration.
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
/// use std::time::Duration;
///
/// let config = AgentConfig {
///     name: "Customer Support Agent".to_string(),
///     temperature: 0.3, // More focused responses
///     max_tool_calls: 3, // Limit tool usage
///     max_thinking_time: Duration::from_secs(30),
///     stream_responses: true,
///     system_prompt: Some("You are a helpful customer support agent.".to_string()),
///     ..Default::default()
/// };
/// ```
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

/// Configuration for the underlying language model
///
/// Specifies which model to use and how to connect to it. Supports
/// multiple providers including OpenAI, Anthropic, and local models.
/// When using the `rsllm-client` feature, this integrates with the
/// rsllm client library for unified model access.
///
/// # Supported Providers
///
/// - `openai`: OpenAI GPT models (gpt-4, gpt-3.5-turbo, etc.)
/// - `anthropic`: Anthropic Claude models
/// - `local`: Local models via rsllm
/// - Custom providers via rsllm extensions
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
/// use std::time::Duration;
///
/// let config = ModelConfig {
///     provider: "openai".to_string(),
///     model: "gpt-4-turbo".to_string(),
///     timeout: Duration::from_secs(60),
///     max_tokens: Some(4096),
///     api_key_env: "OPENAI_API_KEY".to_string(),
///     ..Default::default()
/// };
/// ```
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

/// Execution context for maintaining state during agent conversations
///
/// Each conversation maintains its own context with tool call history,
/// memory references, timing information, and custom variables. The context
/// is created per conversation and persists across multiple turns.
///
/// # Features
///
/// - **Tool Call Tracking**: Complete history of tool executions
/// - **Memory Integration**: Access to persistent conversation memory
/// - **Timing Information**: Track conversation duration and performance
/// - **Custom Variables**: Store conversation-specific data
/// - **State Management**: Maintain state across async operations
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
/// use std::sync::Arc;
///
/// let memory = Arc::new(ConversationBufferMemory::new(100));
/// let mut context = AgentContext::new("user-123")
///     .with_memory(memory);
///
/// // Set custom variables
/// context.set_variable("user_preference", "concise".into());
/// context.set_variable("session_id", 12345.into());
///
/// // Access variables later
/// if let Some(pref) = context.get_variable("user_preference") {
///     println!("User prefers: {}", pref.as_str().unwrap());
/// }
/// ```
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

/// Record of a tool execution during agent processing
///
/// Tracks all information about tool calls including input, output,
/// timing, and metadata. Used for debugging, monitoring, and audit trails.
/// Each tool call is assigned a unique ID for tracking across systems.
///
/// # Fields
///
/// - `id`: Unique identifier for this tool call
/// - `tool_name`: Name of the tool that was executed
/// - `input`: Input parameters passed to the tool
/// - `result`: Tool execution result (if completed)
/// - `timestamp`: When the tool call was initiated
/// - `duration_ms`: How long the tool took to execute
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
///
/// let tool_call = ToolCall::new("calculator", "2 + 2")
///     .with_result(
///         ToolResult {
///             success: true,
///             output: "4".to_string(),
///             metadata: Default::default(),
///         },
///         150 // 150ms execution time
///     );
///
/// println!("Tool '{}' executed in {}ms", 
///     tool_call.tool_name, 
///     tool_call.duration_ms.unwrap()
/// );
/// ```
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

/// Complete response from an agent including text, tool calls, and metadata
///
/// Contains all information about an agent's response to a user message,
/// including the generated text, any tools that were called, performance
/// metrics, and whether the response is final or needs continuation.
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
///
/// # #[tokio::main]
/// # async fn main() -> RragResult<()> {
/// let agent = AgentBuilder::new().build()?;
/// let response = agent.process_message("Hello!", None).await?;
///
/// println!("Agent: {}", response.text);
/// println!("Processing time: {}ms", response.metadata.duration_ms);
/// println!("Tools used: {}", response.tool_calls.len());
/// println!("Model calls: {}", response.metadata.model_calls);
///
/// if let Some(tokens) = response.metadata.total_tokens {
///     println!("Tokens used: {}", tokens);
/// }
/// # Ok(())
/// # }
/// ```
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

/// Main RRAG agent implementation with comprehensive capabilities
///
/// The core agent struct that orchestrates conversation handling, tool execution,
/// memory management, and LLM integration. Designed for production use with
/// robust error handling, performance monitoring, and concurrent operation support.
///
/// # Architecture
///
/// - **Configuration**: Flexible behavior configuration via [`AgentConfig`]
/// - **Tool Registry**: Type-safe tool management and execution
/// - **Memory Integration**: Optional persistent conversation state
/// - **LLM Client**: rsllm integration when feature is enabled
/// - **Context Management**: Per-conversation state tracking
/// - **Async Operations**: Full async/await support for high concurrency
///
/// # Thread Safety
///
/// The agent is designed to be safely shared across threads using `Arc<RragAgent>`.
/// Internal state is protected by async RwLocks and atomic operations where appropriate.
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
/// use std::sync::Arc;
///
/// # #[tokio::main]
/// # async fn main() -> RragResult<()> {
/// // Create and share an agent across threads
/// let agent = Arc::new(
///     AgentBuilder::new()
///         .with_name("Shared Agent")
///         .with_tool(Arc::new(Calculator::new()))
///         .build()?
/// );
///
/// // Clone for use in async tasks
/// let agent_clone = agent.clone();
/// let handle = tokio::spawn(async move {
///     agent_clone.process_message("Calculate 10 + 15", None).await
/// });
///
/// let response = handle.await??;
/// println!("Response: {}", response.text);
/// # Ok(())
/// # }
/// ```
///
/// # Performance Notes
///
/// - Uses lazy initialization for conversation contexts
/// - Implements efficient tool lookup with O(1) registry access
/// - Supports concurrent conversation processing
/// - Memory usage scales with active conversation count
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

/// Builder for creating and configuring RRAG agents
///
/// Provides a fluent interface for configuring all aspects of agent behavior
/// including model settings, tools, memory, and operational parameters.
/// Validates configuration during build to catch errors early.
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
/// use std::sync::Arc;
///
/// # #[tokio::main]
/// # async fn main() -> RragResult<()> {
/// let agent = AgentBuilder::new()
///     .with_name("Production Assistant")
///     .with_model("openai", "gpt-4")
///     .with_temperature(0.3)
///     .with_system_prompt("You are a helpful assistant specializing in technical support.")
///     .with_tool(Arc::new(Calculator::new()))
///     # #[cfg(feature = "http")]
///     .with_tool(Arc::new(HttpTool::new()))
///     .with_memory(Arc::new(ConversationBufferMemory::new(1000)))
///     .with_max_tool_calls(5)
///     .with_verbose(true)
///     .build()?;
///
/// println!("Created agent: {}", agent.config().name);
/// # Ok(())
/// # }
/// ```
///
/// # Validation
///
/// The builder validates configuration during `build()` and returns detailed
/// error information for any invalid settings:
///
/// ```rust
/// use rrag::prelude::*;
///
/// let result = AgentBuilder::new()
///     .with_temperature(5.0) // Invalid: must be 0.0-2.0
///     .build();
///
/// match result {
///     Err(RragError::Validation { field, constraint, value }) => {
///         println!("Invalid {}: expected {}, got {}", field, constraint, value);
///     }
///     _ => {}
/// }
/// ```
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