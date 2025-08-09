//! # RRAG Tools System
//! 
//! Type-safe tool system leveraging Rust's trait system for zero-cost abstractions.
//! Designed for async execution with proper error handling and resource management.

use crate::{RragError, RragResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Tool execution result with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the tool executed successfully
    pub success: bool,
    
    /// Tool output content
    pub output: String,
    
    /// Execution metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    
    /// Resource usage information
    pub resource_usage: Option<ResourceUsage>,
}

impl ToolResult {
    /// Create a successful result
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            metadata: HashMap::new(),
            execution_time_ms: 0,
            resource_usage: None,
        }
    }

    /// Create an error result
    pub fn error(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: error.into(),
            metadata: HashMap::new(),
            execution_time_ms: 0,
            resource_usage: None,
        }
    }

    /// Add metadata using builder pattern
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set execution timing
    pub fn with_timing(mut self, execution_time_ms: u64) -> Self {
        self.execution_time_ms = execution_time_ms;
        self
    }

    /// Set resource usage
    pub fn with_resource_usage(mut self, usage: ResourceUsage) -> Self {
        self.resource_usage = Some(usage);
        self
    }
}

/// Resource usage tracking for tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory allocated in bytes
    pub memory_bytes: Option<u64>,
    
    /// CPU time used in microseconds
    pub cpu_time_us: Option<u64>,
    
    /// Network requests made
    pub network_requests: Option<u32>,
    
    /// Files accessed
    pub files_accessed: Option<u32>,
}

/// Core tool trait optimized for Rust's async ecosystem
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool identifier (used for registration and calling)
    fn name(&self) -> &str;
    
    /// Human-readable description for LLM context
    fn description(&self) -> &str;
    
    /// JSON schema for parameter validation (optional)
    fn schema(&self) -> Option<serde_json::Value> {
        None
    }
    
    /// Execute the tool with string input
    async fn execute(&self, input: &str) -> RragResult<ToolResult>;
    
    /// Execute with structured parameters (default delegates to execute)
    async fn execute_with_params(&self, params: serde_json::Value) -> RragResult<ToolResult> {
        let input = match params {
            serde_json::Value::String(s) => s,
            _ => params.to_string(),
        };
        self.execute(&input).await
    }
    
    /// Tool capabilities for filtering and discovery
    fn capabilities(&self) -> Vec<&'static str> {
        vec![]
    }
    
    /// Whether this tool requires authentication
    fn requires_auth(&self) -> bool {
        false
    }
    
    /// Tool category for organization
    fn category(&self) -> &'static str {
        "general"
    }
    
    /// Whether this tool can be cached
    fn is_cacheable(&self) -> bool {
        false
    }
    
    /// Cost estimate for execution (arbitrary units)
    fn cost_estimate(&self) -> u32 {
        1
    }
}

/// Macro for creating simple tools with less boilerplate
#[macro_export]
macro_rules! rrag_tool {
    (
        name: $name:expr,
        description: $desc:expr,
        execute: $exec:expr
    ) => {
        #[derive(Debug)]
        pub struct GeneratedTool;
        
        #[async_trait::async_trait]
        impl Tool for GeneratedTool {
            fn name(&self) -> &str {
                $name
            }
            
            fn description(&self) -> &str {
                $desc
            }
            
            async fn execute(&self, input: &str) -> RragResult<ToolResult> {
                let start = std::time::Instant::now();
                let result = ($exec)(input).await;
                let execution_time = start.elapsed().as_millis() as u64;
                
                match result {
                    Ok(output) => Ok(ToolResult::success(output).with_timing(execution_time)),
                    Err(e) => Ok(ToolResult::error(e.to_string()).with_timing(execution_time)),
                }
            }
        }
    };
    
    (
        name: $name:expr,
        description: $desc:expr,
        category: $category:expr,
        execute: $exec:expr
    ) => {
        #[derive(Debug)]
        pub struct GeneratedTool;
        
        #[async_trait::async_trait]
        impl Tool for GeneratedTool {
            fn name(&self) -> &str {
                $name
            }
            
            fn description(&self) -> &str {
                $desc
            }
            
            fn category(&self) -> &'static str {
                $category
            }
            
            async fn execute(&self, input: &str) -> RragResult<ToolResult> {
                let start = std::time::Instant::now();
                let result = ($exec)(input).await;
                let execution_time = start.elapsed().as_millis() as u64;
                
                match result {
                    Ok(output) => Ok(ToolResult::success(output).with_timing(execution_time)),
                    Err(e) => Ok(ToolResult::error(e.to_string()).with_timing(execution_time)),
                }
            }
        }
    };
}

/// Thread-safe tool registry using Arc for efficient sharing
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Create registry with pre-registered tools
    pub fn with_tools(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut registry = HashMap::new();
        for tool in tools {
            registry.insert(tool.name().to_string(), tool);
        }
        
        Self { tools: registry }
    }

    /// Register a new tool
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> RragResult<()> {
        let name = tool.name().to_string();
        
        if self.tools.contains_key(&name) {
            return Err(RragError::config(
                "tool_name",
                "unique name",
                format!("duplicate: {}", name),
            ));
        }
        
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// List all registered tool names
    pub fn list_tools(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// List tools by category
    pub fn list_by_category(&self, category: &str) -> Vec<Arc<dyn Tool>> {
        self.tools
            .values()
            .filter(|tool| tool.category() == category)
            .cloned()
            .collect()
    }

    /// List tools by capability
    pub fn list_by_capability(&self, capability: &str) -> Vec<Arc<dyn Tool>> {
        self.tools
            .values()
            .filter(|tool| tool.capabilities().contains(&capability))
            .cloned()
            .collect()
    }

    /// Execute a tool by name
    pub async fn execute(&self, tool_name: &str, input: &str) -> RragResult<ToolResult> {
        let tool = self.get(tool_name)
            .ok_or_else(|| RragError::tool_execution(tool_name, "Tool not found"))?;
        
        tool.execute(input).await
    }

    /// Get tool schemas for LLM context
    pub fn get_tool_schemas(&self) -> HashMap<String, serde_json::Value> {
        self.tools
            .iter()
            .filter_map(|(name, tool)| {
                tool.schema().map(|schema| (name.clone(), schema))
            })
            .collect()
    }

    /// Get tool descriptions for LLM context
    pub fn get_tool_descriptions(&self) -> HashMap<String, String> {
        self.tools
            .iter()
            .map(|(name, tool)| (name.clone(), tool.description().to_string()))
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in calculator tool
#[derive(Debug)]
pub struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }
    
    fn description(&self) -> &str {
        "Performs mathematical calculations. Input should be a mathematical expression like '2+2', '10*5', or '15/3'."
    }
    
    fn category(&self) -> &'static str {
        "math"
    }
    
    fn capabilities(&self) -> Vec<&'static str> {
        vec!["math", "calculation", "arithmetic"]
    }
    
    fn is_cacheable(&self) -> bool {
        true // Math results are deterministic
    }
    
    async fn execute(&self, input: &str) -> RragResult<ToolResult> {
        let start = Instant::now();
        
        match calculate(input) {
            Ok(result) => {
                let execution_time = start.elapsed().as_millis() as u64;
                Ok(ToolResult::success(result.to_string())
                    .with_timing(execution_time)
                    .with_metadata("expression", serde_json::Value::String(input.to_string()))
                    .with_metadata("result_type", serde_json::Value::String("number".to_string())))
            }
            Err(e) => {
                let execution_time = start.elapsed().as_millis() as u64;
                Ok(ToolResult::error(format!("Calculation error: {}", e))
                    .with_timing(execution_time))
            }
        }
    }
    
    fn schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                    "examples": ["2+2", "10*5", "15/3", "sqrt(16)", "2^3"]
                }
            },
            "required": ["expression"]
        }))
    }
}

/// Simple calculator implementation
fn calculate(expr: &str) -> RragResult<f64> {
    let expr = expr.trim().replace(" ", "");
    
    // Handle basic operations in order of precedence
    if let Some(result) = try_parse_number(&expr) {
        return Ok(result);
    }
    
    // Addition and subtraction (lowest precedence)
    if let Some(pos) = expr.rfind('+') {
        let (left, right) = expr.split_at(pos);
        let right = &right[1..];
        return Ok(calculate(left)? + calculate(right)?);
    }
    
    if let Some(pos) = expr.rfind('-') {
        if pos > 0 { // Avoid treating negative numbers as subtraction
            let (left, right) = expr.split_at(pos);
            let right = &right[1..];
            return Ok(calculate(left)? - calculate(right)?);
        }
    }
    
    // Multiplication and division
    if let Some(pos) = expr.rfind('*') {
        let (left, right) = expr.split_at(pos);
        let right = &right[1..];
        return Ok(calculate(left)? * calculate(right)?);
    }
    
    if let Some(pos) = expr.rfind('/') {
        let (left, right) = expr.split_at(pos);
        let right = &right[1..];
        let right_val = calculate(right)?;
        if right_val == 0.0 {
            return Err(RragError::tool_execution("calculator", "Division by zero"));
        }
        return Ok(calculate(left)? / right_val);
    }
    
    // Power operation
    if let Some(pos) = expr.find('^') {
        let (left, right) = expr.split_at(pos);
        let right = &right[1..];
        return Ok(calculate(left)?.powf(calculate(right)?));
    }
    
    // Functions
    if expr.starts_with("sqrt(") && expr.ends_with(')') {
        let inner = &expr[5..expr.len()-1];
        let value = calculate(inner)?;
        if value < 0.0 {
            return Err(RragError::tool_execution("calculator", "Square root of negative number"));
        }
        return Ok(value.sqrt());
    }
    
    if expr.starts_with("sin(") && expr.ends_with(')') {
        let inner = &expr[4..expr.len()-1];
        return Ok(calculate(inner)?.sin());
    }
    
    if expr.starts_with("cos(") && expr.ends_with(')') {
        let inner = &expr[4..expr.len()-1];
        return Ok(calculate(inner)?.cos());
    }
    
    // Parentheses
    if expr.starts_with('(') && expr.ends_with(')') {
        let inner = &expr[1..expr.len()-1];
        return calculate(inner);
    }
    
    Err(RragError::tool_execution("calculator", format!("Invalid expression: {}", expr)))
}

fn try_parse_number(s: &str) -> Option<f64> {
    s.parse().ok()
}

/// Echo tool for testing and debugging
#[derive(Debug)]
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    
    fn description(&self) -> &str {
        "Echoes back the input text. Useful for testing and debugging."
    }
    
    fn category(&self) -> &'static str {
        "utility"
    }
    
    fn capabilities(&self) -> Vec<&'static str> {
        vec!["test", "debug", "echo"]
    }
    
    async fn execute(&self, input: &str) -> RragResult<ToolResult> {
        let start = Instant::now();
        let output = format!("Echo: {}", input);
        let execution_time = start.elapsed().as_millis() as u64;
        
        Ok(ToolResult::success(output)
            .with_timing(execution_time)
            .with_metadata("input_length", serde_json::Value::Number(input.len().into())))
    }
}

/// HTTP client tool for web requests (requires "http" feature)
#[cfg(feature = "http")]
#[derive(Debug)]
pub struct HttpTool {
    client: reqwest::Client,
}

#[cfg(feature = "http")]
impl HttpTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }
}

#[cfg(feature = "http")]
#[async_trait]
impl Tool for HttpTool {
    fn name(&self) -> &str {
        "http"
    }
    
    fn description(&self) -> &str {
        "Makes HTTP GET requests to fetch web content. Input should be a valid URL."
    }
    
    fn category(&self) -> &'static str {
        "web"
    }
    
    fn capabilities(&self) -> Vec<&'static str> {
        vec!["web", "http", "fetch", "scraping"]
    }
    
    async fn execute(&self, input: &str) -> RragResult<ToolResult> {
        let start = Instant::now();
        
        let url = input.trim();
        if !url.starts_with("http://") && !url.starts_with("https://") {
            let execution_time = start.elapsed().as_millis() as u64;
            return Ok(ToolResult::error("URL must start with http:// or https://")
                .with_timing(execution_time));
        }
        
        match self.client.get(url).send().await {
            Ok(response) => {
                let status = response.status();
                let headers_count = response.headers().len();
                
                match response.text().await {
                    Ok(body) => {
                        let execution_time = start.elapsed().as_millis() as u64;
                        let truncated_body = if body.len() > 10000 {
                            format!("{}... [truncated from {} chars]", &body[..10000], body.len())
                        } else {
                            body
                        };
                        
                        Ok(ToolResult::success(truncated_body)
                            .with_timing(execution_time)
                            .with_metadata("status_code", serde_json::Value::Number(status.as_u16().into()))
                            .with_metadata("headers_count", serde_json::Value::Number(headers_count.into()))
                            .with_metadata("url", serde_json::Value::String(url.to_string())))
                    }
                    Err(e) => {
                        let execution_time = start.elapsed().as_millis() as u64;
                        Ok(ToolResult::error(format!("Failed to read response body: {}", e))
                            .with_timing(execution_time))
                    }
                }
            }
            Err(e) => {
                let execution_time = start.elapsed().as_millis() as u64;
                Ok(ToolResult::error(format!("HTTP request failed: {}", e))
                    .with_timing(execution_time))
            }
        }
    }
    
    fn schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calculator_tool() {
        let calc = Calculator;
        
        let result = calc.execute("2+2").await.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "4");
        
        let result = calc.execute("10*5").await.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "50");
        
        let result = calc.execute("sqrt(16)").await.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "4");
    }

    #[tokio::test]
    async fn test_echo_tool() {
        let echo = EchoTool;
        let result = echo.execute("hello world").await.unwrap();
        
        assert!(result.success);
        assert_eq!(result.output, "Echo: hello world");
        assert!(result.execution_time_ms > 0);
    }

    #[tokio::test]
    async fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        
        registry.register(Arc::new(Calculator)).unwrap();
        registry.register(Arc::new(EchoTool)).unwrap();
        
        assert_eq!(registry.list_tools().len(), 2);
        assert!(registry.list_tools().contains(&"calculator".to_string()));
        assert!(registry.list_tools().contains(&"echo".to_string()));
        
        let result = registry.execute("calculator", "5*5").await.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "25");
    }

    #[test]
    fn test_calculator_functions() {
        assert_eq!(calculate("2+2").unwrap(), 4.0);
        assert_eq!(calculate("10-3").unwrap(), 7.0);
        assert_eq!(calculate("4*5").unwrap(), 20.0);
        assert_eq!(calculate("15/3").unwrap(), 5.0);
        assert_eq!(calculate("2^3").unwrap(), 8.0);
        assert_eq!(calculate("sqrt(9)").unwrap(), 3.0);
        assert_eq!(calculate("(2+3)*4").unwrap(), 20.0);
    }

    #[test]
    fn test_calculator_errors() {
        assert!(calculate("5/0").is_err());
        assert!(calculate("sqrt(-1)").is_err());
        assert!(calculate("invalid").is_err());
    }

    #[test]
    fn test_tool_categories() {
        let calc = Calculator;
        assert_eq!(calc.category(), "math");
        assert!(calc.capabilities().contains(&"math"));
        assert!(calc.is_cacheable());
        
        let echo = EchoTool;
        assert_eq!(echo.category(), "utility");
        assert!(echo.capabilities().contains(&"test"));
    }
}