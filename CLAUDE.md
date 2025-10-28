# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is a Rust workspace containing the **RRAG** (Rust RAG Framework) with three main crates:

- **`crates/rsllm/`** - Multi-provider LLM client library with tool calling support
- **`crates/rrag/`** - RAG framework with agent system, retrieval, and document processing
- **`crates/rgraph/`** - Graph-based agent orchestration
- **`crates/schemars/`** - Vendored JSON Schema library (modified for OpenAI compatibility)
- **`crates/rsllm-macros/`** - Procedural macros for `#[tool]` attribute

## Common Commands

### Building
```bash
# Build entire workspace
cargo build

# Build specific crate
cargo build -p rsllm
cargo build -p rrag
cargo build -p rsllm-macros

# Build with all features
cargo build --all-features

# Build specific example
cargo build --bin agent_demo
cargo build --bin simple_agent
```

### Testing
```bash
# Run all tests
cargo test --workspace

# Test specific crate
cargo test -p rsllm --lib
cargo test -p rrag --all-features

# Test with features
cargo test -p rsllm --features ollama
```

### Code Quality
```bash
# Format all code
cargo fmt --all

# Run clippy
cargo clippy --all-features --workspace -- -D warnings

# Check without building
cargo check --workspace
```

### Running Examples
```bash
# RSLLM tool calling guide (comprehensive)
cargo run -p rsllm --example tool_calling_guide --all-features

# OpenAI compatibility verification
cargo run -p rsllm --example openai_compatibility_test --all-features

# RRAG agent demo (stateful/stateless modes)
cargo run --bin agent_demo

# Simple agent prototype
cargo run --bin simple_agent
```

## Architecture Overview

### RSLLM - LLM Client Library

**Key Innovation**: Automatic JSON Schema generation from Rust types with OpenAI compatibility.

**Tool Calling Framework** - Three approaches:
1. **`#[tool]` macro** (Recommended - 15 lines per tool):
```rust
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct Params {
    /// Field description (critical for LLM!)
    pub field: Type,
}

#[tool(description = "Tool description")]
fn my_tool(params: Params) -> Result<Result, Error> { }
```

2. **SchemaBasedTool trait** (For complex/stateful tools)
3. **Manual JSON** (Full control)
4. **simple_tool! macro** (Quick prototypes)

**Critical Details**:
- Vendored `schemars` configured for OpenAI compatibility (inline schemas, no `$ref`)
- Tool schemas MUST have descriptions on all fields to prevent LLM hallucination
- Supports Ollama and OpenAI providers with tool calling
- Ollama returns arguments as strings - automatic conversion to numbers implemented

**Provider Implementation**:
- `OllamaProvider::chat_completion_with_tools()` - Handles Ollama's tool format
- `OpenAIProvider::chat_completion_with_tools()` - Standard OpenAI format
- Both generate OpenAI-compatible tool schemas (no `$defs`, no `$ref`)

### RRAG - Agent Framework

**Key Innovation**: LangChain-style agent in Rust with real tool calling.

**Agent Module** (`crates/rrag/src/agent/`):
- `agent.rs` - Core Agent with tool calling loop
- `builder.rs` - Fluent AgentBuilder pattern
- `config.rs` - AgentConfig with ConversationMode enum (Stateless/Stateful)
- `memory.rs` - ConversationMemory with history management
- `executor.rs` - ToolExecutor for type-safe tool execution

**Agent Loop Flow**:
```
User Input
  ↓
Agent.run()
  ↓
LLM Call (with tool schemas)
  ↓
Tool Calls?
  Yes → Execute Tools → Add Results → Loop
  No → Final Answer
```

**Conversation Modes**:
- **Stateless**: Each `run()` call is independent (fresh conversation)
- **Stateful**: Maintains conversation history across `run()` calls

**Usage**:
```rust
let agent = AgentBuilder::new()
    .with_llm(client)
    .with_tools(vec![Box::new(CalculatorTool)])
    .stateful()  // or .stateless()
    .verbose(true)
    .build()?;

let response = agent.run("What is 2+2?").await?;
```

### Critical Implementation Details

**Schemars Modifications** (`crates/schemars/schemars/src/generate.rs`):
- Default changed from `draft2020_12()` to `draft07()`
- `inline_subschemas = true` - ALL schemas inlined (no `$ref`)
- `meta_schema = None` - No `$schema` field
- This ensures 100% OpenAI compatibility (verified with tests)

**Tool Arguments Type Conversion** (`crates/rsllm/src/provider.rs`):
- Ollama returns numbers as strings: `{"a": "156", "b": "23"}`
- Automatic conversion to numbers: `{"a": 156.0, "b": 23.0}`
- Handles both `f64` and `i64` parsing

**URL Handling** (`crates/rsllm/src/provider.rs`):
- `normalize_base_url()` adds trailing slash if missing
- Supports both `http://localhost:11434/api` and `http://localhost:11434/api/`
- Ensures correct URL joining for all endpoints

## Environment Variables

### RSLLM Configuration
```bash
# Provider and model
RSLLM_PROVIDER=ollama  # or openai, claude
RSLLM_MODEL=llama3.2:3b

# Provider-specific (takes precedence over generic)
RSLLM_OLLAMA_BASE_URL=http://localhost:11434/api/
RSLLM_OLLAMA_MODEL=llama3.2:3b
RSLLM_OPENAI_BASE_URL=https://api.openai.com/v1/
RSLLM_OPENAI_MODEL=gpt-4
RSLLM_API_KEY=your-key

# Other settings
RSLLM_TEMPERATURE=0.7
RSLLM_MAX_TOKENS=2000
```

## Key Design Patterns

### Tool Creation Pattern
Always include descriptions on struct fields using doc comments (`///`):
```rust
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct Params {
    /// First number (REQUIRED - prevents hallucination!)
    #[schemars(range(min = 0.0, max = 100.0))]
    pub a: f64,
}
```

### Agent Pattern
Stateful for chat, stateless for one-off queries:
```rust
// Chat application
agent.stateful().build()

// API endpoint (independent requests)
agent.stateless().build()
```

### Error Handling
RRAG errors include `agent_id` field:
```rust
RragError::Agent {
    agent_id: "agent-name".to_string(),
    message: "error message".to_string(),
    source: Some(Box::new(source_error)),
}
```

## Testing Strategy

**RSLLM Tests**:
- Unit tests in `crates/rsllm/src/` modules
- Integration tests via examples
- OpenAI compatibility verified in `openai_compatibility_test.rs`

**Agent Tests**:
- Run `agent_demo` for full integration test
- Requires Ollama running: `ollama serve` + `ollama pull llama3.2:3b`
- Tests both stateless and stateful modes

## Publishing to crates.io

Workflow: `.github/workflows/publish-crates.yml`

**Order matters** (rsllm-macros must be published first):
```bash
git tag rsllm-macros-v0.1.0
git tag rsllm-v0.1.0
git push origin rsllm-macros-v0.1.0 rsllm-v0.1.0
```

GitHub Actions automatically publishes in correct order.

## Important Notes

**Tool Descriptions**: Without field descriptions, LLMs hallucinate parameter meanings. Always use `///` doc comments.

**Schemars is Vendored**: Located in `crates/schemars/`. Do NOT update from crates.io - we use a modified version for OpenAI compatibility.

**Agent Loop**: Max 10 iterations by default to prevent infinite loops. Configure with `.with_max_iterations()`.

**Ollama Tool Support**: Only models with "Tools" badge support function calling (e.g., llama3.1, llama3.2).

## Repository Metadata

- **Author**: vasanth <vasanth@0xteam.io>
- **Repository**: https://github.com/0xteamhq/rrag
- **License**: MIT
