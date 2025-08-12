# RGraph

Graph-based agent orchestration system built on the RRAG framework.

## Overview

RGraph provides a powerful graph-based orchestration layer for building complex AI agent workflows. It enables developers to create sophisticated multi-agent systems with conditional routing, parallel execution, and state management.

## Features

- **Graph-Based Workflows**: Define agent interactions as directed graphs
- **Conditional Routing**: Dynamic path selection based on runtime conditions
- **Parallel Execution**: Run multiple agents concurrently for improved performance
- **State Management**: Thread-safe state handling across agent executions
- **Tool Integration**: Seamless integration with RRAG's tool ecosystem
- **Observability**: Built-in monitoring and tracing capabilities
- **Type Safety**: Rust's type system ensures correctness at compile time

## Quick Start

```rust
use rgraph::prelude::*;

// Create a graph with agents
let graph = GraphBuilder::new()
    .add_node("agent1", AgentNode::new("Assistant 1"))
    .add_node("agent2", AgentNode::new("Assistant 2"))
    .add_edge("agent1", "agent2")
    .build()?;

// Execute the graph
let result = graph.execute(initial_state).await?;
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rgraph = "0.1.0"
```

## Documentation

For detailed documentation, visit [docs.rs/rgraph](https://docs.rs/rgraph).

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/leval-ai/rrag/blob/main/CONTRIBUTING.md).