# Changelog

All notable changes to RGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - 2025-08-12

### Added
- Initial release of RGraph orchestration framework
- Graph-based agent workflow system
- Core graph execution engine
- Agent node implementation with RRAG integration
- Tool node for external tool execution
- Condition node for conditional routing
- Transform node for data transformations
- State management across graph executions
- Parallel execution capabilities
- Observability and monitoring support
- Integration with RRAG framework v0.1.0-alpha.2

### Features
- **Core Graph System**: Directed graph execution with topological ordering
- **Node Types**: Agent, Tool, Condition, and Transform nodes
- **Routing**: Dynamic conditional routing based on runtime state
- **State Management**: Thread-safe state handling with DashMap
- **Execution Modes**: Sequential and parallel execution strategies
- **Error Handling**: Comprehensive error types with recovery strategies
- **Type Safety**: Full Rust type safety with compile-time guarantees

### Dependencies
- Built on RRAG v0.1.0-alpha.2
- Tokio async runtime
- Petgraph for graph algorithms
- DashMap for concurrent state management

## License

MIT License - see LICENSE file for details.