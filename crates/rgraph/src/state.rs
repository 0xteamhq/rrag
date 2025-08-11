//! # Graph State Management
//! 
//! This module provides the state management system for RGraph workflows.
//! The state flows through the graph execution, accumulating results and
//! providing context for decision-making.

use crate::{RGraphError, RGraphResult};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Path to a value in the graph state (supports nested access)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StatePath(pub String);

impl StatePath {
    /// Create a new state path
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }
    
    /// Create a nested path
    pub fn nested(parent: impl Into<String>, child: impl Into<String>) -> Self {
        Self(format!("{}.{}", parent.into(), child.into()))
    }
    
    /// Get the path string
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Split path into components
    pub fn components(&self) -> Vec<&str> {
        self.0.split('.').collect()
    }
}

impl From<String> for StatePath {
    fn from(path: String) -> Self {
        StatePath(path)
    }
}

impl From<&str> for StatePath {
    fn from(path: &str) -> Self {
        StatePath(path.to_string())
    }
}

/// Values that can be stored in the graph state
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StateValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Array of values
    Array(Vec<StateValue>),
    /// Object/Map of values
    Object(HashMap<String, StateValue>),
    /// Null value
    Null,
    /// Binary data
    Bytes(Vec<u8>),
}

impl StateValue {
    /// Convert to string if possible
    pub fn as_string(&self) -> Option<&str> {
        match self {
            StateValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    /// Convert to integer if possible
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            StateValue::Integer(i) => Some(*i),
            StateValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }
    
    /// Convert to float if possible
    pub fn as_float(&self) -> Option<f64> {
        match self {
            StateValue::Float(f) => Some(*f),
            StateValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
    
    /// Convert to boolean if possible
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            StateValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    
    /// Convert to array if possible
    pub fn as_array(&self) -> Option<&Vec<StateValue>> {
        match self {
            StateValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
    
    /// Convert to object if possible
    pub fn as_object(&self) -> Option<&HashMap<String, StateValue>> {
        match self {
            StateValue::Object(obj) => Some(obj),
            _ => None,
        }
    }
    
    /// Check if the value is null
    pub fn is_null(&self) -> bool {
        matches!(self, StateValue::Null)
    }
    
    /// Get the type name of the value
    pub fn type_name(&self) -> &'static str {
        match self {
            StateValue::String(_) => "string",
            StateValue::Integer(_) => "integer",
            StateValue::Float(_) => "float",
            StateValue::Boolean(_) => "boolean",
            StateValue::Array(_) => "array",
            StateValue::Object(_) => "object",
            StateValue::Null => "null",
            StateValue::Bytes(_) => "bytes",
        }
    }
}

// Convenient conversions
impl From<String> for StateValue {
    fn from(s: String) -> Self {
        StateValue::String(s)
    }
}

impl From<&str> for StateValue {
    fn from(s: &str) -> Self {
        StateValue::String(s.to_string())
    }
}

impl From<i64> for StateValue {
    fn from(i: i64) -> Self {
        StateValue::Integer(i)
    }
}

impl From<i32> for StateValue {
    fn from(i: i32) -> Self {
        StateValue::Integer(i as i64)
    }
}

impl From<f64> for StateValue {
    fn from(f: f64) -> Self {
        StateValue::Float(f)
    }
}

impl From<f32> for StateValue {
    fn from(f: f32) -> Self {
        StateValue::Float(f as f64)
    }
}

impl From<bool> for StateValue {
    fn from(b: bool) -> Self {
        StateValue::Boolean(b)
    }
}

impl From<Vec<StateValue>> for StateValue {
    fn from(arr: Vec<StateValue>) -> Self {
        StateValue::Array(arr)
    }
}

impl From<HashMap<String, StateValue>> for StateValue {
    fn from(obj: HashMap<String, StateValue>) -> Self {
        StateValue::Object(obj)
    }
}

impl From<Vec<u8>> for StateValue {
    fn from(bytes: Vec<u8>) -> Self {
        StateValue::Bytes(bytes)
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Value> for StateValue {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(s) => StateValue::String(s),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    StateValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    StateValue::Float(f)
                } else {
                    StateValue::Null
                }
            },
            serde_json::Value::Bool(b) => StateValue::Boolean(b),
            serde_json::Value::Array(arr) => {
                StateValue::Array(arr.into_iter().map(StateValue::from).collect())
            },
            serde_json::Value::Object(obj) => {
                StateValue::Object(
                    obj.into_iter()
                        .map(|(k, v)| (k, StateValue::from(v)))
                        .collect()
                )
            },
            serde_json::Value::Null => StateValue::Null,
        }
    }
}

#[cfg(feature = "serde")]
impl From<StateValue> for serde_json::Value {
    fn from(value: StateValue) -> Self {
        match value {
            StateValue::String(s) => serde_json::Value::String(s),
            StateValue::Integer(i) => serde_json::Value::Number(i.into()),
            StateValue::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0))
            ),
            StateValue::Boolean(b) => serde_json::Value::Bool(b),
            StateValue::Array(arr) => {
                serde_json::Value::Array(arr.into_iter().map(serde_json::Value::from).collect())
            },
            StateValue::Object(obj) => {
                serde_json::Value::Object(
                    obj.into_iter()
                        .map(|(k, v)| (k, serde_json::Value::from(v)))
                        .collect()
                )
            },
            StateValue::Null => serde_json::Value::Null,
            StateValue::Bytes(_) => serde_json::Value::Null, // Can't represent bytes in JSON
        }
    }
}

/// The shared state that flows through the graph execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphState {
    /// The state data
    #[cfg_attr(feature = "serde", serde(skip, default = "default_data"))]
    data: Arc<RwLock<HashMap<String, StateValue>>>,
    /// Metadata about the state
    #[cfg_attr(feature = "serde", serde(skip, default = "default_metadata"))]
    metadata: Arc<RwLock<HashMap<String, StateValue>>>,
    /// Execution history
    #[cfg_attr(feature = "serde", serde(skip, default = "default_execution_log"))]
    execution_log: Arc<RwLock<Vec<StateHistoryEntry>>>,
}

/// Entry in the state execution history
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StateHistoryEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub node_id: String,
    pub operation: StateOperation,
    pub key: String,
    pub old_value: Option<StateValue>,
    pub new_value: Option<StateValue>,
}

/// Types of state operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StateOperation {
    Set,
    Get,
    Remove,
    Clear,
}

impl GraphState {
    /// Create a new empty graph state
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            execution_log: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Create a new graph state with initial data
    pub fn with_data(data: HashMap<String, StateValue>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            execution_log: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Set a value in the state
    pub fn set(&self, key: impl Into<String>, value: impl Into<StateValue>) -> &Self {
        let key = key.into();
        let value = value.into();
        
        // Log the operation
        self.log_operation("system", StateOperation::Set, &key, None, Some(value.clone()));
        
        // Set the value
        let mut data = self.data.write();
        data.insert(key, value);
        
        self
    }
    
    /// Set a value in the state with node context
    pub fn set_with_context(
        &self, 
        node_id: &str,
        key: impl Into<String>, 
        value: impl Into<StateValue>
    ) -> &Self {
        let key = key.into();
        let value = value.into();
        
        // Get old value for logging
        let old_value = self.data.read().get(&key).cloned();
        
        // Log the operation
        self.log_operation(node_id, StateOperation::Set, &key, old_value, Some(value.clone()));
        
        // Set the value
        let mut data = self.data.write();
        data.insert(key, value);
        
        self
    }
    
    /// Get a value from the state
    pub fn get(&self, key: &str) -> RGraphResult<StateValue> {
        let path = StatePath::new(key);
        self.get_by_path(&path)
    }
    
    /// Get a value by path (supports nested access)
    pub fn get_by_path(&self, path: &StatePath) -> RGraphResult<StateValue> {
        let components = path.components();
        let data = self.data.read();
        
        if components.len() == 1 {
            // Simple key access
            data.get(components[0])
                .cloned()
                .ok_or_else(|| RGraphError::state(format!("Key '{}' not found", components[0])))
        } else {
            // Nested access
            let mut current_value = data.get(components[0])
                .ok_or_else(|| RGraphError::state(format!("Key '{}' not found", components[0])))?;
            
            for component in &components[1..] {
                match current_value {
                    StateValue::Object(ref obj) => {
                        current_value = obj.get(*component)
                            .ok_or_else(|| RGraphError::state(
                                format!("Nested key '{}' not found", component)
                            ))?;
                    },
                    _ => return Err(RGraphError::state(
                        format!("Cannot access '{}' on non-object value", component)
                    )),
                }
            }
            
            Ok(current_value.clone())
        }
    }
    
    /// Get a typed value from the state
    pub fn get_typed<T>(&self, key: &str) -> RGraphResult<T>
    where
        T: TryFrom<StateValue>,
        T::Error: std::fmt::Display,
    {
        let value = self.get(key)?;
        T::try_from(value).map_err(|e| RGraphError::state(e.to_string()))
    }
    
    /// Check if a key exists in the state
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.read().contains_key(key)
    }
    
    /// Remove a value from the state
    pub fn remove(&self, key: &str) -> Option<StateValue> {
        let mut data = self.data.write();
        let old_value = data.remove(key);
        
        // Log the operation
        self.log_operation("system", StateOperation::Remove, key, old_value.clone(), None);
        
        old_value
    }
    
    /// Clear all data from the state
    pub fn clear(&self) {
        let mut data = self.data.write();
        data.clear();
        
        // Log the operation
        self.log_operation("system", StateOperation::Clear, "all", None, None);
    }
    
    /// Get all keys in the state
    pub fn keys(&self) -> Vec<String> {
        self.data.read().keys().cloned().collect()
    }
    
    /// Get the number of items in the state
    pub fn len(&self) -> usize {
        self.data.read().len()
    }
    
    /// Check if the state is empty
    pub fn is_empty(&self) -> bool {
        self.data.read().is_empty()
    }
    
    /// Merge another state into this one
    pub fn merge(&self, other: &GraphState) {
        let other_data = other.data.read();
        let mut data = self.data.write();
        
        for (key, value) in other_data.iter() {
            data.insert(key.clone(), value.clone());
        }
    }
    
    /// Create a snapshot of the current state
    pub fn snapshot(&self) -> HashMap<String, StateValue> {
        self.data.read().clone()
    }
    
    /// Set metadata
    pub fn set_metadata(&self, key: impl Into<String>, value: impl Into<StateValue>) {
        let mut metadata = self.metadata.write();
        metadata.insert(key.into(), value.into());
    }
    
    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<StateValue> {
        self.metadata.read().get(key).cloned()
    }
    
    /// Get execution history
    pub fn execution_history(&self) -> Vec<StateHistoryEntry> {
        self.execution_log.read().clone()
    }
    
    /// Convenience method to add input data
    pub fn with_input(self, key: impl Into<String>, value: impl Into<StateValue>) -> Self {
        self.set(key, value);
        self
    }
    
    /// Get output data as a specific type
    pub fn get_output<T>(&self, key: &str) -> RGraphResult<T>
    where
        T: TryFrom<StateValue>,
        T::Error: std::fmt::Display,
    {
        self.get_typed(key)
    }
    
    /// Log a state operation
    fn log_operation(
        &self,
        node_id: &str,
        operation: StateOperation,
        key: &str,
        old_value: Option<StateValue>,
        new_value: Option<StateValue>,
    ) {
        let entry = StateHistoryEntry {
            timestamp: chrono::Utc::now(),
            node_id: node_id.to_string(),
            operation,
            key: key.to_string(),
            old_value,
            new_value,
        };
        
        self.execution_log.write().push(entry);
    }
}

impl Default for GraphState {
    fn default() -> Self {
        Self::new()
    }
}

// Implement TryFrom for common types from StateValue
impl TryFrom<StateValue> for String {
    type Error = RGraphError;
    
    fn try_from(value: StateValue) -> Result<Self, Self::Error> {
        match value {
            StateValue::String(s) => Ok(s),
            _ => Err(RGraphError::state(
                format!("Cannot convert {} to String", value.type_name())
            )),
        }
    }
}

impl TryFrom<StateValue> for i64 {
    type Error = RGraphError;
    
    fn try_from(value: StateValue) -> Result<Self, Self::Error> {
        match value {
            StateValue::Integer(i) => Ok(i),
            StateValue::Float(f) => Ok(f as i64),
            _ => Err(RGraphError::state(
                format!("Cannot convert {} to i64", value.type_name())
            )),
        }
    }
}

impl TryFrom<StateValue> for f64 {
    type Error = RGraphError;
    
    fn try_from(value: StateValue) -> Result<Self, Self::Error> {
        match value {
            StateValue::Float(f) => Ok(f),
            StateValue::Integer(i) => Ok(i as f64),
            _ => Err(RGraphError::state(
                format!("Cannot convert {} to f64", value.type_name())
            )),
        }
    }
}

impl TryFrom<StateValue> for bool {
    type Error = RGraphError;
    
    fn try_from(value: StateValue) -> Result<Self, Self::Error> {
        match value {
            StateValue::Boolean(b) => Ok(b),
            _ => Err(RGraphError::state(
                format!("Cannot convert {} to bool", value.type_name())
            )),
        }
    }
}

impl TryFrom<StateValue> for Vec<StateValue> {
    type Error = RGraphError;
    
    fn try_from(value: StateValue) -> Result<Self, Self::Error> {
        match value {
            StateValue::Array(arr) => Ok(arr),
            _ => Err(RGraphError::state(
                format!("Cannot convert {} to Vec<StateValue>", value.type_name())
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_value_conversions() {
        let string_val: StateValue = "hello".into();
        assert_eq!(string_val.as_string(), Some("hello"));
        
        let int_val: StateValue = 42i64.into();
        assert_eq!(int_val.as_integer(), Some(42));
        
        let float_val: StateValue = 3.14f64.into();
        assert_eq!(float_val.as_float(), Some(3.14));
        
        let bool_val: StateValue = true.into();
        assert_eq!(bool_val.as_boolean(), Some(true));
    }
    
    #[test]
    fn test_graph_state_basic_operations() {
        let state = GraphState::new();
        
        // Test set and get
        state.set("key1", "value1");
        assert_eq!(state.get("key1").unwrap(), StateValue::String("value1".to_string()));
        
        // Test contains_key
        assert!(state.contains_key("key1"));
        assert!(!state.contains_key("nonexistent"));
        
        // Test remove
        let removed = state.remove("key1");
        assert_eq!(removed, Some(StateValue::String("value1".to_string())));
        assert!(!state.contains_key("key1"));
    }
    
    #[test]
    fn test_state_path() {
        let path = StatePath::new("parent.child.grandchild");
        let components = path.components();
        assert_eq!(components, vec!["parent", "child", "grandchild"]);
        
        let nested_path = StatePath::nested("parent", "child");
        assert_eq!(nested_path.as_str(), "parent.child");
    }
    
    #[test]
    fn test_state_with_input() {
        let state = GraphState::new()
            .with_input("name", "Alice")
            .with_input("age", 30);
        
        assert_eq!(state.get("name").unwrap().as_string(), Some("Alice"));
        assert_eq!(state.get("age").unwrap().as_integer(), Some(30));
    }
    
    #[test]
    fn test_state_merge() {
        let state1 = GraphState::new();
        state1.set("key1", "value1");
        
        let state2 = GraphState::new();
        state2.set("key2", "value2");
        
        state1.merge(&state2);
        
        assert!(state1.contains_key("key1"));
        assert!(state1.contains_key("key2"));
    }
    
    #[test]
    fn test_execution_history() {
        let state = GraphState::new();
        state.set_with_context("node1", "key1", "value1");
        state.set_with_context("node2", "key2", "value2");
        
        let history = state.execution_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].node_id, "node1");
        assert_eq!(history[1].node_id, "node2");
    }
}

// Default functions for serde skipped fields
#[cfg(feature = "serde")]
fn default_data() -> Arc<RwLock<HashMap<String, StateValue>>> {
    Arc::new(RwLock::new(HashMap::new()))
}

#[cfg(feature = "serde")]
fn default_metadata() -> Arc<RwLock<HashMap<String, StateValue>>> {
    Arc::new(RwLock::new(HashMap::new()))
}

#[cfg(feature = "serde")]
fn default_execution_log() -> Arc<RwLock<Vec<StateHistoryEntry>>> {
    Arc::new(RwLock::new(Vec::new()))
}