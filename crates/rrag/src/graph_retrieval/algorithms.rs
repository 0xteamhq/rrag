//! # Graph Algorithms
//!
//! Implementation of graph-based retrieval algorithms including PageRank,
//! graph traversal, and semantic path finding.

use super::{GraphEdge, GraphError, KnowledgeGraph};
use crate::RragResult;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Graph algorithms implementation
pub struct GraphAlgorithms;

impl GraphAlgorithms {
    /// Calculate PageRank scores for all nodes
    pub fn pagerank(
        graph: &KnowledgeGraph,
        config: &PageRankConfig,
    ) -> RragResult<HashMap<String, f32>> {
        let mut scores = HashMap::new();
        let node_count = graph.nodes.len();

        if node_count == 0 {
            return Ok(scores);
        }

        // Initialize scores
        let initial_score = 1.0 / node_count as f32;
        for node_id in graph.nodes.keys() {
            scores.insert(node_id.clone(), initial_score);
        }

        // Calculate outbound link counts
        let mut outbound_counts = HashMap::new();
        for (node_id, neighbors) in &graph.adjacency_list {
            outbound_counts.insert(node_id.clone(), neighbors.len().max(1)); // Avoid division by zero
        }

        // Iterative PageRank calculation
        for _iteration in 0..config.max_iterations {
            let mut new_scores = HashMap::new();
            let mut convergence_diff = 0.0;

            for node_id in graph.nodes.keys() {
                let mut score = config.damping_factor / node_count as f32;

                // Add contributions from incoming links
                if let Some(incoming_neighbors) = graph.reverse_adjacency_list.get(node_id) {
                    for neighbor_id in incoming_neighbors {
                        if let (Some(neighbor_score), Some(neighbor_outbound_count)) =
                            (scores.get(neighbor_id), outbound_counts.get(neighbor_id))
                        {
                            // Get edge weight if available
                            let edge_weight = graph
                                .edges
                                .values()
                                .find(|edge| {
                                    edge.source_id == *neighbor_id && edge.target_id == *node_id
                                })
                                .map(|edge| edge.weight)
                                .unwrap_or(1.0);

                            score += (1.0 - config.damping_factor) * (neighbor_score * edge_weight)
                                / (*neighbor_outbound_count as f32);
                        }
                    }
                }

                // Apply personalization if configured
                if let Some(personalization) = &config.personalization {
                    if let Some(personal_score) = personalization.get(node_id) {
                        score = config.personalization_weight * personal_score
                            + (1.0 - config.personalization_weight) * score;
                    }
                }

                let old_score = scores.get(node_id).unwrap_or(&0.0);
                convergence_diff += (score - old_score).abs();
                new_scores.insert(node_id.clone(), score);
            }

            scores = new_scores;

            // Check for convergence
            if convergence_diff < config.convergence_threshold {
                break;
            }
        }

        Ok(scores)
    }

    /// Find shortest paths from a source node using Dijkstra's algorithm
    pub fn shortest_paths(
        graph: &KnowledgeGraph,
        source_node_id: &str,
        config: &TraversalConfig,
    ) -> RragResult<HashMap<String, PathInfo>> {
        if !graph.nodes.contains_key(source_node_id) {
            return Err(GraphError::Algorithm {
                algorithm: "shortest_paths".to_string(),
                message: format!("Source node '{}' not found", source_node_id),
            }
            .into());
        }

        let mut distances = HashMap::new();
        let mut previous = HashMap::new();
        let mut heap = BinaryHeap::new();

        // Initialize distances
        for node_id in graph.nodes.keys() {
            distances.insert(node_id.clone(), f32::INFINITY);
        }
        distances.insert(source_node_id.to_string(), 0.0);
        heap.push(DijkstraState {
            cost: 0.0,
            node_id: source_node_id.to_string(),
        });

        while let Some(current) = heap.pop() {
            if current.cost > *distances.get(&current.node_id).unwrap_or(&f32::INFINITY) {
                continue;
            }

            // Check max distance limit
            if current.cost > config.max_distance {
                continue;
            }

            // Explore neighbors
            if let Some(neighbors) = graph.adjacency_list.get(&current.node_id) {
                for neighbor_id in neighbors {
                    // Calculate edge weight/cost
                    let edge_cost = graph
                        .edges
                        .values()
                        .find(|edge| {
                            edge.source_id == current.node_id && edge.target_id == *neighbor_id
                        })
                        .map(|edge| Self::calculate_edge_cost(edge, config))
                        .unwrap_or(1.0);

                    let new_cost = current.cost + edge_cost;
                    let neighbor_distance = distances.get(neighbor_id).unwrap_or(&f32::INFINITY);

                    if new_cost < *neighbor_distance {
                        distances.insert(neighbor_id.clone(), new_cost);
                        previous.insert(neighbor_id.clone(), current.node_id.clone());
                        heap.push(DijkstraState {
                            cost: new_cost,
                            node_id: neighbor_id.clone(),
                        });
                    }
                }
            }
        }

        // Build path information
        let mut result = HashMap::new();
        for (node_id, distance) in distances {
            if distance < f32::INFINITY {
                let path = Self::reconstruct_path(&previous, source_node_id, &node_id);
                let hop_count = path.len().saturating_sub(1);
                result.insert(
                    node_id,
                    PathInfo {
                        distance,
                        path,
                        hop_count,
                    },
                );
            }
        }

        Ok(result)
    }

    /// Find semantic paths between two nodes
    pub fn semantic_paths(
        graph: &KnowledgeGraph,
        source_node_id: &str,
        target_node_id: &str,
        config: &PathFindingConfig,
    ) -> RragResult<Vec<SemanticPath>> {
        if !graph.nodes.contains_key(source_node_id) {
            return Err(GraphError::Algorithm {
                algorithm: "semantic_paths".to_string(),
                message: format!("Source node '{}' not found", source_node_id),
            }
            .into());
        }

        if !graph.nodes.contains_key(target_node_id) {
            return Err(GraphError::Algorithm {
                algorithm: "semantic_paths".to_string(),
                message: format!("Target node '{}' not found", target_node_id),
            }
            .into());
        }

        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut current_path = Vec::new();

        Self::dfs_semantic_paths(
            graph,
            source_node_id,
            target_node_id,
            config,
            &mut visited,
            &mut current_path,
            &mut paths,
            0.0,
            0,
        );

        // Sort paths by semantic score (descending)
        paths.sort_by(|a, b| {
            b.semantic_score
                .partial_cmp(&a.semantic_score)
                .unwrap_or(Ordering::Equal)
        });

        // Limit number of returned paths
        paths.truncate(config.max_paths);

        Ok(paths)
    }

    /// Breadth-first search from a source node
    pub fn bfs_search(
        graph: &KnowledgeGraph,
        source_node_id: &str,
        config: &TraversalConfig,
    ) -> RragResult<Vec<String>> {
        if !graph.nodes.contains_key(source_node_id) {
            return Err(GraphError::Algorithm {
                algorithm: "bfs_search".to_string(),
                message: format!("Source node '{}' not found", source_node_id),
            }
            .into());
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back((source_node_id.to_string(), 0));
        visited.insert(source_node_id.to_string());

        while let Some((current_node_id, depth)) = queue.pop_front() {
            result.push(current_node_id.clone());

            // Check depth limit
            if depth >= config.max_depth {
                continue;
            }

            // Explore neighbors
            if let Some(neighbors) = graph.adjacency_list.get(&current_node_id) {
                for neighbor_id in neighbors {
                    if !visited.contains(neighbor_id) {
                        visited.insert(neighbor_id.clone());
                        queue.push_back((neighbor_id.clone(), depth + 1));

                        // Check max nodes limit
                        if result.len() + queue.len() >= config.max_nodes {
                            break;
                        }
                    }
                }

                if result.len() + queue.len() >= config.max_nodes {
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Depth-first search from a source node
    pub fn dfs_search(
        graph: &KnowledgeGraph,
        source_node_id: &str,
        config: &TraversalConfig,
    ) -> RragResult<Vec<String>> {
        if !graph.nodes.contains_key(source_node_id) {
            return Err(GraphError::Algorithm {
                algorithm: "dfs_search".to_string(),
                message: format!("Source node '{}' not found", source_node_id),
            }
            .into());
        }

        let mut visited = HashSet::new();
        let mut result = Vec::new();

        Self::dfs_recursive(graph, source_node_id, config, &mut visited, &mut result, 0);

        Ok(result)
    }

    /// Find strongly connected components using Tarjan's algorithm
    pub fn strongly_connected_components(graph: &KnowledgeGraph) -> Vec<Vec<String>> {
        let mut components = Vec::new();
        let mut visited = HashMap::new();
        let mut low_link = HashMap::new();
        let mut stack = Vec::new();
        let mut on_stack = HashSet::new();
        let mut index = 0;

        for node_id in graph.nodes.keys() {
            if !visited.contains_key(node_id) {
                Self::tarjan_scc(
                    graph,
                    node_id,
                    &mut visited,
                    &mut low_link,
                    &mut stack,
                    &mut on_stack,
                    &mut components,
                    &mut index,
                );
            }
        }

        components
    }

    /// Calculate betweenness centrality for all nodes
    pub fn betweenness_centrality(graph: &KnowledgeGraph) -> HashMap<String, f32> {
        let mut centrality = HashMap::new();
        let nodes: Vec<_> = graph.nodes.keys().collect();

        // Initialize centrality scores
        for node_id in &nodes {
            centrality.insert(node_id.to_string(), 0.0);
        }

        // For each node as source
        for &source in &nodes {
            let mut stack = Vec::new();
            let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();
            let mut sigma: HashMap<String, f32> = HashMap::new();
            let mut distance: HashMap<String, i32> = HashMap::new();
            let mut delta: HashMap<String, f32> = HashMap::new();
            let mut queue = VecDeque::new();

            // Initialize
            for node_id in &nodes {
                predecessors.insert(node_id.to_string(), Vec::new());
                sigma.insert(node_id.to_string(), 0.0);
                distance.insert(node_id.to_string(), -1);
                delta.insert(node_id.to_string(), 0.0);
            }

            sigma.insert(source.to_string(), 1.0);
            distance.insert(source.to_string(), 0);
            queue.push_back(source.to_string());

            // BFS
            while let Some(current) = queue.pop_front() {
                stack.push(current.clone());

                if let Some(neighbors) = graph.adjacency_list.get(&current) {
                    for neighbor in neighbors {
                        let neighbor_distance = *distance.get(neighbor).unwrap();
                        let current_distance = *distance.get(&current).unwrap();

                        // Found for the first time?
                        if neighbor_distance < 0 {
                            queue.push_back(neighbor.clone());
                            distance.insert(neighbor.clone(), current_distance + 1);
                        }

                        // Shortest path to neighbor via current?
                        if neighbor_distance == current_distance + 1 {
                            let current_sigma = *sigma.get(&current).unwrap();
                            let neighbor_sigma = sigma.get_mut(neighbor).unwrap();
                            *neighbor_sigma += current_sigma;
                            predecessors
                                .get_mut(neighbor)
                                .unwrap()
                                .push(current.clone());
                        }
                    }
                }
            }

            // Accumulation
            while let Some(node) = stack.pop() {
                if let Some(preds) = predecessors.get(&node) {
                    for pred in preds {
                        let node_sigma = *sigma.get(&node).unwrap();
                        let pred_sigma = *sigma.get(pred).unwrap();
                        let node_delta = *delta.get(&node).unwrap();

                        if pred_sigma > 0.0 {
                            let pred_delta = delta.get_mut(pred).unwrap();
                            *pred_delta += (pred_sigma / node_sigma) * (1.0 + node_delta);
                        }
                    }
                }

                if node != *source {
                    let node_delta = *delta.get(&node).unwrap();
                    let node_centrality = centrality.get_mut(&node).unwrap();
                    *node_centrality += node_delta;
                }
            }
        }

        // Normalize
        let node_count = nodes.len();
        if node_count > 2 {
            let normalization = ((node_count - 1) * (node_count - 2)) as f32;
            for value in centrality.values_mut() {
                *value /= normalization;
            }
        }

        centrality
    }

    // Helper methods

    fn calculate_edge_cost(edge: &GraphEdge, config: &TraversalConfig) -> f32 {
        match config.cost_function {
            CostFunction::Weight => 1.0 / edge.weight.max(0.001), // Higher weight = lower cost
            CostFunction::InverseConfidence => 1.0 / edge.confidence.max(0.001),
            CostFunction::Uniform => 1.0,
        }
    }

    fn reconstruct_path(
        previous: &HashMap<String, String>,
        source: &str,
        target: &str,
    ) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = target.to_string();

        while current != source {
            path.push(current.clone());
            if let Some(prev) = previous.get(&current) {
                current = prev.clone();
            } else {
                return Vec::new(); // No path found
            }
        }

        path.push(source.to_string());
        path.reverse();
        path
    }

    fn dfs_semantic_paths(
        graph: &KnowledgeGraph,
        current_node_id: &str,
        target_node_id: &str,
        config: &PathFindingConfig,
        visited: &mut HashSet<String>,
        current_path: &mut Vec<String>,
        all_paths: &mut Vec<SemanticPath>,
        current_score: f32,
        depth: usize,
    ) {
        if depth > config.max_depth || all_paths.len() >= config.max_paths {
            return;
        }

        current_path.push(current_node_id.to_string());
        visited.insert(current_node_id.to_string());

        if current_node_id == target_node_id {
            // Found a path
            let semantic_path = SemanticPath {
                nodes: current_path.clone(),
                semantic_score: current_score,
                path_length: current_path.len() - 1,
                edge_types: Self::extract_edge_types_from_path(graph, current_path),
            };
            all_paths.push(semantic_path);
        } else {
            // Continue exploring
            if let Some(neighbors) = graph.adjacency_list.get(current_node_id) {
                for neighbor_id in neighbors {
                    if !visited.contains(neighbor_id) {
                        // Calculate semantic score contribution
                        let edge_score = graph
                            .edges
                            .values()
                            .find(|edge| {
                                edge.source_id == current_node_id && edge.target_id == *neighbor_id
                            })
                            .map(|edge| Self::calculate_semantic_score(edge, config))
                            .unwrap_or(0.0);

                        let new_score = current_score + edge_score;

                        if new_score >= config.min_semantic_score {
                            Self::dfs_semantic_paths(
                                graph,
                                neighbor_id,
                                target_node_id,
                                config,
                                visited,
                                current_path,
                                all_paths,
                                new_score,
                                depth + 1,
                            );
                        }
                    }
                }
            }
        }

        current_path.pop();
        visited.remove(current_node_id);
    }

    fn extract_edge_types_from_path(graph: &KnowledgeGraph, path: &[String]) -> Vec<String> {
        let mut edge_types = Vec::new();

        for i in 0..(path.len().saturating_sub(1)) {
            if let Some(edge) = graph
                .edges
                .values()
                .find(|edge| edge.source_id == path[i] && edge.target_id == path[i + 1])
            {
                edge_types.push(edge.label.clone());
            }
        }

        edge_types
    }

    fn calculate_semantic_score(edge: &GraphEdge, config: &PathFindingConfig) -> f32 {
        let base_score = edge.confidence * edge.weight;

        // Apply semantic type weighting
        let type_weight = config
            .semantic_weights
            .get(&edge.edge_type)
            .copied()
            .unwrap_or(1.0);

        base_score * type_weight
    }

    fn dfs_recursive(
        graph: &KnowledgeGraph,
        current_node_id: &str,
        config: &TraversalConfig,
        visited: &mut HashSet<String>,
        result: &mut Vec<String>,
        depth: usize,
    ) {
        if depth > config.max_depth || result.len() >= config.max_nodes {
            return;
        }

        visited.insert(current_node_id.to_string());
        result.push(current_node_id.to_string());

        if let Some(neighbors) = graph.adjacency_list.get(current_node_id) {
            for neighbor_id in neighbors {
                if !visited.contains(neighbor_id) && result.len() < config.max_nodes {
                    Self::dfs_recursive(graph, neighbor_id, config, visited, result, depth + 1);
                }
            }
        }
    }

    fn tarjan_scc(
        graph: &KnowledgeGraph,
        node_id: &str,
        visited: &mut HashMap<String, usize>,
        low_link: &mut HashMap<String, usize>,
        stack: &mut Vec<String>,
        on_stack: &mut HashSet<String>,
        components: &mut Vec<Vec<String>>,
        index: &mut usize,
    ) {
        visited.insert(node_id.to_string(), *index);
        low_link.insert(node_id.to_string(), *index);
        stack.push(node_id.to_string());
        on_stack.insert(node_id.to_string());
        *index += 1;

        if let Some(neighbors) = graph.adjacency_list.get(node_id) {
            for neighbor_id in neighbors {
                if !visited.contains_key(neighbor_id) {
                    Self::tarjan_scc(
                        graph,
                        neighbor_id,
                        visited,
                        low_link,
                        stack,
                        on_stack,
                        components,
                        index,
                    );

                    let node_low = *low_link.get(node_id).unwrap();
                    let neighbor_low = *low_link.get(neighbor_id).unwrap();
                    low_link.insert(node_id.to_string(), node_low.min(neighbor_low));
                } else if on_stack.contains(neighbor_id) {
                    let node_low = *low_link.get(node_id).unwrap();
                    let neighbor_visited = *visited.get(neighbor_id).unwrap();
                    low_link.insert(node_id.to_string(), node_low.min(neighbor_visited));
                }
            }
        }

        // If node_id is a root node, pop the stack and create a component
        if low_link[node_id] == visited[node_id] {
            let mut component = Vec::new();
            loop {
                if let Some(w) = stack.pop() {
                    on_stack.remove(&w);
                    component.push(w.clone());
                    if w == node_id {
                        break;
                    }
                } else {
                    break;
                }
            }
            components.push(component);
        }
    }
}

/// PageRank algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankConfig {
    /// Damping factor (typically 0.85)
    pub damping_factor: f32,

    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence threshold
    pub convergence_threshold: f32,

    /// Personalization vector (optional)
    pub personalization: Option<HashMap<String, f32>>,

    /// Weight for personalization
    pub personalization_weight: f32,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            personalization: None,
            personalization_weight: 0.15,
        }
    }
}

/// Graph traversal configuration
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum traversal depth
    pub max_depth: usize,

    /// Maximum number of nodes to visit
    pub max_nodes: usize,

    /// Maximum distance for shortest path algorithms
    pub max_distance: f32,

    /// Cost function for edge traversal
    pub cost_function: CostFunction,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_nodes: 100,
            max_distance: f32::INFINITY,
            cost_function: CostFunction::Weight,
        }
    }
}

/// Path finding configuration
#[derive(Debug, Clone)]
pub struct PathFindingConfig {
    /// Maximum path depth
    pub max_depth: usize,

    /// Maximum number of paths to find
    pub max_paths: usize,

    /// Minimum semantic score threshold
    pub min_semantic_score: f32,

    /// Semantic weights for different edge types
    pub semantic_weights: HashMap<super::EdgeType, f32>,
}

impl Default for PathFindingConfig {
    fn default() -> Self {
        let mut semantic_weights = HashMap::new();
        semantic_weights.insert(super::EdgeType::Semantic("is_a".to_string()), 1.0);
        semantic_weights.insert(super::EdgeType::Semantic("part_of".to_string()), 0.9);
        semantic_weights.insert(super::EdgeType::Similar, 0.8);
        semantic_weights.insert(super::EdgeType::CoOccurs, 0.5);

        Self {
            max_depth: 4,
            max_paths: 10,
            min_semantic_score: 0.1,
            semantic_weights,
        }
    }
}

/// Edge cost functions
#[derive(Debug, Clone)]
pub enum CostFunction {
    /// Use edge weight (higher weight = lower cost)
    Weight,

    /// Use inverse confidence
    InverseConfidence,

    /// Uniform cost for all edges
    Uniform,
}

/// Path information from shortest path algorithm
#[derive(Debug, Clone)]
pub struct PathInfo {
    /// Total distance/cost
    pub distance: f32,

    /// Node IDs in the path
    pub path: Vec<String>,

    /// Number of hops
    pub hop_count: usize,
}

/// Semantic path between nodes
#[derive(Debug, Clone)]
pub struct SemanticPath {
    /// Node IDs in the path
    pub nodes: Vec<String>,

    /// Semantic score of the path
    pub semantic_score: f32,

    /// Path length (number of edges)
    pub path_length: usize,

    /// Edge types in the path
    pub edge_types: Vec<String>,
}

/// State for Dijkstra's algorithm
#[derive(Debug, Clone)]
struct DijkstraState {
    cost: f32,
    node_id: String,
}

impl Eq for DijkstraState {}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse the ordering
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_retrieval::{EdgeType, GraphEdge, GraphNode, NodeType};

    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        // Add nodes
        let node1 = GraphNode::new("Node1", NodeType::Concept);
        let node2 = GraphNode::new("Node2", NodeType::Concept);
        let node3 = GraphNode::new("Node3", NodeType::Concept);
        let node4 = GraphNode::new("Node4", NodeType::Concept);

        let node1_id = node1.id.clone();
        let node2_id = node2.id.clone();
        let node3_id = node3.id.clone();
        let node4_id = node4.id.clone();

        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();
        graph.add_node(node3).unwrap();
        graph.add_node(node4).unwrap();

        // Add edges: 1 -> 2 -> 3, 1 -> 4
        graph
            .add_edge(
                GraphEdge::new(
                    node1_id.clone(),
                    node2_id.clone(),
                    "edge1",
                    EdgeType::Similar,
                )
                .with_weight(0.8),
            )
            .unwrap();

        graph
            .add_edge(
                GraphEdge::new(
                    node2_id.clone(),
                    node3_id.clone(),
                    "edge2",
                    EdgeType::Similar,
                )
                .with_weight(0.6),
            )
            .unwrap();

        graph
            .add_edge(
                GraphEdge::new(
                    node1_id.clone(),
                    node4_id.clone(),
                    "edge3",
                    EdgeType::Similar,
                )
                .with_weight(0.9),
            )
            .unwrap();

        graph
    }

    #[test]
    fn test_pagerank() {
        let graph = create_test_graph();
        let config = PageRankConfig::default();

        let scores = GraphAlgorithms::pagerank(&graph, &config).unwrap();
        assert_eq!(scores.len(), 4);

        // All scores should be positive and sum to approximately 4.0 (number of nodes)
        let total: f32 = scores.values().sum();
        assert!((total - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_shortest_paths() {
        let graph = create_test_graph();
        let config = TraversalConfig::default();
        let node_ids: Vec<_> = graph.nodes.keys().cloned().collect();

        let paths = GraphAlgorithms::shortest_paths(&graph, &node_ids[0], &config).unwrap();

        // Should find paths to all reachable nodes
        assert!(!paths.is_empty());

        // Path to self should have distance 0
        assert_eq!(paths[&node_ids[0]].distance, 0.0);
        assert_eq!(paths[&node_ids[0]].hop_count, 0);
    }

    #[test]
    fn test_bfs_search() {
        let graph = create_test_graph();
        let config = TraversalConfig::default();
        let node_ids: Vec<_> = graph.nodes.keys().cloned().collect();

        let result = GraphAlgorithms::bfs_search(&graph, &node_ids[0], &config).unwrap();

        // Should visit at least the source node
        assert!(!result.is_empty());
        assert_eq!(result[0], node_ids[0]);
    }

    #[test]
    fn test_dfs_search() {
        let graph = create_test_graph();
        let config = TraversalConfig::default();
        let node_ids: Vec<_> = graph.nodes.keys().cloned().collect();

        let result = GraphAlgorithms::dfs_search(&graph, &node_ids[0], &config).unwrap();

        // Should visit at least the source node
        assert!(!result.is_empty());
        assert_eq!(result[0], node_ids[0]);
    }

    #[test]
    fn test_betweenness_centrality() {
        let graph = create_test_graph();
        let centrality = GraphAlgorithms::betweenness_centrality(&graph);

        assert_eq!(centrality.len(), 4);

        // All centrality scores should be non-negative
        for score in centrality.values() {
            assert!(*score >= 0.0);
        }
    }

    #[test]
    fn test_strongly_connected_components() {
        let graph = create_test_graph();
        let components = GraphAlgorithms::strongly_connected_components(&graph);

        // Should have at least one component
        assert!(!components.is_empty());

        // Total nodes in all components should equal graph node count
        let total_nodes: usize = components.iter().map(|c| c.len()).sum();
        assert_eq!(total_nodes, graph.nodes.len());
    }
}
