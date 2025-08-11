//! # Cache Eviction Policies
//!
//! Advanced eviction policies for intelligent cache management.

use super::CacheEntryMetadata;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::time::SystemTime;

/// Eviction policy trait
pub trait EvictionPolicyTrait: Send + Sync {
    /// Select entry to evict
    fn select_eviction<K, V>(&self, entries: &HashMap<K, V>) -> Option<K>
    where
        K: Clone + Eq + std::hash::Hash,
        V: HasMetadata;

    /// Update policy state after access
    fn on_access<K>(&mut self, key: &K)
    where
        K: Clone + Eq + std::hash::Hash;

    /// Update policy state after eviction
    fn on_evict<K>(&mut self, key: &K)
    where
        K: Clone + Eq + std::hash::Hash;

    /// Get policy name
    fn name(&self) -> &str;
}

/// Trait for types that have cache metadata
pub trait HasMetadata {
    fn metadata(&self) -> &CacheEntryMetadata;
    fn metadata_mut(&mut self) -> &mut CacheEntryMetadata;
}

/// Adaptive Replacement Cache (ARC) policy
pub struct ARCPolicy {
    /// Target size for recently used list
    pub p: f32,

    /// Maximum cache size
    pub max_size: usize,

    /// Recently used list
    pub t1: Vec<String>,

    /// Frequently used list
    pub t2: Vec<String>,

    /// Ghost list for T1
    pub b1: Vec<String>,

    /// Ghost list for T2
    pub b2: Vec<String>,
}

impl ARCPolicy {
    /// Create new ARC policy
    pub fn new(max_size: usize) -> Self {
        Self {
            p: max_size as f32 / 2.0,
            max_size,
            t1: Vec::new(),
            t2: Vec::new(),
            b1: Vec::new(),
            b2: Vec::new(),
        }
    }

    /// Adapt the target size based on hit patterns
    pub fn adapt(&mut self, hit_in_b1: bool, hit_in_b2: bool) {
        if hit_in_b1 {
            let delta = if self.b1.len() >= self.b2.len() {
                1.0
            } else {
                self.b2.len() as f32 / self.b1.len() as f32
            };
            self.p = (self.p + delta).min(self.max_size as f32);
        } else if hit_in_b2 {
            let delta = if self.b2.len() >= self.b1.len() {
                1.0
            } else {
                self.b1.len() as f32 / self.b2.len() as f32
            };
            self.p = (self.p - delta).max(0.0);
        }
    }
}

/// Semantic-aware eviction policy
pub struct SemanticEvictionPolicy {
    /// Semantic similarity threshold
    pub similarity_threshold: f32,

    /// Cluster importance scores
    pub cluster_scores: HashMap<usize, f32>,

    /// Query-to-cluster mapping
    pub query_clusters: HashMap<String, usize>,

    /// Access frequency per cluster
    pub cluster_access_freq: HashMap<usize, u64>,
}

impl SemanticEvictionPolicy {
    /// Create new semantic eviction policy
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
            cluster_scores: HashMap::new(),
            query_clusters: HashMap::new(),
            cluster_access_freq: HashMap::new(),
        }
    }

    /// Update cluster scores based on access patterns
    pub fn update_cluster_scores(&mut self) {
        for (cluster_id, freq) in &self.cluster_access_freq {
            let score = (*freq as f32).log2() + 1.0;
            self.cluster_scores.insert(*cluster_id, score);
        }
    }

    /// Get eviction score for entry
    pub fn get_eviction_score(&self, key: &str, metadata: &CacheEntryMetadata) -> f32 {
        let cluster_score = self
            .query_clusters
            .get(key)
            .and_then(|cid| self.cluster_scores.get(cid))
            .copied()
            .unwrap_or(1.0);

        let time_score = metadata
            .last_accessed
            .elapsed()
            .unwrap_or_default()
            .as_secs() as f32;

        let access_score = (metadata.access_count as f32).log2() + 1.0;

        // Lower score = more likely to evict
        access_score * cluster_score / (time_score + 1.0)
    }
}

/// Window-TinyLFU policy
pub struct WindowTinyLFU {
    /// Window size (percentage of cache)
    pub window_size_ratio: f32,

    /// Frequency sketch for admission
    pub frequency_sketch: FrequencySketch,

    /// Window entries (FIFO)
    pub window: Vec<String>,

    /// Main cache entries
    pub main_cache: Vec<String>,

    /// Maximum size
    pub max_size: usize,
}

impl WindowTinyLFU {
    /// Create new Window-TinyLFU policy
    pub fn new(max_size: usize, window_size_ratio: f32) -> Self {
        Self {
            window_size_ratio,
            frequency_sketch: FrequencySketch::new(max_size * 4),
            window: Vec::new(),
            main_cache: Vec::new(),
            max_size,
        }
    }

    /// Get window capacity
    pub fn window_capacity(&self) -> usize {
        (self.max_size as f32 * self.window_size_ratio) as usize
    }

    /// Get main cache capacity
    pub fn main_capacity(&self) -> usize {
        self.max_size - self.window_capacity()
    }

    /// Admit entry to main cache
    pub fn should_admit(&self, key: &str, victim_key: &str) -> bool {
        let key_freq = self.frequency_sketch.estimate(key);
        let victim_freq = self.frequency_sketch.estimate(victim_key);
        key_freq > victim_freq
    }
}

/// Count-Min Sketch for frequency estimation
pub struct FrequencySketch {
    /// Sketch table
    table: Vec<Vec<u32>>,

    /// Number of hash functions
    depth: usize,

    /// Width of each row
    width: usize,
}

impl FrequencySketch {
    /// Create new frequency sketch
    pub fn new(size: usize) -> Self {
        let width = size.next_power_of_two();
        let depth = 4; // Use 4 hash functions

        Self {
            table: vec![vec![0; width]; depth],
            depth,
            width,
        }
    }

    /// Increment frequency for key
    pub fn increment(&mut self, key: &str) {
        for i in 0..self.depth {
            let hash = self.hash(key, i);
            let idx = hash % self.width;
            self.table[i][idx] = self.table[i][idx].saturating_add(1);
        }
    }

    /// Estimate frequency for key
    pub fn estimate(&self, key: &str) -> u32 {
        (0..self.depth)
            .map(|i| {
                let hash = self.hash(key, i);
                let idx = hash % self.width;
                self.table[i][idx]
            })
            .min()
            .unwrap_or(0)
    }

    /// Hash function with seed
    fn hash(&self, key: &str, seed: usize) -> usize {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        seed.hash(&mut hasher);
        hasher.finish() as usize
    }
}

/// GDSF (Greedy-Dual Size Frequency) policy
pub struct GDSFPolicy {
    /// Priority queue of entries
    priorities: BinaryHeap<GDSFEntry>,

    /// Current clock value
    clock: f32,

    /// Size weight factor
    size_weight: f32,
}

#[derive(Clone)]
struct GDSFEntry {
    key: String,
    priority: f32,
    size: usize,
    frequency: u64,
}

impl GDSFPolicy {
    /// Create new GDSF policy
    pub fn new(size_weight: f32) -> Self {
        Self {
            priorities: BinaryHeap::new(),
            clock: 0.0,
            size_weight,
        }
    }

    /// Calculate priority for entry
    pub fn calculate_priority(&self, frequency: u64, size: usize, age: f32) -> f32 {
        let freq_factor = (frequency as f32).log2() + 1.0;
        let size_factor = 1.0 / (size as f32).powf(self.size_weight);
        let age_factor = 1.0 / (age + 1.0);

        freq_factor * size_factor * age_factor + self.clock
    }

    /// Update clock on eviction
    pub fn update_clock(&mut self, evicted_priority: f32) {
        self.clock = evicted_priority;
    }
}

impl Ord for GDSFEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority = evict first (min-heap behavior)
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for GDSFEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for GDSFEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for GDSFEntry {}

/// Machine learning-based eviction predictor
pub struct MLEvictionPolicy {
    /// Feature weights
    pub weights: FeatureWeights,

    /// Learning rate
    pub learning_rate: f32,

    /// Historical eviction outcomes
    pub history: Vec<EvictionOutcome>,
}

#[derive(Debug, Clone)]
pub struct FeatureWeights {
    pub recency_weight: f32,
    pub frequency_weight: f32,
    pub size_weight: f32,
    pub semantic_weight: f32,
    pub pattern_weight: f32,
}

#[derive(Debug, Clone)]
pub struct EvictionOutcome {
    pub key: String,
    pub features: Vec<f32>,
    pub was_good_eviction: bool,
    pub timestamp: SystemTime,
}

impl MLEvictionPolicy {
    /// Create new ML eviction policy
    pub fn new() -> Self {
        Self {
            weights: FeatureWeights {
                recency_weight: 1.0,
                frequency_weight: 1.0,
                size_weight: 1.0,
                semantic_weight: 1.0,
                pattern_weight: 1.0,
            },
            learning_rate: 0.01,
            history: Vec::new(),
        }
    }

    /// Extract features from cache entry
    pub fn extract_features(&self, metadata: &CacheEntryMetadata) -> Vec<f32> {
        vec![
            // Recency: time since last access
            metadata
                .last_accessed
                .elapsed()
                .unwrap_or_default()
                .as_secs() as f32,
            // Frequency: access count
            metadata.access_count as f32,
            // Size: entry size in bytes
            metadata.size_bytes as f32,
            // Age: time since creation
            metadata.created_at.elapsed().unwrap_or_default().as_secs() as f32,
            // TTL remaining
            metadata
                .ttl
                .and_then(|ttl| {
                    metadata
                        .created_at
                        .elapsed()
                        .ok()
                        .map(|elapsed| (ttl.as_secs() as f32) - (elapsed.as_secs() as f32))
                })
                .unwrap_or(0.0),
        ]
    }

    /// Predict eviction score
    pub fn predict_eviction_score(&self, features: &[f32]) -> f32 {
        let mut score = 0.0;

        if features.len() >= 5 {
            score += self.weights.recency_weight * (1.0 / (features[0] + 1.0));
            score += self.weights.frequency_weight * features[1].log2();
            score += self.weights.size_weight * (1.0 / (features[2] + 1.0));
            score += self.weights.pattern_weight * features[3];
            score += self.weights.semantic_weight * features[4];
        }

        score
    }

    /// Update weights based on outcome
    pub fn update_weights(&mut self, outcome: &EvictionOutcome) {
        // Simple gradient update
        let predicted = self.predict_eviction_score(&outcome.features);
        let target = if outcome.was_good_eviction { 1.0 } else { 0.0 };
        let error = target - predicted;

        // Update each weight
        if outcome.features.len() >= 5 {
            self.weights.recency_weight += self.learning_rate * error * outcome.features[0];
            self.weights.frequency_weight += self.learning_rate * error * outcome.features[1];
            self.weights.size_weight += self.learning_rate * error * outcome.features[2];
            self.weights.pattern_weight += self.learning_rate * error * outcome.features[3];
            self.weights.semantic_weight += self.learning_rate * error * outcome.features[4];
        }

        // Add to history
        self.history.push(outcome.clone());

        // Keep history bounded
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_policy() {
        let mut policy = ARCPolicy::new(100);

        assert_eq!(policy.p, 50.0);

        policy.adapt(true, false);
        assert!(policy.p > 50.0);

        policy.adapt(false, true);
        assert!(policy.p < 100.0);
    }

    #[test]
    fn test_frequency_sketch() {
        let mut sketch = FrequencySketch::new(1000);

        sketch.increment("key1");
        sketch.increment("key1");
        sketch.increment("key2");

        assert!(sketch.estimate("key1") >= 2);
        assert!(sketch.estimate("key2") >= 1);
        assert_eq!(sketch.estimate("key3"), 0);
    }

    #[test]
    fn test_window_tiny_lfu() {
        let policy = WindowTinyLFU::new(100, 0.1);

        assert_eq!(policy.window_capacity(), 10);
        assert_eq!(policy.main_capacity(), 90);
    }

    #[test]
    fn test_gdsf_priority() {
        let policy = GDSFPolicy::new(0.5);

        let priority1 = policy.calculate_priority(10, 100, 1.0);
        let priority2 = policy.calculate_priority(10, 1000, 1.0);

        // Smaller size should have higher priority
        assert!(priority1 > priority2);
    }

    #[test]
    fn test_ml_eviction_features() {
        let policy = MLEvictionPolicy::new();
        let mut metadata = CacheEntryMetadata::new();
        metadata.access_count = 10;
        metadata.size_bytes = 1024;

        let features = policy.extract_features(&metadata);
        assert_eq!(features.len(), 5);
        assert_eq!(features[1], 10.0); // frequency
        assert_eq!(features[2], 1024.0); // size
    }

    #[test]
    fn test_semantic_eviction_scoring() {
        let mut policy = SemanticEvictionPolicy::new(0.8);

        policy.cluster_scores.insert(1, 2.0);
        policy.query_clusters.insert("query1".to_string(), 1);

        let metadata = CacheEntryMetadata::new();
        let score = policy.get_eviction_score("query1", &metadata);

        assert!(score > 0.0);
    }
}
