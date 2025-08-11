//! # Common Evaluation Metrics
//! 
//! Shared metrics and utilities used across different evaluation modules.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Common statistical metrics
pub struct StatisticalMetrics;

impl StatisticalMetrics {
    /// Calculate mean of a vector
    pub fn mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }
    
    /// Calculate standard deviation
    pub fn std_dev(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = Self::mean(values);
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        variance.sqrt()
    }
    
    /// Calculate median
    pub fn median(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }
    
    /// Calculate percentile
    pub fn percentile(values: &[f32], p: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
    
    /// Calculate correlation coefficient
    pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        
        let mean_x = Self::mean(x);
        let mean_y = Self::mean(y);
        
        let numerator: f32 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        
        let sum_sq_x: f32 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f32 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Text similarity metrics
pub struct TextSimilarityMetrics;

impl TextSimilarityMetrics {
    /// Calculate Jaccard similarity
    pub fn jaccard_similarity(text1: &str, text2: &str) -> f32 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
    
    /// Calculate cosine similarity (simplified word-based)
    pub fn cosine_similarity(text1: &str, text2: &str) -> f32 {
        let words1: Vec<&str> = text1.split_whitespace().collect();
        let words2: Vec<&str> = text2.split_whitespace().collect();
        
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }
        
        // Create term frequency maps
        let mut tf1: HashMap<&str, f32> = HashMap::new();
        let mut tf2: HashMap<&str, f32> = HashMap::new();
        
        for word in &words1 {
            *tf1.entry(word).or_insert(0.0) += 1.0;
        }
        
        for word in &words2 {
            *tf2.entry(word).or_insert(0.0) += 1.0;
        }
        
        // Calculate dot product
        let mut dot_product = 0.0;
        for (word, freq1) in &tf1 {
            if let Some(freq2) = tf2.get(word) {
                dot_product += freq1 * freq2;
            }
        }
        
        // Calculate magnitudes
        let magnitude1: f32 = tf1.values().map(|f| f * f).sum::<f32>().sqrt();
        let magnitude2: f32 = tf2.values().map(|f| f * f).sum::<f32>().sqrt();
        
        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            0.0
        } else {
            dot_product / (magnitude1 * magnitude2)
        }
    }
    
    /// Calculate BLEU score (simplified)
    pub fn bleu_score(candidate: &str, reference: &str, n: usize) -> f32 {
        let candidate_words: Vec<&str> = candidate.split_whitespace().collect();
        let reference_words: Vec<&str> = reference.split_whitespace().collect();
        
        if candidate_words.len() < n || reference_words.len() < n {
            return 0.0;
        }
        
        // Generate n-grams
        let candidate_ngrams: Vec<Vec<&str>> = (0..=candidate_words.len() - n)
            .map(|i| candidate_words[i..i + n].to_vec())
            .collect();
        
        let reference_ngrams: Vec<Vec<&str>> = (0..=reference_words.len() - n)
            .map(|i| reference_words[i..i + n].to_vec())
            .collect();
        
        // Count matches
        let mut matches = 0;
        for candidate_ngram in &candidate_ngrams {
            if reference_ngrams.contains(candidate_ngram) {
                matches += 1;
            }
        }
        
        if candidate_ngrams.is_empty() {
            0.0
        } else {
            matches as f32 / candidate_ngrams.len() as f32
        }
    }
    
    /// Calculate ROUGE-L score (simplified)
    pub fn rouge_l_score(candidate: &str, reference: &str) -> f32 {
        let candidate_words: Vec<&str> = candidate.split_whitespace().collect();
        let reference_words: Vec<&str> = reference.split_whitespace().collect();
        
        // Find longest common subsequence
        let lcs_length = Self::lcs_length(&candidate_words, &reference_words);
        
        if candidate_words.is_empty() && reference_words.is_empty() {
            1.0
        } else if candidate_words.is_empty() || reference_words.is_empty() {
            0.0
        } else {
            let recall = lcs_length as f32 / reference_words.len() as f32;
            let precision = lcs_length as f32 / candidate_words.len() as f32;
            
            if recall + precision == 0.0 {
                0.0
            } else {
                2.0 * recall * precision / (recall + precision)
            }
        }
    }
    
    /// Calculate longest common subsequence length
    fn lcs_length(x: &[&str], y: &[&str]) -> usize {
        let m = x.len();
        let n = y.len();
        
        if m == 0 || n == 0 {
            return 0;
        }
        
        let mut dp = vec![vec![0; n + 1]; m + 1];
        
        for i in 1..=m {
            for j in 1..=n {
                if x[i - 1] == y[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        
        dp[m][n]
    }
    
    /// Calculate edit distance (Levenshtein)
    pub fn edit_distance(s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        
        let m = chars1.len();
        let n = chars2.len();
        
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }
        
        let mut dp = vec![vec![0; n + 1]; m + 1];
        
        // Initialize base cases
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }
        
        // Fill the DP table
        for i in 1..=m {
            for j in 1..=n {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                
                dp[i][j] = (dp[i - 1][j] + 1)          // deletion
                    .min(dp[i][j - 1] + 1)             // insertion
                    .min(dp[i - 1][j - 1] + cost);     // substitution
            }
        }
        
        dp[m][n]
    }
    
    /// Calculate normalized edit distance
    pub fn normalized_edit_distance(s1: &str, s2: &str) -> f32 {
        let distance = Self::edit_distance(s1, s2);
        let max_len = s1.len().max(s2.len());
        
        if max_len == 0 {
            0.0
        } else {
            distance as f32 / max_len as f32
        }
    }
}

/// Information Retrieval metrics
pub struct IRMetrics;

impl IRMetrics {
    /// Calculate precision at K
    pub fn precision_at_k(relevant_docs: &[bool], k: usize) -> f32 {
        if k == 0 {
            return 0.0;
        }
        
        let top_k = &relevant_docs[..k.min(relevant_docs.len())];
        let relevant_count = top_k.iter().filter(|&&r| r).count();
        
        relevant_count as f32 / top_k.len() as f32
    }
    
    /// Calculate recall at K
    pub fn recall_at_k(relevant_docs: &[bool], k: usize, total_relevant: usize) -> f32 {
        if total_relevant == 0 {
            return 1.0;
        }
        
        let top_k = &relevant_docs[..k.min(relevant_docs.len())];
        let retrieved_relevant = top_k.iter().filter(|&&r| r).count();
        
        retrieved_relevant as f32 / total_relevant as f32
    }
    
    /// Calculate F1 score at K
    pub fn f1_at_k(relevant_docs: &[bool], k: usize, total_relevant: usize) -> f32 {
        let precision = Self::precision_at_k(relevant_docs, k);
        let recall = Self::recall_at_k(relevant_docs, k, total_relevant);
        
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
    
    /// Calculate Average Precision
    pub fn average_precision(relevant_docs: &[bool]) -> f32 {
        let total_relevant = relevant_docs.iter().filter(|&&r| r).count();
        
        if total_relevant == 0 {
            return 0.0;
        }
        
        let mut sum_precision = 0.0;
        let mut relevant_count = 0;
        
        for (i, &is_relevant) in relevant_docs.iter().enumerate() {
            if is_relevant {
                relevant_count += 1;
                sum_precision += relevant_count as f32 / (i + 1) as f32;
            }
        }
        
        sum_precision / total_relevant as f32
    }
    
    /// Calculate Reciprocal Rank
    pub fn reciprocal_rank(relevant_docs: &[bool]) -> f32 {
        for (i, &is_relevant) in relevant_docs.iter().enumerate() {
            if is_relevant {
                return 1.0 / (i + 1) as f32;
            }
        }
        0.0
    }
    
    /// Calculate NDCG at K
    pub fn ndcg_at_k(relevance_scores: &[f32], k: usize) -> f32 {
        if k == 0 || relevance_scores.is_empty() {
            return 0.0;
        }
        
        let k = k.min(relevance_scores.len());
        
        // Calculate DCG
        let dcg = Self::dcg(&relevance_scores[..k]);
        
        // Calculate IDCG (Ideal DCG)
        let mut ideal_scores = relevance_scores.to_vec();
        ideal_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let idcg = Self::dcg(&ideal_scores[..k]);
        
        if idcg == 0.0 {
            0.0
        } else {
            dcg / idcg
        }
    }
    
    /// Calculate Discounted Cumulative Gain
    fn dcg(relevance_scores: &[f32]) -> f32 {
        relevance_scores.iter().enumerate()
            .map(|(i, &score)| score / (i as f32 + 2.0).log2())
            .sum()
    }
}

/// Quality assessment metrics
pub struct QualityMetrics;

impl QualityMetrics {
    /// Calculate perplexity (simplified)
    pub fn perplexity(text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return f32::INFINITY;
        }
        
        // Simplified perplexity calculation based on word frequency
        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for word in &words {
            *word_counts.entry(word).or_insert(0) += 1;
        }
        
        let mut log_prob_sum = 0.0;
        let vocab_size = word_counts.len() as f32;
        
        for count in word_counts.values() {
            let prob = *count as f32 / words.len() as f32;
            log_prob_sum += prob * prob.ln();
        }
        
        // Add smoothing
        let avg_log_prob = log_prob_sum / vocab_size;
        (-avg_log_prob).exp()
    }
    
    /// Calculate readability score (simplified Flesch)
    pub fn readability_score(text: &str) -> f32 {
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        let words: Vec<&str> = text.split_whitespace().collect();
        let syllables = Self::count_syllables(text);
        
        if sentences.is_empty() || words.is_empty() {
            return 0.0;
        }
        
        let avg_sentence_length = words.len() as f32 / sentences.len() as f32;
        let avg_syllables_per_word = syllables as f32 / words.len() as f32;
        
        // Simplified Flesch Reading Ease
        let score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word;
        score.max(0.0).min(100.0)
    }
    
    /// Count syllables in text (simplified)
    fn count_syllables(text: &str) -> usize {
        let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
        let mut syllable_count = 0;
        
        for word in text.split_whitespace() {
            let mut word_syllables = 0;
            let mut previous_was_vowel = false;
            
            for ch in word.to_lowercase().chars() {
                if vowels.contains(&ch) {
                    if !previous_was_vowel {
                        word_syllables += 1;
                    }
                    previous_was_vowel = true;
                } else {
                    previous_was_vowel = false;
                }
            }
            
            // Every word has at least one syllable
            if word_syllables == 0 {
                word_syllables = 1;
            }
            
            syllable_count += word_syllables;
        }
        
        syllable_count
    }
    
    /// Calculate lexical diversity (Type-Token Ratio)
    pub fn lexical_diversity(text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        unique_words.len() as f32 / words.len() as f32
    }
    
    /// Calculate semantic coherence (simplified)
    pub fn semantic_coherence(sentences: &[&str]) -> f32 {
        if sentences.len() < 2 {
            return 1.0;
        }
        
        let mut coherence_scores = Vec::new();
        
        for i in 0..sentences.len() - 1 {
            let similarity = TextSimilarityMetrics::jaccard_similarity(sentences[i], sentences[i + 1]);
            coherence_scores.push(similarity);
        }
        
        StatisticalMetrics::mean(&coherence_scores)
    }
}

/// Specialized evaluation metrics for different domains
pub struct DomainMetrics;

impl DomainMetrics {
    /// Calculate factual accuracy (simplified)
    pub fn factual_accuracy(generated_text: &str, reference_facts: &[&str]) -> f32 {
        if reference_facts.is_empty() {
            return 1.0; // No facts to check against
        }
        
        let generated_lower = generated_text.to_lowercase();
        let mut supported_facts = 0;
        
        for fact in reference_facts {
            // Very simplified fact checking - look for key terms
            let fact_words: Vec<&str> = fact.split_whitespace().collect();
            let fact_words_len = fact_words.len();
            let mut word_matches = 0;
            
            for word in &fact_words {
                if generated_lower.contains(&word.to_lowercase()) {
                    word_matches += 1;
                }
            }
            
            // Consider fact supported if most words are present
            if word_matches as f32 / fact_words_len as f32 > 0.7 {
                supported_facts += 1;
            }
        }
        
        supported_facts as f32 / reference_facts.len() as f32
    }
    
    /// Calculate bias score (simplified)
    pub fn bias_score(text: &str) -> f32 {
        // Simplified bias detection based on certain terms
        let biased_terms = [
            "always", "never", "all", "none", "everyone", "nobody",
            "obviously", "clearly", "definitely", "certainly"
        ];
        
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        
        if words.is_empty() {
            return 0.0;
        }
        
        let biased_count = words.iter()
            .filter(|word| biased_terms.iter().any(|term| word.contains(term)))
            .count();
        
        biased_count as f32 / words.len() as f32
    }
    
    /// Calculate toxicity score (simplified)
    pub fn toxicity_score(text: &str) -> f32 {
        // Very simplified toxicity detection
        let toxic_patterns = ["hate", "stupid", "idiot", "kill", "die"];
        
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        
        if words.is_empty() {
            return 0.0;
        }
        
        let toxic_count = words.iter()
            .filter(|word| toxic_patterns.iter().any(|pattern| word.contains(pattern)))
            .count();
        
        toxic_count as f32 / words.len() as f32
    }
}

/// Metric aggregation utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricAggregator {
    /// Collected metrics
    pub metrics: HashMap<String, Vec<f32>>,
}

impl MetricAggregator {
    /// Create new metric aggregator
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
    
    /// Add a metric value
    pub fn add_metric(&mut self, name: &str, value: f32) {
        self.metrics.entry(name.to_string()).or_insert_with(Vec::new).push(value);
    }
    
    /// Get summary statistics for all metrics
    pub fn get_summary(&self) -> HashMap<String, MetricSummary> {
        let mut summaries = HashMap::new();
        
        for (name, values) in &self.metrics {
            let summary = MetricSummary {
                count: values.len(),
                mean: StatisticalMetrics::mean(values),
                std_dev: StatisticalMetrics::std_dev(values),
                median: StatisticalMetrics::median(values),
                min: values.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                max: values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                percentile_25: StatisticalMetrics::percentile(values, 25.0),
                percentile_75: StatisticalMetrics::percentile(values, 75.0),
                percentile_95: StatisticalMetrics::percentile(values, 95.0),
            };
            
            summaries.insert(name.clone(), summary);
        }
        
        summaries
    }
    
    /// Calculate confidence interval
    pub fn confidence_interval(&self, metric_name: &str, confidence_level: f32) -> Option<(f32, f32)> {
        if let Some(values) = self.metrics.get(metric_name) {
            if values.len() < 2 {
                return None;
            }
            
            let mean = StatisticalMetrics::mean(values);
            let std_dev = StatisticalMetrics::std_dev(values);
            let n = values.len() as f32;
            
            // Simplified confidence interval (assuming normal distribution)
            let z_score = match confidence_level {
                0.90 => 1.645,
                0.95 => 1.96,
                0.99 => 2.576,
                _ => 1.96, // Default to 95%
            };
            
            let margin_of_error = z_score * std_dev / n.sqrt();
            Some((mean - margin_of_error, mean + margin_of_error))
        } else {
            None
        }
    }
}

impl Default for MetricAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Number of observations
    pub count: usize,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Median value
    pub median: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// 25th percentile
    pub percentile_25: f32,
    /// 75th percentile
    pub percentile_75: f32,
    /// 95th percentile
    pub percentile_95: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_statistical_metrics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(StatisticalMetrics::mean(&values), 3.0);
        assert_eq!(StatisticalMetrics::median(&values), 3.0);
        assert!((StatisticalMetrics::std_dev(&values) - 1.5811).abs() < 0.01);
    }
    
    #[test]
    fn test_text_similarity() {
        let text1 = "the quick brown fox";
        let text2 = "the quick brown dog";
        
        let jaccard = TextSimilarityMetrics::jaccard_similarity(text1, text2);
        assert!(jaccard > 0.5); // Should be similar
        
        let cosine = TextSimilarityMetrics::cosine_similarity(text1, text2);
        assert!(cosine > 0.5); // Should be similar
    }
    
    #[test]
    fn test_ir_metrics() {
        let relevant_docs = vec![true, false, true, false, true];
        
        assert_eq!(IRMetrics::precision_at_k(&relevant_docs, 3), 2.0 / 3.0);
        assert_eq!(IRMetrics::recall_at_k(&relevant_docs, 3, 3), 2.0 / 3.0);
        assert_eq!(IRMetrics::reciprocal_rank(&relevant_docs), 1.0);
    }
    
    #[test]
    fn test_metric_aggregator() {
        let mut aggregator = MetricAggregator::new();
        
        aggregator.add_metric("precision", 0.8);
        aggregator.add_metric("precision", 0.9);
        aggregator.add_metric("recall", 0.7);
        
        let summary = aggregator.get_summary();
        
        assert_eq!(summary["precision"].count, 2);
        assert_eq!(summary["precision"].mean, 0.85);
        assert_eq!(summary["recall"].count, 1);
    }
    
    #[test]
    fn test_quality_metrics() {
        let text = "This is a simple test sentence.";
        
        let diversity = QualityMetrics::lexical_diversity(text);
        assert!(diversity > 0.8); // Most words are unique
        
        let readability = QualityMetrics::readability_score(text);
        assert!(readability > 0.0); // Should have some readability score
    }
}