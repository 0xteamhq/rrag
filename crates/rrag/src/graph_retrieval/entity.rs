//! # Entity and Relationship Extraction
//! 
//! Advanced entity recognition and relationship extraction for knowledge graph construction.

use crate::{RragResult, Document, DocumentChunk, Embedding};
use super::{GraphNode, GraphEdge, NodeType, EdgeType, GraphError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use regex::Regex;
use async_trait::async_trait;

/// Entity extracted from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text/mention
    pub text: String,
    
    /// Entity type
    pub entity_type: EntityType,
    
    /// Start position in source text
    pub start_pos: usize,
    
    /// End position in source text
    pub end_pos: usize,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    
    /// Normalized form of the entity
    pub normalized_form: Option<String>,
    
    /// Additional attributes
    pub attributes: HashMap<String, serde_json::Value>,
    
    /// Source document/chunk ID
    pub source_id: String,
}

/// Relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source entity
    pub source_entity: String,
    
    /// Target entity
    pub target_entity: String,
    
    /// Relationship type
    pub relation_type: RelationType,
    
    /// Relationship text/context
    pub context: String,
    
    /// Start position in source text
    pub start_pos: usize,
    
    /// End position in source text
    pub end_pos: usize,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    
    /// Additional attributes
    pub attributes: HashMap<String, serde_json::Value>,
    
    /// Source document/chunk ID
    pub source_id: String,
}

/// Entity types for classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// Person names
    Person,
    
    /// Organization names
    Organization,
    
    /// Locations (cities, countries, etc.)
    Location,
    
    /// Dates and times
    DateTime,
    
    /// Monetary values
    Money,
    
    /// Percentages
    Percentage,
    
    /// Technical terms
    Technical,
    
    /// Concepts
    Concept,
    
    /// Products or services
    Product,
    
    /// Events
    Event,
    
    /// Custom entity type
    Custom(String),
}

/// Relationship types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RelationType {
    /// "is a" relationship
    IsA,
    
    /// "part of" relationship
    PartOf,
    
    /// "located in" relationship
    LocatedIn,
    
    /// "works for" relationship
    WorksFor,
    
    /// "owns" relationship
    Owns,
    
    /// "causes" relationship
    Causes,
    
    /// "similar to" relationship
    SimilarTo,
    
    /// "happened on" relationship
    HappenedOn,
    
    /// "mentioned with" relationship
    MentionedWith,
    
    /// Custom relationship type
    Custom(String),
}

/// Entity extraction configuration
#[derive(Debug, Clone)]
pub struct EntityExtractionConfig {
    /// Minimum confidence threshold
    pub min_confidence: f32,
    
    /// Maximum entity length in characters
    pub max_entity_length: usize,
    
    /// Whether to extract technical terms
    pub extract_technical_terms: bool,
    
    /// Whether to extract concepts
    pub extract_concepts: bool,
    
    /// Custom entity patterns
    pub custom_patterns: HashMap<String, Regex>,
    
    /// Stop words to ignore
    pub stop_words: HashSet<String>,
    
    /// Entity type priorities (higher = more important)
    pub entity_priorities: HashMap<EntityType, f32>,
}

impl Default for EntityExtractionConfig {
    fn default() -> Self {
        let mut stop_words = HashSet::new();
        stop_words.extend(vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "this", "that",
            "these", "those", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
            "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "whose", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "would", "should",
            "could", "can", "may", "might", "must", "shall", "will", "would"
        ].into_iter().map(|s| s.to_string()));
        
        let mut entity_priorities = HashMap::new();
        entity_priorities.insert(EntityType::Person, 0.9);
        entity_priorities.insert(EntityType::Organization, 0.8);
        entity_priorities.insert(EntityType::Location, 0.8);
        entity_priorities.insert(EntityType::DateTime, 0.7);
        entity_priorities.insert(EntityType::Technical, 0.6);
        entity_priorities.insert(EntityType::Concept, 0.5);
        
        Self {
            min_confidence: 0.5,
            max_entity_length: 100,
            extract_technical_terms: true,
            extract_concepts: true,
            custom_patterns: HashMap::new(),
            stop_words,
            entity_priorities,
        }
    }
}

/// Entity extractor trait
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Extract entities from text
    async fn extract_entities(&self, text: &str, source_id: &str) -> RragResult<Vec<Entity>>;
    
    /// Extract relationships from text and entities
    async fn extract_relationships(
        &self,
        text: &str,
        entities: &[Entity],
        source_id: &str,
    ) -> RragResult<Vec<Relationship>>;
    
    /// Extract both entities and relationships
    async fn extract_all(
        &self,
        text: &str,
        source_id: &str,
    ) -> RragResult<(Vec<Entity>, Vec<Relationship>)> {
        let entities = self.extract_entities(text, source_id).await?;
        let relationships = self.extract_relationships(text, &entities, source_id).await?;
        Ok((entities, relationships))
    }
}

/// Rule-based entity extractor
pub struct RuleBasedEntityExtractor {
    /// Configuration
    config: EntityExtractionConfig,
    
    /// Compiled regex patterns
    patterns: HashMap<EntityType, Vec<Regex>>,
    
    /// Relationship patterns
    relationship_patterns: HashMap<RelationType, Vec<Regex>>,
}

impl RuleBasedEntityExtractor {
    /// Create a new rule-based entity extractor
    pub fn new(config: EntityExtractionConfig) -> RragResult<Self> {
        let patterns = Self::compile_entity_patterns(&config)?;
        let relationship_patterns = Self::compile_relationship_patterns()?;
        
        Ok(Self {
            config,
            patterns,
            relationship_patterns,
        })
    }

    /// Compile entity recognition patterns
    fn compile_entity_patterns(config: &EntityExtractionConfig) -> RragResult<HashMap<EntityType, Vec<Regex>>> {
        let mut patterns = HashMap::new();
        
        // Person patterns
        let person_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile person pattern: {}", e)
                }
            })?,
            Regex::new(r"\b(?:Mr|Mrs|Dr|Prof|Ms)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile person title pattern: {}", e)
                }
            })?,
        ];
        patterns.insert(EntityType::Person, person_patterns);
        
        // Organization patterns
        let org_patterns = vec![
            Regex::new(r"\b[A-Z][a-zA-Z]*\s+(?:Inc|Corp|Company|Ltd|LLC|Organization|Institute|University|College|School)\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile organization pattern: {}", e)
                }
            })?,
            Regex::new(r"\b(?:the\s+)?[A-Z][a-zA-Z\s]+(?:Corporation|Foundation|Association|Agency|Department)\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile organization pattern 2: {}", e)
                }
            })?,
        ];
        patterns.insert(EntityType::Organization, org_patterns);
        
        // Location patterns
        let location_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile location pattern: {}", e)
                }
            })?,
            Regex::new(r"\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|San Francisco|Indianapolis|Seattle|Denver|Washington|Boston|El Paso|Detroit|Nashville|Portland|Memphis|Oklahoma City|Las Vegas|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Sacramento|Mesa|Kansas City|Atlanta|Long Beach|Colorado Springs|Raleigh|Miami|Virginia Beach|Omaha|Oakland|Minneapolis|Tulsa|Arlington|Tampa|New Orleans|Wichita|Cleveland|Bakersfield|Aurora|Anaheim|Honolulu|Santa Ana|Riverside|Corpus Christi|Lexington|Stockton|Henderson|Saint Paul|St. Paul|Cincinnati|St. Louis|Pittsburgh|Greensboro|Lincoln|Plano|Anchorage|Durham|Jersey City|Chula Vista|Orlando|Chandler|Henderson|Laredo|Buffalo|North Las Vegas|Madison|Lubbock|Reno|Akron|Hialeah|Garland|Rochester|Modesto|Montgomery|Yonkers|Spokane|Tacoma|Shreveport|Des Moines|Fremont|Baton Rouge|Richmond|Birmingham|Chesapeake|Glendale|Irving|Scottsdale|North Hempstead|Fayetteville|Grand Rapids|Santa Clarita|Salt Lake City|Huntsville)\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile major cities pattern: {}", e)
                }
            })?,
        ];
        patterns.insert(EntityType::Location, location_patterns);
        
        // DateTime patterns
        let datetime_patterns = vec![
            Regex::new(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile date pattern: {}", e)
                }
            })?,
            Regex::new(r"\b\d{1,2}/\d{1,2}/\d{4}\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile date pattern 2: {}", e)
                }
            })?,
            Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile ISO date pattern: {}", e)
                }
            })?,
        ];
        patterns.insert(EntityType::DateTime, datetime_patterns);
        
        // Money patterns
        let money_patterns = vec![
            Regex::new(r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile money pattern: {}", e)
                }
            })?,
            Regex::new(r"\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|cents?)\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile money pattern 2: {}", e)
                }
            })?,
        ];
        patterns.insert(EntityType::Money, money_patterns);
        
        // Percentage patterns
        let percentage_patterns = vec![
            Regex::new(r"\b\d+(?:\.\d+)?%\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile percentage pattern: {}", e)
                }
            })?,
            Regex::new(r"\b\d+(?:\.\d+)?\s*percent\b").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile percentage pattern 2: {}", e)
                }
            })?,
        ];
        patterns.insert(EntityType::Percentage, percentage_patterns);
        
        // Technical terms (if enabled)
        if config.extract_technical_terms {
            let technical_patterns = vec![
                Regex::new(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b").map_err(|e| {
                    GraphError::EntityExtraction {
                        message: format!("Failed to compile technical acronym pattern: {}", e)
                    }
                })?,
                Regex::new(r"\b(?:API|SDK|HTTP|HTTPS|JSON|XML|SQL|NoSQL|REST|GraphQL|OAuth|JWT|SSL|TLS|CSS|HTML|JavaScript|TypeScript|Python|Java|Rust|Go|C\+\+|PHP|Ruby|Swift|Kotlin|React|Vue|Angular|Docker|Kubernetes|AWS|GCP|Azure|MongoDB|PostgreSQL|MySQL|Redis|Elasticsearch|TensorFlow|PyTorch|OpenAI|GPT|BERT|Transformer|Neural Network|Machine Learning|Deep Learning|AI|ML|DL|NLP|Computer Vision|Data Science|Big Data|Cloud Computing|DevOps|CI/CD|Git|GitHub|GitLab|Bitbucket|Jenkins|Travis CI|CircleCI|Terraform|Ansible|Chef|Puppet)\b").map_err(|e| {
                    GraphError::EntityExtraction {
                        message: format!("Failed to compile technical terms pattern: {}", e)
                    }
                })?,
            ];
            patterns.insert(EntityType::Technical, technical_patterns);
        }
        
        // Add custom patterns
        for (pattern_name, regex) in &config.custom_patterns {
            let entity_type = EntityType::Custom(pattern_name.clone());
            patterns.entry(entity_type).or_insert_with(Vec::new).push(regex.clone());
        }
        
        Ok(patterns)
    }

    /// Compile relationship patterns
    fn compile_relationship_patterns() -> RragResult<HashMap<RelationType, Vec<Regex>>> {
        let mut patterns = HashMap::new();
        
        // "is a" relationships
        let is_a_patterns = vec![
            Regex::new(r"(.+?)\s+is\s+a\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile is-a pattern: {}", e)
                }
            })?,
            Regex::new(r"(.+?)\s+(?:are|is)\s+(?:an?|the)\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile is-a pattern 2: {}", e)
                }
            })?,
        ];
        patterns.insert(RelationType::IsA, is_a_patterns);
        
        // "part of" relationships
        let part_of_patterns = vec![
            Regex::new(r"(.+?)\s+(?:is|are)\s+part\s+of\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile part-of pattern: {}", e)
                }
            })?,
            Regex::new(r"(.+?)\s+belongs\s+to\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile belongs-to pattern: {}", e)
                }
            })?,
        ];
        patterns.insert(RelationType::PartOf, part_of_patterns);
        
        // "located in" relationships
        let located_in_patterns = vec![
            Regex::new(r"(.+?)\s+(?:is|are)\s+located\s+in\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile located-in pattern: {}", e)
                }
            })?,
            Regex::new(r"(.+?)\s+in\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile in pattern: {}", e)
                }
            })?,
        ];
        patterns.insert(RelationType::LocatedIn, located_in_patterns);
        
        // "works for" relationships
        let works_for_patterns = vec![
            Regex::new(r"(.+?)\s+works\s+(?:for|at)\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile works-for pattern: {}", e)
                }
            })?,
            Regex::new(r"(.+?)\s+(?:is|was)\s+employed\s+(?:by|at)\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile employed-by pattern: {}", e)
                }
            })?,
        ];
        patterns.insert(RelationType::WorksFor, works_for_patterns);
        
        // "owns" relationships
        let owns_patterns = vec![
            Regex::new(r"(.+?)\s+owns\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile owns pattern: {}", e)
                }
            })?,
            Regex::new(r"(.+?)\s+(?:has|possesses)\s+(.+?)").map_err(|e| {
                GraphError::EntityExtraction {
                    message: format!("Failed to compile has pattern: {}", e)
                }
            })?,
        ];
        patterns.insert(RelationType::Owns, owns_patterns);
        
        Ok(patterns)
    }

    /// Extract entities using pattern matching
    fn extract_entities_with_patterns(&self, text: &str, source_id: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let mut seen_positions = HashSet::new();
        
        for (entity_type, patterns) in &self.patterns {
            let priority = self.config.entity_priorities.get(entity_type).copied().unwrap_or(0.5);
            
            for pattern in patterns {
                for mat in pattern.find_iter(text) {
                    let start_pos = mat.start();
                    let end_pos = mat.end();
                    let entity_text = mat.as_str().trim();
                    
                    // Skip if we've already found an entity at this position
                    if seen_positions.contains(&(start_pos, end_pos)) {
                        continue;
                    }
                    
                    // Skip if it's too long or contains only stop words
                    if entity_text.len() > self.config.max_entity_length ||
                       self.is_stop_word_only(entity_text) {
                        continue;
                    }
                    
                    // Calculate confidence based on pattern match and entity type priority
                    let base_confidence = match entity_type {
                        EntityType::DateTime | EntityType::Money | EntityType::Percentage => 0.9,
                        EntityType::Technical => 0.8,
                        _ => 0.7,
                    };
                    let confidence = (base_confidence * priority).min(1.0);
                    
                    if confidence >= self.config.min_confidence {
                        let entity = Entity {
                            text: entity_text.to_string(),
                            entity_type: entity_type.clone(),
                            start_pos,
                            end_pos,
                            confidence,
                            normalized_form: Some(self.normalize_entity(entity_text)),
                            attributes: HashMap::new(),
                            source_id: source_id.to_string(),
                        };
                        
                        entities.push(entity);
                        seen_positions.insert((start_pos, end_pos));
                    }
                }
            }
        }
        
        // Sort by position for consistent ordering
        entities.sort_by_key(|e| e.start_pos);
        entities
    }

    /// Check if text contains only stop words
    fn is_stop_word_only(&self, text: &str) -> bool {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return true;
        }
        
        words.iter().all(|word| {
            self.config.stop_words.contains(&word.to_lowercase())
        })
    }

    /// Normalize entity text
    fn normalize_entity(&self, text: &str) -> String {
        text.trim()
            .chars()
            .map(|c| if c.is_whitespace() { ' ' } else { c })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract relationships using pattern matching
    fn extract_relationships_with_patterns(
        &self,
        text: &str,
        entities: &[Entity],
        source_id: &str,
    ) -> Vec<Relationship> {
        let mut relationships = Vec::new();
        
        // Create entity lookup by position
        let mut entity_spans: Vec<(usize, usize, &Entity)> = entities.iter()
            .map(|e| (e.start_pos, e.end_pos, e))
            .collect();
        entity_spans.sort_by_key(|&(start, _, _)| start);
        
        for (relation_type, patterns) in &self.relationship_patterns {
            for pattern in patterns {
                for mat in pattern.find_iter(text) {
                    if let Some(captures) = pattern.captures(mat.as_str()) {
                        if captures.len() >= 3 {
                            let source_text = captures.get(1).unwrap().as_str().trim();
                            let target_text = captures.get(2).unwrap().as_str().trim();
                            
                            // Find entities that match the captured groups
                            if let (Some(source_entity), Some(target_entity)) = (
                                self.find_matching_entity(source_text, &entity_spans),
                                self.find_matching_entity(target_text, &entity_spans)
                            ) {
                                let relationship = Relationship {
                                    source_entity: source_entity.normalized_form
                                        .as_ref()
                                        .unwrap_or(&source_entity.text)
                                        .clone(),
                                    target_entity: target_entity.normalized_form
                                        .as_ref()
                                        .unwrap_or(&target_entity.text)
                                        .clone(),
                                    relation_type: relation_type.clone(),
                                    context: mat.as_str().to_string(),
                                    start_pos: mat.start(),
                                    end_pos: mat.end(),
                                    confidence: 0.7, // Base confidence for pattern-matched relationships
                                    attributes: HashMap::new(),
                                    source_id: source_id.to_string(),
                                };
                                
                                relationships.push(relationship);
                            }
                        }
                    }
                }
            }
        }
        
        // Add co-occurrence relationships for entities that appear close together
        self.extract_co_occurrence_relationships(text, entities, source_id, &mut relationships);
        
        relationships
    }

    /// Find entity that matches the given text
    fn find_matching_entity<'a>(
        &self,
        text: &str,
        entity_spans: &'a [(usize, usize, &'a Entity)]
    ) -> Option<&'a Entity> {
        entity_spans.iter()
            .find(|(_, _, entity)| {
                entity.text.eq_ignore_ascii_case(text) ||
                entity.normalized_form.as_ref().map_or(false, |norm| norm.eq_ignore_ascii_case(text))
            })
            .map(|(_, _, entity)| *entity)
    }

    /// Extract co-occurrence relationships
    fn extract_co_occurrence_relationships(
        &self,
        text: &str,
        entities: &[Entity],
        source_id: &str,
        relationships: &mut Vec<Relationship>
    ) {
        let max_distance = 100; // Maximum character distance for co-occurrence
        
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity1 = &entities[i];
                let entity2 = &entities[j];
                
                // Check if entities are close enough
                let distance = if entity1.end_pos < entity2.start_pos {
                    entity2.start_pos - entity1.end_pos
                } else if entity2.end_pos < entity1.start_pos {
                    entity1.start_pos - entity2.end_pos
                } else {
                    0 // Overlapping
                };
                
                if distance <= max_distance {
                    // Calculate confidence based on distance and entity types
                    let base_confidence = 0.3;
                    let distance_factor = 1.0 - (distance as f32 / max_distance as f32);
                    let confidence = base_confidence * distance_factor;
                    
                    if confidence >= self.config.min_confidence {
                        let relationship = Relationship {
                            source_entity: entity1.normalized_form
                                .as_ref()
                                .unwrap_or(&entity1.text)
                                .clone(),
                            target_entity: entity2.normalized_form
                                .as_ref()
                                .unwrap_or(&entity2.text)
                                .clone(),
                            relation_type: RelationType::MentionedWith,
                            context: format!(
                                "Co-occurrence within {} characters",
                                distance
                            ),
                            start_pos: entity1.start_pos.min(entity2.start_pos),
                            end_pos: entity1.end_pos.max(entity2.end_pos),
                            confidence,
                            attributes: {
                                let mut attrs = HashMap::new();
                                attrs.insert("distance".to_string(), serde_json::Value::Number(distance.into()));
                                attrs.insert("type".to_string(), serde_json::Value::String("co_occurrence".to_string()));
                                attrs
                            },
                            source_id: source_id.to_string(),
                        };
                        
                        relationships.push(relationship);
                    }
                }
            }
        }
    }
}

#[async_trait]
impl EntityExtractor for RuleBasedEntityExtractor {
    async fn extract_entities(&self, text: &str, source_id: &str) -> RragResult<Vec<Entity>> {
        Ok(self.extract_entities_with_patterns(text, source_id))
    }

    async fn extract_relationships(
        &self,
        text: &str,
        entities: &[Entity],
        source_id: &str,
    ) -> RragResult<Vec<Relationship>> {
        Ok(self.extract_relationships_with_patterns(text, entities, source_id))
    }
}

/// Convert entities to graph nodes
pub fn entities_to_nodes(entities: &[Entity]) -> Vec<GraphNode> {
    entities.iter().map(|entity| {
        let node_type = match &entity.entity_type {
            EntityType::Person => NodeType::Entity("Person".to_string()),
            EntityType::Organization => NodeType::Entity("Organization".to_string()),
            EntityType::Location => NodeType::Entity("Location".to_string()),
            EntityType::DateTime => NodeType::Entity("DateTime".to_string()),
            EntityType::Money => NodeType::Entity("Money".to_string()),
            EntityType::Percentage => NodeType::Entity("Percentage".to_string()),
            EntityType::Technical => NodeType::Entity("Technical".to_string()),
            EntityType::Concept => NodeType::Concept,
            EntityType::Product => NodeType::Entity("Product".to_string()),
            EntityType::Event => NodeType::Entity("Event".to_string()),
            EntityType::Custom(custom_type) => NodeType::Custom(custom_type.clone()),
        };
        
        let mut node = GraphNode::new(
            entity.normalized_form.as_ref().unwrap_or(&entity.text),
            node_type,
        )
        .with_confidence(entity.confidence)
        .with_source_document(entity.source_id.clone());
        
        // Add entity attributes
        for (key, value) in &entity.attributes {
            node = node.with_attribute(key, value.clone());
        }
        
        node = node.with_attribute("original_text", serde_json::Value::String(entity.text.clone()));
        node = node.with_attribute("start_pos", serde_json::Value::Number(entity.start_pos.into()));
        node = node.with_attribute("end_pos", serde_json::Value::Number(entity.end_pos.into()));
        
        node
    }).collect()
}

/// Convert relationships to graph edges
pub fn relationships_to_edges(relationships: &[Relationship], entity_node_map: &HashMap<String, String>) -> Vec<GraphEdge> {
    relationships.iter().filter_map(|relationship| {
        // Find node IDs for source and target entities
        let source_node_id = entity_node_map.get(&relationship.source_entity)?;
        let target_node_id = entity_node_map.get(&relationship.target_entity)?;
        
        let edge_type = match &relationship.relation_type {
            RelationType::IsA => EdgeType::Semantic("is_a".to_string()),
            RelationType::PartOf => EdgeType::Semantic("part_of".to_string()),
            RelationType::LocatedIn => EdgeType::Semantic("located_in".to_string()),
            RelationType::WorksFor => EdgeType::Semantic("works_for".to_string()),
            RelationType::Owns => EdgeType::Semantic("owns".to_string()),
            RelationType::Causes => EdgeType::Semantic("causes".to_string()),
            RelationType::SimilarTo => EdgeType::Similar,
            RelationType::HappenedOn => EdgeType::Semantic("happened_on".to_string()),
            RelationType::MentionedWith => EdgeType::CoOccurs,
            RelationType::Custom(custom_type) => EdgeType::Custom(custom_type.clone()),
        };
        
        let mut edge = GraphEdge::new(
            source_node_id,
            target_node_id,
            &relationship.relation_type.to_string(),
            edge_type,
        )
        .with_confidence(relationship.confidence)
        .with_weight(relationship.confidence)
        .with_source_document(relationship.source_id.clone());
        
        // Add relationship attributes
        for (key, value) in &relationship.attributes {
            edge = edge.with_attribute(key, value.clone());
        }
        
        edge = edge.with_attribute("context", serde_json::Value::String(relationship.context.clone()));
        edge = edge.with_attribute("start_pos", serde_json::Value::Number(relationship.start_pos.into()));
        edge = edge.with_attribute("end_pos", serde_json::Value::Number(relationship.end_pos.into()));
        
        Some(edge)
    }).collect()
}

impl std::fmt::Display for RelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelationType::IsA => write!(f, "is_a"),
            RelationType::PartOf => write!(f, "part_of"),
            RelationType::LocatedIn => write!(f, "located_in"),
            RelationType::WorksFor => write!(f, "works_for"),
            RelationType::Owns => write!(f, "owns"),
            RelationType::Causes => write!(f, "causes"),
            RelationType::SimilarTo => write!(f, "similar_to"),
            RelationType::HappenedOn => write!(f, "happened_on"),
            RelationType::MentionedWith => write!(f, "mentioned_with"),
            RelationType::Custom(custom) => write!(f, "{}", custom),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rule_based_entity_extraction() {
        let config = EntityExtractionConfig::default();
        let extractor = RuleBasedEntityExtractor::new(config).unwrap();
        
        let text = "John Smith works at Microsoft Corporation in Seattle. The company was founded in 1975.";
        let entities = extractor.extract_entities(text, "test_doc").await.unwrap();
        
        assert!(!entities.is_empty());
        
        // Should find at least person, organization, and location
        let person_entities: Vec<_> = entities.iter()
            .filter(|e| matches!(e.entity_type, EntityType::Person))
            .collect();
        assert!(!person_entities.is_empty());
        
        let org_entities: Vec<_> = entities.iter()
            .filter(|e| matches!(e.entity_type, EntityType::Organization))
            .collect();
        assert!(!org_entities.is_empty());
    }

    #[tokio::test]
    async fn test_relationship_extraction() {
        let config = EntityExtractionConfig::default();
        let extractor = RuleBasedEntityExtractor::new(config).unwrap();
        
        let text = "Alice is a software engineer. She works for Google.";
        let (entities, relationships) = extractor.extract_all(text, "test_doc").await.unwrap();
        
        assert!(!entities.is_empty());
        assert!(!relationships.is_empty());
        
        // Should find work relationship
        let work_relations: Vec<_> = relationships.iter()
            .filter(|r| matches!(r.relation_type, RelationType::WorksFor))
            .collect();
        assert!(!work_relations.is_empty());
    }

    #[test]
    fn test_entity_to_node_conversion() {
        let entity = Entity {
            text: "John Smith".to_string(),
            entity_type: EntityType::Person,
            start_pos: 0,
            end_pos: 10,
            confidence: 0.9,
            normalized_form: Some("John Smith".to_string()),
            attributes: HashMap::new(),
            source_id: "test_doc".to_string(),
        };
        
        let nodes = entities_to_nodes(&[entity]);
        assert_eq!(nodes.len(), 1);
        assert!(matches!(nodes[0].node_type, NodeType::Entity(_)));
        assert_eq!(nodes[0].confidence, 0.9);
    }
}