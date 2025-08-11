//! # Optical Character Recognition (OCR)
//! 
//! Multi-engine OCR with text extraction, layout analysis, and confidence scoring.

use super::{
    OCREngine, OCRResult, OCRWord, BoundingBox, TextLayout, TextBlock, BlockType, 
    Column, OCRConfig, OCREngineType
};
use crate::{RragResult, RragError};
use std::path::Path;
use std::collections::HashMap;

/// Default OCR engine implementation
pub struct DefaultOCREngine {
    /// Configuration
    config: OCRConfig,
    
    /// Primary OCR engine
    primary_engine: Box<dyn OCREngineImpl>,
    
    /// Fallback engines
    fallback_engines: Vec<Box<dyn OCREngineImpl>>,
    
    /// Text post-processor
    post_processor: TextPostProcessor,
    
    /// Layout analyzer
    layout_analyzer: OCRLayoutAnalyzer,
}

/// OCR engine implementation trait
pub trait OCREngineImpl: Send + Sync {
    /// Extract text from image
    fn extract_text(&self, image_path: &Path) -> RragResult<OCRResult>;
    
    /// Get engine capabilities
    fn capabilities(&self) -> EngineCapabilities;
    
    /// Engine name
    fn name(&self) -> &str;
}

/// Engine capabilities
#[derive(Debug, Clone)]
pub struct EngineCapabilities {
    /// Supported languages
    pub languages: Vec<String>,
    
    /// Supports layout detection
    pub layout_detection: bool,
    
    /// Supports confidence scores
    pub confidence_scores: bool,
    
    /// Supports word-level results
    pub word_level: bool,
    
    /// Processing speed (relative)
    pub speed: ProcessingSpeed,
    
    /// Accuracy (relative)
    pub accuracy: AccuracyLevel,
}

/// Processing speed levels
#[derive(Debug, Clone, Copy)]
pub enum ProcessingSpeed {
    Fast,
    Medium,
    Slow,
}

/// Accuracy levels
#[derive(Debug, Clone, Copy)]
pub enum AccuracyLevel {
    Low,
    Medium,
    High,
}

/// Text post-processor
pub struct TextPostProcessor {
    /// Spell checker
    spell_checker: Option<SpellChecker>,
    
    /// Language detector
    language_detector: LanguageDetector,
    
    /// Text formatter
    formatter: TextFormatter,
}

/// Spell checker
pub struct SpellChecker {
    /// Dictionary paths by language
    dictionaries: HashMap<String, String>,
    
    /// Confidence threshold for corrections
    confidence_threshold: f32,
}

/// Language detector
pub struct LanguageDetector {
    /// Supported languages
    supported_languages: Vec<String>,
    
    /// Detection confidence threshold
    min_confidence: f32,
}

/// Text formatter
pub struct TextFormatter {
    /// Preserve line breaks
    preserve_line_breaks: bool,
    
    /// Preserve spacing
    preserve_spacing: bool,
    
    /// Clean up artifacts
    cleanup_artifacts: bool,
}

/// OCR layout analyzer
pub struct OCRLayoutAnalyzer {
    /// Block detection threshold
    block_threshold: f32,
    
    /// Column detection enabled
    column_detection: bool,
    
    /// Reading order detection
    reading_order_detection: bool,
}

/// Tesseract OCR engine
pub struct TesseractEngine {
    /// Language configuration
    languages: Vec<String>,
    
    /// OCR engine mode
    ocr_mode: TesseractOCRMode,
    
    /// Page segmentation mode
    psm: PageSegmentationMode,
}

/// Tesseract OCR modes
#[derive(Debug, Clone, Copy)]
pub enum TesseractOCRMode {
    LegacyOnly,
    NeuralOnly,
    LegacyAndNeural,
}

/// Page segmentation modes
#[derive(Debug, Clone, Copy)]
pub enum PageSegmentationMode {
    Auto,
    SingleColumn,
    SingleBlockVertText,
    SingleBlock,
    SingleLine,
    SingleWord,
    SingleCharacter,
    SparseText,
}

/// EasyOCR engine
pub struct EasyOCREngine {
    /// Language codes
    languages: Vec<String>,
    
    /// GPU acceleration
    gpu_enabled: bool,
    
    /// Text detection model
    detection_model: String,
    
    /// Text recognition model
    recognition_model: String,
}

/// PaddleOCR engine
pub struct PaddleOCREngine {
    /// Language
    language: String,
    
    /// Model precision
    precision: ModelPrecision,
    
    /// Text direction detection
    direction_detection: bool,
}

/// Model precision levels
#[derive(Debug, Clone, Copy)]
pub enum ModelPrecision {
    FP16,
    FP32,
    INT8,
}

/// Cloud Vision OCR engine
pub struct CloudVisionEngine {
    /// API credentials
    credentials: CloudCredentials,
    
    /// API endpoint
    endpoint: String,
    
    /// Request timeout
    timeout_ms: u64,
}

/// Cloud credentials
#[derive(Debug, Clone)]
pub struct CloudCredentials {
    pub api_key: String,
    pub project_id: Option<String>,
    pub region: Option<String>,
}

/// OCR quality assessment
#[derive(Debug, Clone)]
pub struct OCRQuality {
    /// Overall confidence
    pub overall_confidence: f32,
    
    /// Text quality score
    pub text_quality: f32,
    
    /// Layout quality score
    pub layout_quality: f32,
    
    /// Language detection confidence
    pub language_confidence: f32,
    
    /// Quality issues
    pub issues: Vec<QualityIssue>,
}

/// Quality issues in OCR
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: OCRIssueType,
    
    /// Issue description
    pub description: String,
    
    /// Severity
    pub severity: IssueSeverity,
    
    /// Location
    pub location: Option<BoundingBox>,
    
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// OCR issue types
#[derive(Debug, Clone, Copy)]
pub enum OCRIssueType {
    LowConfidence,
    PoorImageQuality,
    UnsupportedLanguage,
    LayoutComplexity,
    FontIssues,
    SkewedText,
    NoiseArtifacts,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl DefaultOCREngine {
    /// Create new OCR engine
    pub fn new(config: OCRConfig) -> RragResult<Self> {
        let primary_engine = Self::create_engine(config.engine, &config)?;
        let fallback_engines = Self::create_fallback_engines(&config)?;
        let post_processor = TextPostProcessor::new(&config)?;
        let layout_analyzer = OCRLayoutAnalyzer::new();
        
        Ok(Self {
            config,
            primary_engine,
            fallback_engines,
            post_processor,
            layout_analyzer,
        })
    }
    
    /// Create OCR engine based on type
    fn create_engine(engine_type: OCREngineType, config: &OCRConfig) -> RragResult<Box<dyn OCREngineImpl>> {
        match engine_type {
            OCREngineType::Tesseract => {
                Ok(Box::new(TesseractEngine::new(config.languages.clone())?))
            }
            OCREngineType::EasyOCR => {
                Ok(Box::new(EasyOCREngine::new(config.languages.clone())?))
            }
            OCREngineType::PaddleOCR => {
                let lang = config.languages.first().unwrap_or(&"en".to_string()).clone();
                Ok(Box::new(PaddleOCREngine::new(lang)?))
            }
            OCREngineType::CloudVision => {
                Ok(Box::new(CloudVisionEngine::new()?))
            }
        }
    }
    
    /// Create fallback engines
    fn create_fallback_engines(config: &OCRConfig) -> RragResult<Vec<Box<dyn OCREngineImpl>>> {
        let mut engines = Vec::new();
        
        // Add Tesseract as fallback if not primary
        if config.engine != OCREngineType::Tesseract {
            engines.push(Box::new(TesseractEngine::new(config.languages.clone())?) as Box<dyn OCREngineImpl>);
        }
        
        // Add EasyOCR as fallback if not primary
        if config.engine != OCREngineType::EasyOCR {
            engines.push(Box::new(EasyOCREngine::new(config.languages.clone())?) as Box<dyn OCREngineImpl>);
        }
        
        Ok(engines)
    }
    
    /// Perform OCR with fallback
    pub fn ocr_with_fallback(&self, image_path: &Path) -> RragResult<OCRResult> {
        // Try primary engine first
        match self.primary_engine.extract_text(image_path) {
            Ok(result) if result.confidence >= self.config.confidence_threshold => {
                return Ok(result);
            }
            Ok(primary_result) => {
                // Primary engine succeeded but confidence is low, try fallbacks
                for fallback in &self.fallback_engines {
                    if let Ok(fallback_result) = fallback.extract_text(image_path) {
                        if fallback_result.confidence > primary_result.confidence {
                            return Ok(fallback_result);
                        }
                    }
                }
                // Return primary result if no better fallback found
                Ok(primary_result)
            }
            Err(_) => {
                // Primary engine failed, try fallbacks
                for fallback in &self.fallback_engines {
                    if let Ok(result) = fallback.extract_text(image_path) {
                        return Ok(result);
                    }
                }
                Err(RragError::document_processing("All OCR engines failed"))
            }
        }
    }
    
    /// Assess OCR quality
    pub fn assess_quality(&self, result: &OCRResult) -> OCRQuality {
        let mut issues = Vec::new();
        
        // Check overall confidence
        if result.confidence < 0.7 {
            issues.push(QualityIssue {
                issue_type: OCRIssueType::LowConfidence,
                description: format!("Overall confidence is low: {:.2}", result.confidence),
                severity: if result.confidence < 0.5 { IssueSeverity::High } else { IssueSeverity::Medium },
                location: None,
                suggested_fix: Some("Consider using a higher resolution image or different OCR engine".to_string()),
            });
        }
        
        // Check for words with very low confidence
        let low_confidence_words = result.words.iter()
            .filter(|w| w.confidence < 0.5)
            .count();
        
        if low_confidence_words > result.words.len() / 4 {
            issues.push(QualityIssue {
                issue_type: OCRIssueType::LowConfidence,
                description: format!("{} words have low confidence", low_confidence_words),
                severity: IssueSeverity::Medium,
                location: None,
                suggested_fix: Some("Manual review recommended for low-confidence words".to_string()),
            });
        }
        
        OCRQuality {
            overall_confidence: result.confidence,
            text_quality: self.calculate_text_quality(result),
            layout_quality: 0.8, // Simplified
            language_confidence: 0.9, // Simplified
            issues,
        }
    }
    
    /// Calculate text quality score
    fn calculate_text_quality(&self, result: &OCRResult) -> f32 {
        if result.words.is_empty() {
            return 0.0;
        }
        
        // Average word confidence
        let avg_confidence = result.words.iter()
            .map(|w| w.confidence)
            .sum::<f32>() / result.words.len() as f32;
        
        // Penalize for very short words (likely noise)
        let short_words = result.words.iter()
            .filter(|w| w.text.len() <= 2)
            .count();
        let short_word_penalty = (short_words as f32 / result.words.len() as f32) * 0.2;
        
        (avg_confidence - short_word_penalty).max(0.0)
    }
}

impl OCREngine for DefaultOCREngine {
    fn ocr(&self, image_path: &Path) -> RragResult<OCRResult> {
        let mut result = self.ocr_with_fallback(image_path)?;
        
        // Post-process text if enabled
        if self.config.spell_correction {
            result = self.post_processor.process(result)?;
        }
        
        Ok(result)
    }
    
    fn get_text_with_confidence(&self, image_path: &Path) -> RragResult<Vec<(String, f32)>> {
        let result = self.ocr(image_path)?;
        Ok(result.words.into_iter()
            .map(|word| (word.text, word.confidence))
            .collect())
    }
    
    fn get_layout(&self, image_path: &Path) -> RragResult<TextLayout> {
        let result = self.ocr(image_path)?;
        self.layout_analyzer.analyze_layout(&result)
    }
}

impl TesseractEngine {
    /// Create new Tesseract engine
    pub fn new(languages: Vec<String>) -> RragResult<Self> {
        Ok(Self {
            languages,
            ocr_mode: TesseractOCRMode::LegacyAndNeural,
            psm: PageSegmentationMode::Auto,
        })
    }
}

impl OCREngineImpl for TesseractEngine {
    fn extract_text(&self, image_path: &Path) -> RragResult<OCRResult> {
        // Simulate Tesseract OCR
        let text = format!("Sample text extracted from {:?}", image_path.file_name().unwrap_or_default());
        
        let words = vec![
            OCRWord {
                text: "Sample".to_string(),
                confidence: 0.95,
                bounding_box: BoundingBox { x: 10, y: 10, width: 50, height: 20 },
            },
            OCRWord {
                text: "text".to_string(),
                confidence: 0.90,
                bounding_box: BoundingBox { x: 65, y: 10, width: 30, height: 20 },
            },
        ];
        
        Ok(OCRResult {
            text,
            confidence: 0.925,
            words,
            languages: self.languages.clone(),
        })
    }
    
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            languages: vec!["eng", "fra", "deu", "spa", "chi_sim"].iter().map(|s| s.to_string()).collect(),
            layout_detection: true,
            confidence_scores: true,
            word_level: true,
            speed: ProcessingSpeed::Medium,
            accuracy: AccuracyLevel::High,
        }
    }
    
    fn name(&self) -> &str {
        "Tesseract"
    }
}

impl EasyOCREngine {
    /// Create new EasyOCR engine
    pub fn new(languages: Vec<String>) -> RragResult<Self> {
        Ok(Self {
            languages,
            gpu_enabled: false,
            detection_model: "craft".to_string(),
            recognition_model: "crnn".to_string(),
        })
    }
}

impl OCREngineImpl for EasyOCREngine {
    fn extract_text(&self, image_path: &Path) -> RragResult<OCRResult> {
        // Simulate EasyOCR
        let text = format!("EasyOCR extracted text from {:?}", image_path.file_name().unwrap_or_default());
        
        let words = vec![
            OCRWord {
                text: "EasyOCR".to_string(),
                confidence: 0.88,
                bounding_box: BoundingBox { x: 5, y: 5, width: 60, height: 25 },
            },
            OCRWord {
                text: "extracted".to_string(),
                confidence: 0.92,
                bounding_box: BoundingBox { x: 70, y: 5, width: 70, height: 25 },
            },
        ];
        
        Ok(OCRResult {
            text,
            confidence: 0.90,
            words,
            languages: self.languages.clone(),
        })
    }
    
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            languages: vec!["en", "ch_sim", "ch_tra", "ja", "ko", "fr", "de"].iter().map(|s| s.to_string()).collect(),
            layout_detection: true,
            confidence_scores: true,
            word_level: true,
            speed: ProcessingSpeed::Fast,
            accuracy: AccuracyLevel::Medium,
        }
    }
    
    fn name(&self) -> &str {
        "EasyOCR"
    }
}

impl PaddleOCREngine {
    /// Create new PaddleOCR engine
    pub fn new(language: String) -> RragResult<Self> {
        Ok(Self {
            language,
            precision: ModelPrecision::FP32,
            direction_detection: true,
        })
    }
}

impl OCREngineImpl for PaddleOCREngine {
    fn extract_text(&self, image_path: &Path) -> RragResult<OCRResult> {
        // Simulate PaddleOCR
        let text = format!("PaddleOCR text from {:?}", image_path.file_name().unwrap_or_default());
        
        let words = vec![
            OCRWord {
                text: "PaddleOCR".to_string(),
                confidence: 0.93,
                bounding_box: BoundingBox { x: 8, y: 8, width: 80, height: 22 },
            },
        ];
        
        Ok(OCRResult {
            text,
            confidence: 0.93,
            words,
            languages: vec![self.language.clone()],
        })
    }
    
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            languages: vec!["ch", "en", "fr", "german", "japan", "korean"].iter().map(|s| s.to_string()).collect(),
            layout_detection: true,
            confidence_scores: true,
            word_level: true,
            speed: ProcessingSpeed::Fast,
            accuracy: AccuracyLevel::High,
        }
    }
    
    fn name(&self) -> &str {
        "PaddleOCR"
    }
}

impl CloudVisionEngine {
    /// Create new Cloud Vision engine
    pub fn new() -> RragResult<Self> {
        Ok(Self {
            credentials: CloudCredentials {
                api_key: "demo_key".to_string(),
                project_id: Some("demo_project".to_string()),
                region: Some("us-central1".to_string()),
            },
            endpoint: "https://vision.googleapis.com".to_string(),
            timeout_ms: 30000,
        })
    }
}

impl OCREngineImpl for CloudVisionEngine {
    fn extract_text(&self, image_path: &Path) -> RragResult<OCRResult> {
        // Simulate Cloud Vision API call
        let text = format!("Cloud Vision text from {:?}", image_path.file_name().unwrap_or_default());
        
        let words = vec![
            OCRWord {
                text: "Cloud".to_string(),
                confidence: 0.98,
                bounding_box: BoundingBox { x: 12, y: 12, width: 45, height: 18 },
            },
            OCRWord {
                text: "Vision".to_string(),
                confidence: 0.97,
                bounding_box: BoundingBox { x: 60, y: 12, width: 50, height: 18 },
            },
        ];
        
        Ok(OCRResult {
            text,
            confidence: 0.975,
            words,
            languages: vec!["en".to_string()],
        })
    }
    
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            languages: vec!["en", "zh", "ja", "ko", "hi", "ar", "fr", "de", "es", "pt"].iter().map(|s| s.to_string()).collect(),
            layout_detection: true,
            confidence_scores: true,
            word_level: true,
            speed: ProcessingSpeed::Slow, // Network latency
            accuracy: AccuracyLevel::High,
        }
    }
    
    fn name(&self) -> &str {
        "Cloud Vision"
    }
}

impl TextPostProcessor {
    /// Create new text post-processor
    pub fn new(config: &OCRConfig) -> RragResult<Self> {
        let spell_checker = if config.spell_correction {
            Some(SpellChecker::new(&config.languages)?)
        } else {
            None
        };
        
        let language_detector = LanguageDetector::new(config.languages.clone());
        let formatter = TextFormatter::new(config.preserve_formatting);
        
        Ok(Self {
            spell_checker,
            language_detector,
            formatter,
        })
    }
    
    /// Process OCR result
    pub fn process(&self, mut result: OCRResult) -> RragResult<OCRResult> {
        // Spell checking
        if let Some(ref checker) = self.spell_checker {
            result = checker.correct(result)?;
        }
        
        // Language detection
        let detected_languages = self.language_detector.detect(&result.text)?;
        if !detected_languages.is_empty() {
            result.languages = detected_languages;
        }
        
        // Text formatting
        result = self.formatter.format(result)?;
        
        Ok(result)
    }
}

impl SpellChecker {
    /// Create new spell checker
    pub fn new(languages: &[String]) -> RragResult<Self> {
        let mut dictionaries = HashMap::new();
        for lang in languages {
            dictionaries.insert(lang.clone(), format!("dict_{}.txt", lang));
        }
        
        Ok(Self {
            dictionaries,
            confidence_threshold: 0.7,
        })
    }
    
    /// Correct spelling in OCR result
    pub fn correct(&self, mut result: OCRResult) -> RragResult<OCRResult> {
        // Simple spell correction simulation
        for word in &mut result.words {
            if word.confidence < self.confidence_threshold {
                word.text = self.suggest_correction(&word.text);
                word.confidence = (word.confidence + 0.1).min(1.0);
            }
        }
        
        // Rebuild text from corrected words
        result.text = result.words.iter()
            .map(|w| w.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        
        Ok(result)
    }
    
    /// Suggest spelling correction
    fn suggest_correction(&self, word: &str) -> String {
        // Simple correction rules (in practice would use proper spell checker)
        match word.to_lowercase().as_str() {
            "teh" => "the".to_string(),
            "adn" => "and".to_string(),
            "taht" => "that".to_string(),
            _ => word.to_string(),
        }
    }
}

impl LanguageDetector {
    /// Create new language detector
    pub fn new(supported_languages: Vec<String>) -> Self {
        Self {
            supported_languages,
            min_confidence: 0.8,
        }
    }
    
    /// Detect languages in text
    pub fn detect(&self, text: &str) -> RragResult<Vec<String>> {
        // Simple language detection (would use proper language detection library)
        if text.chars().any(|c| c as u32 > 127) {
            // Contains non-ASCII characters, might be non-English
            if text.chars().any(|c| '\u{4e00}' <= c && c <= '\u{9fff}') {
                Ok(vec!["zh".to_string()])
            } else if text.chars().any(|c| '\u{3040}' <= c && c <= '\u{309f}') {
                Ok(vec!["ja".to_string()])
            } else {
                Ok(vec!["en".to_string()]) // Default to English
            }
        } else {
            Ok(vec!["en".to_string()])
        }
    }
}

impl TextFormatter {
    /// Create new text formatter
    pub fn new(preserve_formatting: bool) -> Self {
        Self {
            preserve_line_breaks: preserve_formatting,
            preserve_spacing: preserve_formatting,
            cleanup_artifacts: true,
        }
    }
    
    /// Format OCR result
    pub fn format(&self, mut result: OCRResult) -> RragResult<OCRResult> {
        if self.cleanup_artifacts {
            result.text = self.cleanup_text(&result.text);
        }
        
        if !self.preserve_spacing {
            result.text = self.normalize_spacing(&result.text);
        }
        
        if !self.preserve_line_breaks {
            result.text = result.text.replace('\n', " ");
        }
        
        Ok(result)
    }
    
    /// Clean up OCR artifacts
    fn cleanup_text(&self, text: &str) -> String {
        text.chars()
            .filter(|&c| c.is_ascii_graphic() || c.is_whitespace())
            .collect::<String>()
            .trim()
            .to_string()
    }
    
    /// Normalize spacing
    fn normalize_spacing(&self, text: &str) -> String {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl OCRLayoutAnalyzer {
    /// Create new layout analyzer
    pub fn new() -> Self {
        Self {
            block_threshold: 0.1,
            column_detection: true,
            reading_order_detection: true,
        }
    }
    
    /// Analyze layout from OCR result
    pub fn analyze_layout(&self, result: &OCRResult) -> RragResult<TextLayout> {
        let blocks = self.detect_blocks(result)?;
        let reading_order = self.determine_reading_order(&blocks)?;
        let columns = if self.column_detection {
            Some(self.detect_columns(&blocks)?)
        } else {
            None
        };
        
        Ok(TextLayout {
            blocks,
            reading_order,
            columns,
        })
    }
    
    /// Detect text blocks
    fn detect_blocks(&self, result: &OCRResult) -> RragResult<Vec<TextBlock>> {
        let mut blocks = Vec::new();
        
        // Group words into blocks based on proximity
        let mut current_block_words = Vec::new();
        let mut current_y = 0u32;
        
        for word in &result.words {
            if current_block_words.is_empty() || 
               (word.bounding_box.y as i32 - current_y as i32).abs() < 10 {
                current_block_words.push(word);
                current_y = word.bounding_box.y;
            } else {
                // Start new block
                if !current_block_words.is_empty() {
                    blocks.push(self.create_block_from_words(&current_block_words, blocks.len()));
                }
                current_block_words = vec![word];
                current_y = word.bounding_box.y;
            }
        }
        
        // Add final block
        if !current_block_words.is_empty() {
            blocks.push(self.create_block_from_words(&current_block_words, blocks.len()));
        }
        
        Ok(blocks)
    }
    
    /// Create text block from words
    fn create_block_from_words(&self, words: &[&OCRWord], id: usize) -> TextBlock {
        let text = words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
        
        // Calculate bounding box
        let min_x = words.iter().map(|w| w.bounding_box.x).min().unwrap_or(0);
        let min_y = words.iter().map(|w| w.bounding_box.y).min().unwrap_or(0);
        let max_x = words.iter().map(|w| w.bounding_box.x + w.bounding_box.width).max().unwrap_or(0);
        let max_y = words.iter().map(|w| w.bounding_box.y + w.bounding_box.height).max().unwrap_or(0);
        
        let bounding_box = BoundingBox {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        };
        
        // Determine block type (simplified)
        let block_type = if text.len() < 20 && words.len() <= 3 {
            BlockType::Title
        } else if text.ends_with(':') {
            BlockType::Heading
        } else {
            BlockType::Paragraph
        };
        
        TextBlock {
            id,
            text,
            bounding_box,
            block_type,
        }
    }
    
    /// Determine reading order
    fn determine_reading_order(&self, blocks: &[TextBlock]) -> RragResult<Vec<usize>> {
        if !self.reading_order_detection {
            return Ok((0..blocks.len()).collect());
        }
        
        // Sort by Y position first, then by X position
        let mut indexed_blocks: Vec<(usize, &TextBlock)> = blocks.iter().enumerate().collect();
        indexed_blocks.sort_by(|a, b| {
            a.1.bounding_box.y.cmp(&b.1.bounding_box.y)
                .then_with(|| a.1.bounding_box.x.cmp(&b.1.bounding_box.x))
        });
        
        Ok(indexed_blocks.into_iter().map(|(idx, _)| idx).collect())
    }
    
    /// Detect columns
    fn detect_columns(&self, blocks: &[TextBlock]) -> RragResult<Vec<Column>> {
        // Simple column detection based on X positions
        let mut columns = Vec::new();
        
        if blocks.is_empty() {
            return Ok(columns);
        }
        
        // Group blocks by X position (simplified)
        let mut x_groups: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
        
        for (idx, block) in blocks.iter().enumerate() {
            let x_group = (block.bounding_box.x / 100) * 100; // Group by 100px
            x_groups.entry(x_group).or_insert_with(Vec::new).push(idx);
        }
        
        // Convert groups to columns
        for (x_pos, block_indices) in x_groups {
            columns.push(Column {
                index: columns.len(),
                blocks: block_indices,
                width: 100, // Simplified
            });
        }
        
        // Sort columns by X position
        columns.sort_by_key(|c| c.index);
        
        Ok(columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_ocr_engine_creation() {
        let config = OCRConfig::default();
        let engine = DefaultOCREngine::new(config).unwrap();
        
        assert_eq!(engine.config.confidence_threshold, 0.7);
        assert!(engine.config.spell_correction);
    }
    
    #[test]
    fn test_tesseract_engine() {
        let engine = TesseractEngine::new(vec!["eng".to_string()]).unwrap();
        let capabilities = engine.capabilities();
        
        assert!(capabilities.confidence_scores);
        assert!(capabilities.layout_detection);
        assert_eq!(engine.name(), "Tesseract");
    }
    
    #[test]
    fn test_spell_checker() {
        let checker = SpellChecker::new(&["en".to_string()]).unwrap();
        let correction = checker.suggest_correction("teh");
        assert_eq!(correction, "the");
    }
    
    #[test]
    fn test_language_detector() {
        let detector = LanguageDetector::new(vec!["en".to_string(), "zh".to_string()]);
        
        let english_result = detector.detect("Hello world").unwrap();
        assert_eq!(english_result, vec!["en"]);
        
        let chinese_result = detector.detect("你好世界").unwrap();
        assert_eq!(chinese_result, vec!["zh"]);
    }
    
    #[test]
    fn test_text_formatter() {
        let formatter = TextFormatter::new(false);
        
        let result = OCRResult {
            text: "  Hello    world  \n  test  ".to_string(),
            confidence: 0.9,
            words: vec![],
            languages: vec!["en".to_string()],
        };
        
        let formatted = formatter.format(result).unwrap();
        assert_eq!(formatted.text, "Hello world test");
    }
    
    #[test]
    fn test_layout_analysis() {
        let analyzer = OCRLayoutAnalyzer::new();
        
        let result = OCRResult {
            text: "Sample text".to_string(),
            confidence: 0.9,
            words: vec![
                OCRWord {
                    text: "Sample".to_string(),
                    confidence: 0.9,
                    bounding_box: BoundingBox { x: 10, y: 10, width: 50, height: 20 },
                },
                OCRWord {
                    text: "text".to_string(),
                    confidence: 0.9,
                    bounding_box: BoundingBox { x: 65, y: 10, width: 30, height: 20 },
                },
            ],
            languages: vec!["en".to_string()],
        };
        
        let layout = analyzer.analyze_layout(&result).unwrap();
        assert!(!layout.blocks.is_empty());
        assert!(!layout.reading_order.is_empty());
    }
}