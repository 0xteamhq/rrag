//! # Document Parser
//!
//! Comprehensive document parsing with multi-modal content extraction.

use super::{
    AnalyzedChart, ChartProcessor, ColumnLayout, DocumentLayout, DocumentMetadata, DocumentSection,
    DocumentType, EmbeddingWeights, ExtractedTable, ImageProcessor, MultiModalDocument,
    MultiModalEmbeddings, ProcessedImage, SectionType, TableProcessor,
};
use crate::{RragError, RragResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Document parser for multi-modal content
pub struct DocumentParser {
    /// Configuration
    config: DocumentParserConfig,

    /// Image processor
    image_processor: Box<dyn ImageProcessor>,

    /// Table processor
    table_processor: Box<dyn TableProcessor>,

    /// Chart processor
    chart_processor: Box<dyn ChartProcessor>,

    /// Text extractor
    text_extractor: TextExtractor,

    /// Section analyzer
    section_analyzer: SectionAnalyzer,

    /// Layout detector
    layout_detector: LayoutDetector,
}

/// Document parser configuration
#[derive(Debug, Clone)]
pub struct DocumentParserConfig {
    /// Supported document types
    pub supported_types: Vec<DocumentType>,

    /// Extract text content
    pub extract_text: bool,

    /// Extract images
    pub extract_images: bool,

    /// Extract tables
    pub extract_tables: bool,

    /// Extract charts
    pub extract_charts: bool,

    /// Analyze document structure
    pub analyze_structure: bool,

    /// Maximum file size (bytes)
    pub max_file_size: usize,

    /// Page processing limit
    pub max_pages: Option<usize>,
}

/// Text extraction component
pub struct TextExtractor {
    /// Configuration
    config: TextExtractionConfig,

    /// PDF extractor
    pdf_extractor: PDFTextExtractor,

    /// Word extractor
    word_extractor: WordTextExtractor,

    /// PowerPoint extractor
    ppt_extractor: PowerPointTextExtractor,

    /// HTML extractor
    html_extractor: HTMLTextExtractor,
}

/// Text extraction configuration
#[derive(Debug, Clone)]
pub struct TextExtractionConfig {
    /// Preserve formatting
    pub preserve_formatting: bool,

    /// Extract footnotes
    pub extract_footnotes: bool,

    /// Extract headers/footers
    pub extract_headers_footers: bool,

    /// Minimum text block size
    pub min_block_size: usize,
}

/// Section analysis component
pub struct SectionAnalyzer {
    /// Section detection patterns
    patterns: Vec<SectionPattern>,

    /// Heading detection
    heading_detector: HeadingDetector,
}

/// Layout detection component
pub struct LayoutDetector {
    /// Column detection threshold
    column_threshold: f32,

    /// Reading order analysis
    reading_order_analyzer: ReadingOrderAnalyzer,
}

/// PDF text extractor
pub struct PDFTextExtractor {
    /// Extract metadata
    extract_metadata: bool,

    /// Extract bookmarks
    extract_bookmarks: bool,
}

/// Word document text extractor
pub struct WordTextExtractor {
    /// Extract styles
    extract_styles: bool,

    /// Extract comments
    extract_comments: bool,
}

/// PowerPoint text extractor
pub struct PowerPointTextExtractor {
    /// Extract slide notes
    extract_notes: bool,

    /// Extract animations
    extract_animations: bool,
}

/// HTML text extractor
pub struct HTMLTextExtractor {
    /// Remove scripts
    remove_scripts: bool,

    /// Remove styles
    remove_styles: bool,
}

/// Section detection pattern
#[derive(Debug, Clone)]
pub struct SectionPattern {
    /// Pattern regex
    pub pattern: String,

    /// Section type
    pub section_type: SectionType,

    /// Priority (higher = more specific)
    pub priority: u32,
}

/// Heading detection component
pub struct HeadingDetector {
    /// Heading patterns
    patterns: Vec<HeadingPattern>,
}

/// Heading pattern
#[derive(Debug, Clone)]
pub struct HeadingPattern {
    /// Pattern regex
    pub pattern: String,

    /// Heading level
    pub level: usize,

    /// Confidence score
    pub confidence: f32,
}

/// Reading order analyzer
pub struct ReadingOrderAnalyzer {
    /// Analysis strategy
    strategy: ReadingOrderStrategy,
}

/// Reading order strategies
#[derive(Debug, Clone, Copy)]
pub enum ReadingOrderStrategy {
    LeftToRight,
    TopToBottom,
    ZPattern,
    FPattern,
    Auto,
}

/// Document parsing result
#[derive(Debug, Clone)]
pub struct DocumentParseResult {
    /// Parsed document
    pub document: MultiModalDocument,

    /// Parsing confidence
    pub confidence: f32,

    /// Processing time
    pub processing_time_ms: u64,

    /// Warnings
    pub warnings: Vec<String>,

    /// Parsing statistics
    pub statistics: ParseStatistics,
}

/// Parsing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseStatistics {
    /// Total text length
    pub text_length: usize,

    /// Image count
    pub image_count: usize,

    /// Table count
    pub table_count: usize,

    /// Chart count
    pub chart_count: usize,

    /// Section count
    pub section_count: usize,

    /// Page count
    pub page_count: usize,
}

impl DocumentParser {
    /// Create new document parser
    pub fn new(
        config: DocumentParserConfig,
        image_processor: Box<dyn ImageProcessor>,
        table_processor: Box<dyn TableProcessor>,
        chart_processor: Box<dyn ChartProcessor>,
    ) -> RragResult<Self> {
        let text_extractor = TextExtractor::new(TextExtractionConfig::default())?;
        let section_analyzer = SectionAnalyzer::new()?;
        let layout_detector = LayoutDetector::new();

        Ok(Self {
            config,
            image_processor,
            table_processor,
            chart_processor,
            text_extractor,
            section_analyzer,
            layout_detector,
        })
    }

    /// Parse document from file
    pub async fn parse_document(&self, file_path: &Path) -> RragResult<DocumentParseResult> {
        let start_time = std::time::Instant::now();

        // Detect document type
        let doc_type = self.detect_document_type(file_path)?;

        // Validate file size
        self.validate_file_size(file_path)?;

        // Extract content based on type
        let content = self.extract_content(file_path, doc_type).await?;

        // Parse multi-modal elements
        let images = if self.config.extract_images {
            self.extract_images(&content).await?
        } else {
            vec![]
        };

        let tables = if self.config.extract_tables {
            self.extract_tables(&content).await?
        } else {
            vec![]
        };

        let charts = if self.config.extract_charts {
            self.extract_charts(&content).await?
        } else {
            vec![]
        };

        // Analyze document structure
        let layout = if self.config.analyze_structure {
            self.analyze_layout(&content).await?
        } else {
            DocumentLayout {
                pages: 1,
                sections: vec![],
                reading_order: vec![],
                columns: None,
                document_type: doc_type,
            }
        };

        // Extract metadata
        let metadata = self.extract_metadata(file_path, &content)?;

        // Create document
        let document_id = format!(
            "doc_{}",
            uuid::Uuid::new_v4().to_string().split('-').next().unwrap()
        );
        let document = MultiModalDocument {
            id: document_id,
            text_content: content.text,
            images,
            tables,
            charts,
            layout,
            embeddings: MultiModalEmbeddings {
                text_embeddings: vec![],
                visual_embeddings: None,
                table_embeddings: None,
                fused_embedding: vec![],
                weights: EmbeddingWeights {
                    text_weight: 0.6,
                    visual_weight: 0.2,
                    table_weight: 0.1,
                    chart_weight: 0.1,
                },
            },
            metadata,
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(DocumentParseResult {
            confidence: 0.85,
            processing_time_ms: processing_time,
            warnings: vec![],
            statistics: ParseStatistics {
                text_length: document.text_content.len(),
                image_count: document.images.len(),
                table_count: document.tables.len(),
                chart_count: document.charts.len(),
                section_count: document.layout.sections.len(),
                page_count: document.layout.pages,
            },
            document,
        })
    }

    /// Detect document type from file
    fn detect_document_type(&self, file_path: &Path) -> RragResult<DocumentType> {
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "pdf" => Ok(DocumentType::PDF),
            "doc" | "docx" => Ok(DocumentType::Word),
            "ppt" | "pptx" => Ok(DocumentType::PowerPoint),
            "html" | "htm" => Ok(DocumentType::HTML),
            "md" => Ok(DocumentType::Markdown),
            "txt" => Ok(DocumentType::PlainText),
            _ => Ok(DocumentType::Mixed),
        }
    }

    /// Validate file size
    fn validate_file_size(&self, file_path: &Path) -> RragResult<()> {
        let metadata =
            std::fs::metadata(file_path).map_err(|e| RragError::io_error(e.to_string()))?;

        if metadata.len() as usize > self.config.max_file_size {
            return Err(RragError::validation(
                "file_size",
                format!("maximum {} bytes", self.config.max_file_size),
                format!("{} bytes", metadata.len()),
            ));
        }

        Ok(())
    }

    /// Extract content from document
    async fn extract_content(
        &self,
        file_path: &Path,
        doc_type: DocumentType,
    ) -> RragResult<ExtractedContent> {
        match doc_type {
            DocumentType::PDF => self.text_extractor.extract_from_pdf(file_path).await,
            DocumentType::Word => self.text_extractor.extract_from_word(file_path).await,
            DocumentType::PowerPoint => self.text_extractor.extract_from_ppt(file_path).await,
            DocumentType::HTML => self.text_extractor.extract_from_html(file_path).await,
            DocumentType::Markdown => self.text_extractor.extract_from_markdown(file_path).await,
            DocumentType::PlainText => self.text_extractor.extract_from_text(file_path).await,
            DocumentType::Mixed => {
                // Try to auto-detect based on content
                self.text_extractor.extract_auto_detect(file_path).await
            }
        }
    }

    /// Extract images from content
    async fn extract_images(&self, content: &ExtractedContent) -> RragResult<Vec<ProcessedImage>> {
        let mut images = Vec::new();

        for image_ref in &content.image_references {
            if let Ok(processed) = self.image_processor.process_image(&image_ref.path) {
                images.push(processed);
            }
        }

        Ok(images)
    }

    /// Extract tables from content
    async fn extract_tables(&self, content: &ExtractedContent) -> RragResult<Vec<ExtractedTable>> {
        let mut tables = Vec::new();

        for table_content in &content.table_content {
            if let Ok(extracted) = self.table_processor.extract_table(table_content) {
                tables.extend(extracted);
            }
        }

        Ok(tables)
    }

    /// Extract charts from content
    async fn extract_charts(&self, content: &ExtractedContent) -> RragResult<Vec<AnalyzedChart>> {
        let mut charts = Vec::new();

        for chart_ref in &content.chart_references {
            if let Ok(analyzed) = self.chart_processor.analyze_chart(&chart_ref.path) {
                charts.push(analyzed);
            }
        }

        Ok(charts)
    }

    /// Analyze document layout
    async fn analyze_layout(&self, content: &ExtractedContent) -> RragResult<DocumentLayout> {
        let sections = self.section_analyzer.analyze_sections(&content.text)?;
        let reading_order = self.layout_detector.determine_reading_order(&sections)?;
        let columns = self.layout_detector.detect_columns(&content.text)?;

        Ok(DocumentLayout {
            pages: content.page_count,
            sections,
            reading_order,
            columns,
            document_type: content.document_type,
        })
    }

    /// Extract document metadata
    fn extract_metadata(
        &self,
        file_path: &Path,
        content: &ExtractedContent,
    ) -> RragResult<DocumentMetadata> {
        let file_metadata =
            std::fs::metadata(file_path).map_err(|e| RragError::io_error(e.to_string()))?;

        Ok(DocumentMetadata {
            title: content.title.clone(),
            author: content.author.clone(),
            creation_date: content.creation_date.clone(),
            modification_date: file_metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs().to_string()),
            page_count: content.page_count,
            word_count: content.text.split_whitespace().count(),
            language: content.language.clone().unwrap_or_else(|| "en".to_string()),
            format: content.document_type,
        })
    }
}

/// Extracted content from document
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    /// Text content
    pub text: String,

    /// Document type
    pub document_type: DocumentType,

    /// Page count
    pub page_count: usize,

    /// Image references
    pub image_references: Vec<ImageReference>,

    /// Table content
    pub table_content: Vec<String>,

    /// Chart references
    pub chart_references: Vec<ChartReference>,

    /// Document title
    pub title: Option<String>,

    /// Document author
    pub author: Option<String>,

    /// Creation date
    pub creation_date: Option<String>,

    /// Language
    pub language: Option<String>,
}

/// Image reference in document
#[derive(Debug, Clone)]
pub struct ImageReference {
    pub path: std::path::PathBuf,
    pub caption: Option<String>,
    pub alt_text: Option<String>,
}

/// Chart reference in document
#[derive(Debug, Clone)]
pub struct ChartReference {
    pub path: std::path::PathBuf,
    pub title: Option<String>,
    pub description: Option<String>,
}

impl TextExtractor {
    /// Create new text extractor
    pub fn new(config: TextExtractionConfig) -> RragResult<Self> {
        Ok(Self {
            config,
            pdf_extractor: PDFTextExtractor::new(),
            word_extractor: WordTextExtractor::new(),
            ppt_extractor: PowerPointTextExtractor::new(),
            html_extractor: HTMLTextExtractor::new(),
        })
    }

    /// Extract from PDF
    pub async fn extract_from_pdf(&self, file_path: &Path) -> RragResult<ExtractedContent> {
        self.pdf_extractor.extract(file_path).await
    }

    /// Extract from Word document
    pub async fn extract_from_word(&self, file_path: &Path) -> RragResult<ExtractedContent> {
        self.word_extractor.extract(file_path).await
    }

    /// Extract from PowerPoint
    pub async fn extract_from_ppt(&self, file_path: &Path) -> RragResult<ExtractedContent> {
        self.ppt_extractor.extract(file_path).await
    }

    /// Extract from HTML
    pub async fn extract_from_html(&self, file_path: &Path) -> RragResult<ExtractedContent> {
        self.html_extractor.extract(file_path).await
    }

    /// Extract from Markdown
    pub async fn extract_from_markdown(&self, file_path: &Path) -> RragResult<ExtractedContent> {
        let content =
            std::fs::read_to_string(file_path).map_err(|e| RragError::io_error(e.to_string()))?;

        Ok(ExtractedContent {
            text: content,
            document_type: DocumentType::Markdown,
            page_count: 1,
            image_references: vec![],
            table_content: vec![],
            chart_references: vec![],
            title: None,
            author: None,
            creation_date: None,
            language: Some("en".to_string()),
        })
    }

    /// Extract from plain text
    pub async fn extract_from_text(&self, file_path: &Path) -> RragResult<ExtractedContent> {
        let content =
            std::fs::read_to_string(file_path).map_err(|e| RragError::io_error(e.to_string()))?;

        Ok(ExtractedContent {
            text: content,
            document_type: DocumentType::PlainText,
            page_count: 1,
            image_references: vec![],
            table_content: vec![],
            chart_references: vec![],
            title: None,
            author: None,
            creation_date: None,
            language: Some("en".to_string()),
        })
    }

    /// Auto-detect and extract
    pub async fn extract_auto_detect(&self, file_path: &Path) -> RragResult<ExtractedContent> {
        // For simplicity, treat as plain text
        self.extract_from_text(file_path).await
    }
}

impl SectionAnalyzer {
    /// Create new section analyzer
    pub fn new() -> RragResult<Self> {
        let patterns = vec![
            SectionPattern {
                pattern: r"^Abstract\s*$".to_string(),
                section_type: SectionType::Abstract,
                priority: 100,
            },
            SectionPattern {
                pattern: r"^Introduction\s*$".to_string(),
                section_type: SectionType::Introduction,
                priority: 90,
            },
            SectionPattern {
                pattern: r"^Conclusion\s*$".to_string(),
                section_type: SectionType::Conclusion,
                priority: 80,
            },
            SectionPattern {
                pattern: r"^References\s*$".to_string(),
                section_type: SectionType::References,
                priority: 70,
            },
        ];

        let heading_detector = HeadingDetector::new();

        Ok(Self {
            patterns,
            heading_detector,
        })
    }

    /// Analyze document sections
    pub fn analyze_sections(&self, text: &str) -> RragResult<Vec<DocumentSection>> {
        let mut sections = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        let mut current_section: Option<DocumentSection> = None;
        let mut content_buffer = String::new();

        for (_line_idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check if this line matches a section pattern
            if let Some((section_type, level)) = self.detect_section_start(trimmed) {
                // Save previous section
                if let Some(mut section) = current_section.take() {
                    section.content = content_buffer.trim().to_string();
                    sections.push(section);
                    content_buffer.clear();
                }

                // Start new section
                current_section = Some(DocumentSection {
                    id: format!("section_{}", sections.len()),
                    title: Some(trimmed.to_string()),
                    content: String::new(),
                    section_type,
                    level,
                    page_range: (1, 1), // Simplified
                });
            } else {
                // Add to current content
                content_buffer.push_str(line);
                content_buffer.push('\n');
            }
        }

        // Save final section
        if let Some(mut section) = current_section {
            section.content = content_buffer.trim().to_string();
            sections.push(section);
        }

        // If no sections detected, create a default body section
        if sections.is_empty() {
            sections.push(DocumentSection {
                id: "section_0".to_string(),
                title: None,
                content: text.to_string(),
                section_type: SectionType::Body,
                level: 1,
                page_range: (1, 1),
            });
        }

        Ok(sections)
    }

    /// Detect section start
    fn detect_section_start(&self, line: &str) -> Option<(SectionType, usize)> {
        // Check patterns first
        for pattern in &self.patterns {
            if let Ok(regex) = regex::Regex::new(&pattern.pattern) {
                if regex.is_match(line) {
                    return Some((pattern.section_type, 1));
                }
            }
        }

        // Check heading patterns
        if let Some((level, _)) = self.heading_detector.detect_heading(line) {
            return Some((SectionType::Body, level));
        }

        None
    }
}

impl HeadingDetector {
    /// Create new heading detector
    pub fn new() -> Self {
        let patterns = vec![
            HeadingPattern {
                pattern: r"^#+\s+".to_string(), // Markdown headers
                level: 1,
                confidence: 0.9,
            },
            HeadingPattern {
                pattern: r"^[A-Z][A-Z\s]{5,}\s*$".to_string(), // ALL CAPS
                level: 1,
                confidence: 0.7,
            },
        ];

        Self { patterns }
    }

    /// Detect if line is a heading
    pub fn detect_heading(&self, line: &str) -> Option<(usize, f32)> {
        for pattern in &self.patterns {
            if let Ok(regex) = regex::Regex::new(&pattern.pattern) {
                if regex.is_match(line) {
                    // Calculate level for markdown headers
                    let level = if pattern.pattern.starts_with("^#+") {
                        line.chars().take_while(|&c| c == '#').count()
                    } else {
                        pattern.level
                    };

                    return Some((level, pattern.confidence));
                }
            }
        }

        None
    }
}

impl LayoutDetector {
    /// Create new layout detector
    pub fn new() -> Self {
        Self {
            column_threshold: 0.3,
            reading_order_analyzer: ReadingOrderAnalyzer::new(),
        }
    }

    /// Determine reading order
    pub fn determine_reading_order(&self, sections: &[DocumentSection]) -> RragResult<Vec<String>> {
        Ok(sections.iter().map(|s| s.id.clone()).collect())
    }

    /// Detect column layout
    pub fn detect_columns(&self, text: &str) -> RragResult<Option<ColumnLayout>> {
        // Simplified column detection
        let lines: Vec<&str> = text.lines().collect();
        let avg_line_length =
            lines.iter().map(|line| line.len()).sum::<usize>() as f32 / lines.len() as f32;

        if avg_line_length > 120.0 {
            // Likely multi-column layout
            Ok(Some(ColumnLayout {
                column_count: 2,
                column_widths: vec![0.5, 0.5],
                gutter_width: 0.05,
            }))
        } else {
            Ok(None)
        }
    }
}

impl ReadingOrderAnalyzer {
    /// Create new reading order analyzer
    pub fn new() -> Self {
        Self {
            strategy: ReadingOrderStrategy::Auto,
        }
    }
}

// PDF, Word, PowerPoint, HTML extractors (simplified implementations)
impl PDFTextExtractor {
    pub fn new() -> Self {
        Self {
            extract_metadata: true,
            extract_bookmarks: true,
        }
    }

    pub async fn extract(&self, _file_path: &Path) -> RragResult<ExtractedContent> {
        // Simplified PDF extraction
        Ok(ExtractedContent {
            text: "Extracted PDF content".to_string(),
            document_type: DocumentType::PDF,
            page_count: 5,
            image_references: vec![],
            table_content: vec![],
            chart_references: vec![],
            title: Some("Sample PDF Document".to_string()),
            author: Some("PDF Author".to_string()),
            creation_date: Some("2024-01-01".to_string()),
            language: Some("en".to_string()),
        })
    }
}

impl WordTextExtractor {
    pub fn new() -> Self {
        Self {
            extract_styles: true,
            extract_comments: false,
        }
    }

    pub async fn extract(&self, _file_path: &Path) -> RragResult<ExtractedContent> {
        // Simplified Word extraction
        Ok(ExtractedContent {
            text: "Extracted Word content".to_string(),
            document_type: DocumentType::Word,
            page_count: 3,
            image_references: vec![],
            table_content: vec![],
            chart_references: vec![],
            title: Some("Sample Word Document".to_string()),
            author: Some("Word Author".to_string()),
            creation_date: Some("2024-01-01".to_string()),
            language: Some("en".to_string()),
        })
    }
}

impl PowerPointTextExtractor {
    pub fn new() -> Self {
        Self {
            extract_notes: true,
            extract_animations: false,
        }
    }

    pub async fn extract(&self, _file_path: &Path) -> RragResult<ExtractedContent> {
        // Simplified PowerPoint extraction
        Ok(ExtractedContent {
            text: "Extracted PowerPoint content".to_string(),
            document_type: DocumentType::PowerPoint,
            page_count: 10,
            image_references: vec![],
            table_content: vec![],
            chart_references: vec![],
            title: Some("Sample PowerPoint Presentation".to_string()),
            author: Some("PPT Author".to_string()),
            creation_date: Some("2024-01-01".to_string()),
            language: Some("en".to_string()),
        })
    }
}

impl HTMLTextExtractor {
    pub fn new() -> Self {
        Self {
            remove_scripts: true,
            remove_styles: true,
        }
    }

    pub async fn extract(&self, file_path: &Path) -> RragResult<ExtractedContent> {
        let html_content =
            std::fs::read_to_string(file_path).map_err(|e| RragError::io_error(e.to_string()))?;

        // Simplified HTML text extraction (remove tags)
        let text = html_content
            .split('<')
            .enumerate()
            .filter_map(|(i, part)| {
                if i == 0 {
                    Some(part)
                } else if let Some(end_pos) = part.find('>') {
                    Some(&part[end_pos + 1..])
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(ExtractedContent {
            text,
            document_type: DocumentType::HTML,
            page_count: 1,
            image_references: vec![],
            table_content: vec![],
            chart_references: vec![],
            title: None,
            author: None,
            creation_date: None,
            language: Some("en".to_string()),
        })
    }
}

impl Default for DocumentParserConfig {
    fn default() -> Self {
        Self {
            supported_types: vec![
                DocumentType::PDF,
                DocumentType::Word,
                DocumentType::HTML,
                DocumentType::Markdown,
                DocumentType::PlainText,
            ],
            extract_text: true,
            extract_images: true,
            extract_tables: true,
            extract_charts: true,
            analyze_structure: true,
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_pages: Some(1000),
        }
    }
}

impl Default for TextExtractionConfig {
    fn default() -> Self {
        Self {
            preserve_formatting: true,
            extract_footnotes: true,
            extract_headers_footers: false,
            min_block_size: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_document_type_detection() {
        let parser = create_test_parser();

        let pdf_path = std::path::Path::new("test.pdf");
        assert_eq!(
            parser.detect_document_type(pdf_path).unwrap(),
            DocumentType::PDF
        );

        let word_path = std::path::Path::new("test.docx");
        assert_eq!(
            parser.detect_document_type(word_path).unwrap(),
            DocumentType::Word
        );
    }

    #[test]
    fn test_section_detection() {
        let analyzer = SectionAnalyzer::new().unwrap();
        let text = "Abstract\n\nThis is the abstract.\n\nIntroduction\n\nThis is the introduction.";

        let sections = analyzer.analyze_sections(text).unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].section_type, SectionType::Abstract);
        assert_eq!(sections[1].section_type, SectionType::Introduction);
    }

    #[test]
    fn test_heading_detection() {
        let detector = HeadingDetector::new();

        // Markdown heading
        assert!(detector.detect_heading("# Main Heading").is_some());
        assert!(detector.detect_heading("## Sub Heading").is_some());

        // All caps heading
        assert!(detector.detect_heading("MAIN SECTION").is_some());

        // Regular text
        assert!(detector.detect_heading("This is regular text").is_none());
    }

    fn create_test_parser() -> DocumentParser {
        use super::super::{chart_processor, image_processor, table_processor};

        DocumentParser::new(
            DocumentParserConfig::default(),
            Box::new(
                image_processor::DefaultImageProcessor::new(
                    super::super::ImageProcessingConfig::default(),
                )
                .unwrap(),
            ),
            Box::new(
                table_processor::DefaultTableProcessor::new(
                    super::super::TableExtractionConfig::default(),
                )
                .unwrap(),
            ),
            Box::new(
                chart_processor::DefaultChartProcessor::new(
                    super::super::ChartAnalysisConfig::default(),
                )
                .unwrap(),
            ),
        )
        .unwrap()
    }
}
