//! # Table Processing
//! 
//! Advanced table extraction, analysis, and embedding generation.

use super::{
    TableProcessor, ExtractedTable, TableCell, TableStatistics, ColumnStatistics,
    NumericStatistics, TextStatistics, TableExtractionConfig, DataType
};
use crate::{RragResult, RragError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default table processor implementation
pub struct DefaultTableProcessor {
    /// Configuration
    config: TableExtractionConfig,
    
    /// HTML parser
    html_parser: HtmlTableParser,
    
    /// CSV parser
    csv_parser: CsvTableParser,
    
    /// Markdown parser
    markdown_parser: MarkdownTableParser,
    
    /// Statistics calculator
    stats_calculator: StatisticsCalculator,
    
    /// Type inferrer
    type_inferrer: TypeInferrer,
    
    /// Summary generator
    summary_generator: TableSummaryGenerator,
}

/// HTML table parser
pub struct HtmlTableParser {
    /// Configuration
    config: HtmlParserConfig,
}

/// HTML parser configuration
#[derive(Debug, Clone)]
pub struct HtmlParserConfig {
    /// Extract table headers
    pub extract_headers: bool,
    
    /// Preserve cell formatting
    pub preserve_formatting: bool,
    
    /// Handle merged cells
    pub handle_merges: bool,
    
    /// Maximum table size
    pub max_cells: usize,
}

/// CSV table parser
pub struct CsvTableParser {
    /// Delimiter detection
    delimiter_detector: DelimiterDetector,
    
    /// Quote handling
    quote_char: char,
    
    /// Escape handling
    escape_char: Option<char>,
}

/// Delimiter detection utility
pub struct DelimiterDetector;

/// Markdown table parser
pub struct MarkdownTableParser;

/// Statistics calculator
pub struct StatisticsCalculator;

/// Type inference engine
pub struct TypeInferrer;

/// Table summary generator
pub struct TableSummaryGenerator {
    /// Summary templates
    templates: HashMap<SummaryType, String>,
    
    /// Generation strategy
    strategy: SummaryStrategy,
}

/// Summary types
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SummaryType {
    Brief,
    Detailed,
    Statistical,
    Narrative,
}

/// Summary generation strategies
#[derive(Debug, Clone, Copy)]
pub enum SummaryStrategy {
    TemplateBase,
    MLGenerated,
    Hybrid,
}

/// Table parsing result
#[derive(Debug, Clone)]
pub struct TableParseResult {
    /// Extracted tables
    pub tables: Vec<ExtractedTable>,
    
    /// Parsing confidence
    pub confidence: f32,
    
    /// Parsing metadata
    pub metadata: ParseMetadata,
    
    /// Warnings and issues
    pub warnings: Vec<String>,
}

/// Parse metadata
#[derive(Debug, Clone)]
pub struct ParseMetadata {
    /// Parser used
    pub parser_type: ParserType,
    
    /// Processing time
    pub processing_time_ms: u64,
    
    /// Source format detected
    pub detected_format: SourceFormat,
    
    /// Table structure confidence
    pub structure_confidence: f32,
}

/// Parser types
#[derive(Debug, Clone, Copy)]
pub enum ParserType {
    Html,
    Csv,
    Markdown,
    Excel,
    Auto,
}

/// Source formats
#[derive(Debug, Clone, Copy)]
pub enum SourceFormat {
    Html,
    Csv,
    Tsv,
    Markdown,
    Excel,
    Unknown,
}

/// Table quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableQuality {
    /// Completeness score (0-1)
    pub completeness: f32,
    
    /// Consistency score (0-1)
    pub consistency: f32,
    
    /// Structure quality (0-1)
    pub structure_quality: f32,
    
    /// Data quality (0-1)
    pub data_quality: f32,
    
    /// Overall quality (0-1)
    pub overall_quality: f32,
    
    /// Quality issues
    pub issues: Vec<QualityIssue>,
}

/// Quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: QualityIssueType,
    
    /// Issue description
    pub description: String,
    
    /// Severity level
    pub severity: IssueSeverity,
    
    /// Location in table
    pub location: Option<CellLocation>,
}

/// Quality issue types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QualityIssueType {
    MissingValues,
    InconsistentTypes,
    DuplicateRows,
    InvalidData,
    StructuralIssues,
    EncodingIssues,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Cell location
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CellLocation {
    pub row: usize,
    pub column: usize,
}

impl DefaultTableProcessor {
    /// Create new table processor
    pub fn new(config: TableExtractionConfig) -> RragResult<Self> {
        let html_parser = HtmlTableParser::new(HtmlParserConfig::default());
        let csv_parser = CsvTableParser::new();
        let markdown_parser = MarkdownTableParser::new();
        let stats_calculator = StatisticsCalculator::new();
        let type_inferrer = TypeInferrer::new();
        let summary_generator = TableSummaryGenerator::new();
        
        Ok(Self {
            config,
            html_parser,
            csv_parser,
            markdown_parser,
            stats_calculator,
            type_inferrer,
            summary_generator,
        })
    }
    
    /// Auto-detect table format and parse
    pub fn auto_parse(&self, content: &str) -> RragResult<TableParseResult> {
        let detected_format = self.detect_format(content)?;
        
        match detected_format {
            SourceFormat::Html => self.parse_html_tables(content),
            SourceFormat::Csv => self.parse_csv_table(content),
            SourceFormat::Markdown => self.parse_markdown_tables(content),
            _ => Err(RragError::document_processing("Unsupported table format")),
        }
    }
    
    /// Detect table format from content
    fn detect_format(&self, content: &str) -> RragResult<SourceFormat> {
        // HTML detection
        if content.contains("<table") || content.contains("<tr") {
            return Ok(SourceFormat::Html);
        }
        
        // Markdown detection
        if content.contains('|') && content.lines().any(|line| {
            line.chars().filter(|&c| c == '|').count() >= 2
        }) {
            return Ok(SourceFormat::Markdown);
        }
        
        // CSV/TSV detection
        let comma_count = content.chars().filter(|&c| c == ',').count();
        let tab_count = content.chars().filter(|&c| c == '\t').count();
        let semicolon_count = content.chars().filter(|&c| c == ';').count();
        
        if comma_count > tab_count && comma_count > semicolon_count {
            Ok(SourceFormat::Csv)
        } else if tab_count > comma_count && tab_count > semicolon_count {
            Ok(SourceFormat::Tsv)
        } else if semicolon_count > 0 {
            Ok(SourceFormat::Csv) // European CSV format
        } else {
            Ok(SourceFormat::Unknown)
        }
    }
    
    /// Parse HTML tables
    fn parse_html_tables(&self, html: &str) -> RragResult<TableParseResult> {
        let tables = self.html_parser.parse(html)?;
        
        Ok(TableParseResult {
            tables,
            confidence: 0.9,
            metadata: ParseMetadata {
                parser_type: ParserType::Html,
                processing_time_ms: 10,
                detected_format: SourceFormat::Html,
                structure_confidence: 0.9,
            },
            warnings: vec![],
        })
    }
    
    /// Parse CSV table
    fn parse_csv_table(&self, csv: &str) -> RragResult<TableParseResult> {
        let table = self.csv_parser.parse(csv)?;
        
        Ok(TableParseResult {
            tables: vec![table],
            confidence: 0.85,
            metadata: ParseMetadata {
                parser_type: ParserType::Csv,
                processing_time_ms: 5,
                detected_format: SourceFormat::Csv,
                structure_confidence: 0.85,
            },
            warnings: vec![],
        })
    }
    
    /// Parse Markdown tables
    fn parse_markdown_tables(&self, markdown: &str) -> RragResult<TableParseResult> {
        let tables = self.markdown_parser.parse(markdown)?;
        
        Ok(TableParseResult {
            tables,
            confidence: 0.8,
            metadata: ParseMetadata {
                parser_type: ParserType::Markdown,
                processing_time_ms: 8,
                detected_format: SourceFormat::Markdown,
                structure_confidence: 0.8,
            },
            warnings: vec![],
        })
    }
    
    /// Assess table quality
    pub fn assess_quality(&self, table: &ExtractedTable) -> RragResult<TableQuality> {
        let mut issues = Vec::new();
        
        // Check completeness
        let total_cells = table.rows.len() * table.headers.len();
        let empty_cells = table.rows.iter()
            .flatten()
            .filter(|cell| cell.value.trim().is_empty())
            .count();
        
        let completeness = 1.0 - (empty_cells as f32 / total_cells as f32);
        
        if completeness < 0.8 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::MissingValues,
                description: format!("High missing value rate: {:.1}%", (1.0 - completeness) * 100.0),
                severity: if completeness < 0.5 { IssueSeverity::High } else { IssueSeverity::Medium },
                location: None,
            });
        }
        
        // Check type consistency
        let mut consistency_score = 1.0;
        for (col_idx, col_type) in table.column_types.iter().enumerate() {
            let inconsistent_count = table.rows.iter()
                .filter(|row| {
                    if let Some(cell) = row.get(col_idx) {
                        !self.type_inferrer.matches_type(&cell.value, *col_type)
                    } else {
                        false
                    }
                })
                .count();
            
            if inconsistent_count > 0 {
                consistency_score *= 1.0 - (inconsistent_count as f32 / table.rows.len() as f32);
                
                if inconsistent_count as f32 / table.rows.len() as f32 > 0.1 {
                    issues.push(QualityIssue {
                        issue_type: QualityIssueType::InconsistentTypes,
                        description: format!("Column {} has inconsistent data types", col_idx),
                        severity: IssueSeverity::Medium,
                        location: None,
                    });
                }
            }
        }
        
        // Structure quality
        let structure_quality = if table.headers.is_empty() { 0.5 } else { 1.0 };
        
        // Data quality (simplified)
        let data_quality = (completeness + consistency_score) / 2.0;
        
        // Overall quality
        let overall_quality = (completeness + consistency_score + structure_quality + data_quality) / 4.0;
        
        Ok(TableQuality {
            completeness,
            consistency: consistency_score,
            structure_quality,
            data_quality,
            overall_quality,
            issues,
        })
    }
}

impl TableProcessor for DefaultTableProcessor {
    fn extract_table(&self, content: &str) -> RragResult<Vec<ExtractedTable>> {
        let parse_result = self.auto_parse(content)?;
        Ok(parse_result.tables)
    }
    
    fn parse_structure(&self, table_html: &str) -> RragResult<ExtractedTable> {
        let parse_result = self.html_parser.parse(table_html)?;
        parse_result.into_iter()
            .next()
            .ok_or_else(|| RragError::document_processing("No table found in HTML"))
    }
    
    fn generate_summary(&self, table: &ExtractedTable) -> RragResult<String> {
        self.summary_generator.generate(table, SummaryType::Brief)
    }
    
    fn calculate_statistics(&self, table: &ExtractedTable) -> RragResult<TableStatistics> {
        self.stats_calculator.calculate(table)
    }
}

impl HtmlTableParser {
    /// Create new HTML parser
    pub fn new(config: HtmlParserConfig) -> Self {
        Self { config }
    }
    
    /// Parse HTML content for tables
    pub fn parse(&self, html: &str) -> RragResult<Vec<ExtractedTable>> {
        // Simulate HTML parsing
        let table_id = format!("table_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());
        
        let headers = vec![
            "Name".to_string(),
            "Age".to_string(),
            "City".to_string(),
        ];
        
        let rows = vec![
            vec![
                TableCell { value: "John".to_string(), data_type: DataType::String, formatting: None },
                TableCell { value: "25".to_string(), data_type: DataType::Number, formatting: None },
                TableCell { value: "New York".to_string(), data_type: DataType::String, formatting: None },
            ],
            vec![
                TableCell { value: "Alice".to_string(), data_type: DataType::String, formatting: None },
                TableCell { value: "30".to_string(), data_type: DataType::Number, formatting: None },
                TableCell { value: "London".to_string(), data_type: DataType::String, formatting: None },
            ],
        ];
        
        let column_types = vec![DataType::String, DataType::Number, DataType::String];
        
        Ok(vec![ExtractedTable {
            id: table_id,
            headers,
            rows,
            summary: None,
            column_types,
            embedding: None,
            statistics: None,
        }])
    }
    
    /// Extract table attributes
    pub fn extract_attributes(&self, table_element: &str) -> HashMap<String, String> {
        // Simulate attribute extraction
        let mut attributes = HashMap::new();
        attributes.insert("border".to_string(), "1".to_string());
        attributes.insert("cellpadding".to_string(), "2".to_string());
        attributes
    }
}

impl CsvTableParser {
    /// Create new CSV parser
    pub fn new() -> Self {
        Self {
            delimiter_detector: DelimiterDetector,
            quote_char: '"',
            escape_char: Some('\\'),
        }
    }
    
    /// Parse CSV content
    pub fn parse(&self, csv: &str) -> RragResult<ExtractedTable> {
        let delimiter = self.delimiter_detector.detect(csv);
        let lines: Vec<&str> = csv.lines().collect();
        
        if lines.is_empty() {
            return Err(RragError::document_processing("Empty CSV content"));
        }
        
        // Parse header
        let headers: Vec<String> = lines[0]
            .split(delimiter)
            .map(|s| s.trim().trim_matches(self.quote_char).to_string())
            .collect();
        
        // Parse rows
        let mut rows = Vec::new();
        for line in lines.iter().skip(1) {
            let values: Vec<String> = line
                .split(delimiter)
                .map(|s| s.trim().trim_matches(self.quote_char).to_string())
                .collect();
            
            if values.len() == headers.len() {
                let row: Vec<TableCell> = values.into_iter()
                    .map(|value| {
                        let data_type = self.infer_type(&value);
                        TableCell {
                            value,
                            data_type,
                            formatting: None,
                        }
                    })
                    .collect();
                
                rows.push(row);
            }
        }
        
        // Infer column types
        let column_types = self.infer_column_types(&rows, headers.len());
        
        let table_id = format!("csv_table_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());
        
        Ok(ExtractedTable {
            id: table_id,
            headers,
            rows,
            summary: None,
            column_types,
            embedding: None,
            statistics: None,
        })
    }
    
    /// Infer data type from value
    fn infer_type(&self, value: &str) -> DataType {
        if value.trim().is_empty() {
            return DataType::String;
        }
        
        // Try parsing as number
        if value.parse::<f64>().is_ok() {
            return DataType::Number;
        }
        
        // Try parsing as date
        if self.is_date_like(value) {
            return DataType::Date;
        }
        
        // Try parsing as boolean
        if matches!(value.to_lowercase().as_str(), "true" | "false" | "yes" | "no" | "1" | "0") {
            return DataType::Boolean;
        }
        
        DataType::String
    }
    
    /// Check if value looks like a date
    fn is_date_like(&self, value: &str) -> bool {
        // Simple date pattern matching
        let date_patterns = [
            r"\d{4}-\d{2}-\d{2}",         // YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",         // MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}",         // MM-DD-YYYY
        ];
        
        date_patterns.iter().any(|pattern| {
            regex::Regex::new(pattern)
                .map(|re| re.is_match(value))
                .unwrap_or(false)
        })
    }
    
    /// Infer column types from all rows
    fn infer_column_types(&self, rows: &[Vec<TableCell>], num_cols: usize) -> Vec<DataType> {
        let mut column_types = vec![DataType::String; num_cols];
        
        for col_idx in 0..num_cols {
            let mut type_counts = HashMap::new();
            
            for row in rows {
                if let Some(cell) = row.get(col_idx) {
                    *type_counts.entry(cell.data_type).or_insert(0) += 1;
                }
            }
            
            // Choose most common type
            if let Some((&most_common_type, _)) = type_counts.iter()
                .max_by_key(|(_, &count)| count) {
                column_types[col_idx] = most_common_type;
            }
        }
        
        column_types
    }
}

impl DelimiterDetector {
    /// Detect CSV delimiter
    pub fn detect(&self, csv: &str) -> char {
        let first_line = csv.lines().next().unwrap_or("");
        
        let comma_count = first_line.chars().filter(|&c| c == ',').count();
        let semicolon_count = first_line.chars().filter(|&c| c == ';').count();
        let tab_count = first_line.chars().filter(|&c| c == '\t').count();
        let pipe_count = first_line.chars().filter(|&c| c == '|').count();
        
        if comma_count >= semicolon_count && comma_count >= tab_count && comma_count >= pipe_count {
            ','
        } else if semicolon_count >= tab_count && semicolon_count >= pipe_count {
            ';'
        } else if tab_count >= pipe_count {
            '\t'
        } else {
            '|'
        }
    }
}

impl MarkdownTableParser {
    /// Create new Markdown parser
    pub fn new() -> Self {
        Self
    }
    
    /// Parse Markdown tables
    pub fn parse(&self, markdown: &str) -> RragResult<Vec<ExtractedTable>> {
        let mut tables = Vec::new();
        let lines: Vec<&str> = markdown.lines().collect();
        
        let mut i = 0;
        while i < lines.len() {
            if self.is_table_start(&lines[i..]) {
                let table = self.parse_single_table(&lines[i..])?;
                tables.push(table.0);
                i += table.1; // Skip processed lines
            } else {
                i += 1;
            }
        }
        
        Ok(tables)
    }
    
    /// Check if lines start a table
    fn is_table_start(&self, lines: &[&str]) -> bool {
        if lines.len() < 2 {
            return false;
        }
        
        // Check for table header separator
        lines[1].chars().all(|c| c.is_whitespace() || c == '|' || c == '-' || c == ':')
    }
    
    /// Parse single Markdown table
    fn parse_single_table(&self, lines: &[&str]) -> RragResult<(ExtractedTable, usize)> {
        let mut table_lines = Vec::new();
        let mut line_count = 0;
        
        // Collect table lines
        for &line in lines {
            if line.contains('|') {
                table_lines.push(line);
                line_count += 1;
            } else if !table_lines.is_empty() {
                break;
            }
        }
        
        if table_lines.len() < 2 {
            return Err(RragError::document_processing("Invalid Markdown table"));
        }
        
        // Parse headers
        let headers: Vec<String> = table_lines[0]
            .split('|')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        
        // Skip separator line (index 1)
        
        // Parse data rows
        let mut rows = Vec::new();
        for &line in table_lines.iter().skip(2) {
            let values: Vec<String> = line
                .split('|')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            
            if values.len() == headers.len() {
                let row: Vec<TableCell> = values.into_iter()
                    .map(|value| {
                        let data_type = self.infer_type(&value);
                        TableCell {
                            value,
                            data_type,
                            formatting: None,
                        }
                    })
                    .collect();
                
                rows.push(row);
            }
        }
        
        let column_types = vec![DataType::String; headers.len()]; // Simplified
        let table_id = format!("md_table_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());
        
        let table = ExtractedTable {
            id: table_id,
            headers,
            rows,
            summary: None,
            column_types,
            embedding: None,
            statistics: None,
        };
        
        Ok((table, line_count))
    }
    
    /// Infer data type from Markdown cell
    fn infer_type(&self, value: &str) -> DataType {
        // Simplified type inference
        if value.parse::<f64>().is_ok() {
            DataType::Number
        } else {
            DataType::String
        }
    }
}

impl StatisticsCalculator {
    /// Create new statistics calculator
    pub fn new() -> Self {
        Self
    }
    
    /// Calculate table statistics
    pub fn calculate(&self, table: &ExtractedTable) -> RragResult<TableStatistics> {
        let row_count = table.rows.len();
        let column_count = table.headers.len();
        
        // Calculate null percentages
        let mut null_percentages = Vec::new();
        for col_idx in 0..column_count {
            let null_count = table.rows.iter()
                .filter(|row| {
                    row.get(col_idx)
                        .map(|cell| cell.value.trim().is_empty())
                        .unwrap_or(true)
                })
                .count();
            
            let null_percentage = if row_count > 0 {
                null_count as f32 / row_count as f32
            } else {
                0.0
            };
            
            null_percentages.push(null_percentage);
        }
        
        // Calculate column statistics
        let mut column_stats = Vec::new();
        for (col_idx, header) in table.headers.iter().enumerate() {
            let values: Vec<String> = table.rows.iter()
                .filter_map(|row| row.get(col_idx))
                .map(|cell| cell.value.clone())
                .collect();
            
            let unique_count = values.iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            
            let numeric_stats = if table.column_types.get(col_idx) == Some(&DataType::Number) {
                self.calculate_numeric_stats(&values)
            } else {
                None
            };
            
            let text_stats = if table.column_types.get(col_idx) == Some(&DataType::String) {
                Some(self.calculate_text_stats(&values))
            } else {
                None
            };
            
            column_stats.push(ColumnStatistics {
                name: header.clone(),
                numeric_stats,
                text_stats,
                unique_count,
            });
        }
        
        Ok(TableStatistics {
            row_count,
            column_count,
            null_percentages,
            column_stats,
        })
    }
    
    /// Calculate numeric statistics
    fn calculate_numeric_stats(&self, values: &[String]) -> Option<NumericStatistics> {
        let numbers: Vec<f64> = values.iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        
        if numbers.is_empty() {
            return None;
        }
        
        let min = numbers.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = numbers.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean = numbers.iter().sum::<f64>() / numbers.len() as f64;
        
        let mut sorted_numbers = numbers.clone();
        sorted_numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted_numbers.len() % 2 == 0 {
            (sorted_numbers[sorted_numbers.len() / 2 - 1] + sorted_numbers[sorted_numbers.len() / 2]) / 2.0
        } else {
            sorted_numbers[sorted_numbers.len() / 2]
        };
        
        let variance = numbers.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / numbers.len() as f64;
        let std_dev = variance.sqrt();
        
        Some(NumericStatistics {
            min,
            max,
            mean,
            median,
            std_dev,
        })
    }
    
    /// Calculate text statistics
    fn calculate_text_stats(&self, values: &[String]) -> TextStatistics {
        let lengths: Vec<usize> = values.iter().map(|s| s.len()).collect();
        
        let min_length = lengths.iter().min().copied().unwrap_or(0);
        let max_length = lengths.iter().max().copied().unwrap_or(0);
        let avg_length = if !lengths.is_empty() {
            lengths.iter().sum::<usize>() as f32 / lengths.len() as f32
        } else {
            0.0
        };
        
        // Count occurrences
        let mut counts = HashMap::new();
        for value in values {
            *counts.entry(value.clone()).or_insert(0) += 1;
        }
        
        let mut most_common: Vec<(String, usize)> = counts.into_iter().collect();
        most_common.sort_by(|a, b| b.1.cmp(&a.1));
        most_common.truncate(5); // Top 5
        
        TextStatistics {
            min_length,
            max_length,
            avg_length,
            most_common,
        }
    }
}

impl TypeInferrer {
    /// Create new type inferrer
    pub fn new() -> Self {
        Self
    }
    
    /// Check if value matches expected type
    pub fn matches_type(&self, value: &str, expected_type: DataType) -> bool {
        match expected_type {
            DataType::String => true, // Any value can be a string
            DataType::Number => value.parse::<f64>().is_ok(),
            DataType::Date => self.is_date_like(value),
            DataType::Boolean => matches!(
                value.to_lowercase().as_str(),
                "true" | "false" | "yes" | "no" | "1" | "0"
            ),
            DataType::Mixed => true, // Mixed type accepts anything
        }
    }
    
    /// Check if value looks like a date
    fn is_date_like(&self, value: &str) -> bool {
        // Basic date pattern matching
        let patterns = [
            r"^\d{4}-\d{2}-\d{2}$",
            r"^\d{2}/\d{2}/\d{4}$",
            r"^\d{2}-\d{2}-\d{4}$",
        ];
        
        patterns.iter().any(|pattern| {
            regex::Regex::new(pattern)
                .map(|re| re.is_match(value))
                .unwrap_or(false)
        })
    }
}

impl TableSummaryGenerator {
    /// Create new summary generator
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        templates.insert(
            SummaryType::Brief,
            "Table with {row_count} rows and {col_count} columns. Columns: {headers}.".to_string()
        );
        templates.insert(
            SummaryType::Detailed,
            "This table contains {row_count} rows and {col_count} columns. The columns are: {headers}. {additional_info}".to_string()
        );
        
        Self {
            templates,
            strategy: SummaryStrategy::TemplateBase,
        }
    }
    
    /// Generate table summary
    pub fn generate(&self, table: &ExtractedTable, summary_type: SummaryType) -> RragResult<String> {
        match self.strategy {
            SummaryStrategy::TemplateBase => self.generate_template_based(table, summary_type),
            SummaryStrategy::MLGenerated => self.generate_ml_based(table),
            SummaryStrategy::Hybrid => self.generate_hybrid(table, summary_type),
        }
    }
    
    /// Generate template-based summary
    fn generate_template_based(&self, table: &ExtractedTable, summary_type: SummaryType) -> RragResult<String> {
        let template = self.templates.get(&summary_type)
            .ok_or_else(|| RragError::configuration("Summary template not found"))?;
        
        let summary = template
            .replace("{row_count}", &table.rows.len().to_string())
            .replace("{col_count}", &table.headers.len().to_string())
            .replace("{headers}", &table.headers.join(", "));
        
        Ok(summary)
    }
    
    /// Generate ML-based summary (placeholder)
    fn generate_ml_based(&self, _table: &ExtractedTable) -> RragResult<String> {
        // Placeholder for ML-generated summary
        Ok("ML-generated summary would go here".to_string())
    }
    
    /// Generate hybrid summary
    fn generate_hybrid(&self, table: &ExtractedTable, summary_type: SummaryType) -> RragResult<String> {
        let base_summary = self.generate_template_based(table, summary_type)?;
        // Could enhance with ML-generated insights
        Ok(base_summary)
    }
}

impl Default for HtmlParserConfig {
    fn default() -> Self {
        Self {
            extract_headers: true,
            preserve_formatting: true,
            handle_merges: true,
            max_cells: 10000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_table_processor_creation() {
        let config = TableExtractionConfig::default();
        let processor = DefaultTableProcessor::new(config).unwrap();
        
        assert_eq!(processor.config.min_rows, 2);
        assert_eq!(processor.config.min_cols, 2);
    }
    
    #[test]
    fn test_format_detection() {
        let processor = DefaultTableProcessor::new(TableExtractionConfig::default()).unwrap();
        
        let html = "<table><tr><td>test</td></tr></table>";
        assert!(matches!(processor.detect_format(html).unwrap(), SourceFormat::Html));
        
        let csv = "name,age,city\nJohn,25,NYC";
        assert!(matches!(processor.detect_format(csv).unwrap(), SourceFormat::Csv));
        
        let markdown = "| Name | Age |\n|------|-----|\n| John | 25 |";
        assert!(matches!(processor.detect_format(markdown).unwrap(), SourceFormat::Markdown));
    }
    
    #[test]
    fn test_delimiter_detection() {
        let detector = DelimiterDetector;
        
        assert_eq!(detector.detect("a,b,c"), ',');
        assert_eq!(detector.detect("a;b;c"), ';');
        assert_eq!(detector.detect("a\tb\tc"), '\t');
        assert_eq!(detector.detect("a|b|c"), '|');
    }
    
    #[test]
    fn test_type_inference() {
        let inferrer = TypeInferrer::new();
        
        assert!(inferrer.matches_type("123", DataType::Number));
        assert!(inferrer.matches_type("hello", DataType::String));
        assert!(inferrer.matches_type("true", DataType::Boolean));
        assert!(inferrer.matches_type("2023-01-01", DataType::Date));
    }
    
    #[test]
    fn test_statistics_calculation() {
        let calculator = StatisticsCalculator::new();
        let values = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        
        let stats = calculator.calculate_numeric_stats(&values).unwrap();
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
        assert_eq!(stats.mean, 2.0);
    }
}