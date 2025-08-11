//! # Chart Processing
//! 
//! Advanced chart analysis, data extraction, and trend analysis.

use super::{
    ChartProcessor, AnalyzedChart, ChartType, ChartAxes, DataPoint, TrendAnalysis,
    TrendDirection, Seasonality, ChartAnalysisConfig
};
use crate::{RragResult, RragError};
use std::path::Path;

/// Default chart processor implementation
pub struct DefaultChartProcessor {
    /// Configuration
    config: ChartAnalysisConfig,
    
    /// Chart type classifier
    type_classifier: ChartTypeClassifier,
    
    /// Data extractor
    data_extractor: ChartDataExtractor,
    
    /// Trend analyzer
    trend_analyzer: TrendAnalyzer,
    
    /// Description generator
    description_generator: ChartDescriptionGenerator,
}

/// Chart type classifier
pub struct ChartTypeClassifier {
    /// Classification models
    models: Vec<ClassificationModel>,
}

/// Chart data extractor
pub struct ChartDataExtractor {
    /// OCR engine for text extraction
    ocr_enabled: bool,
    
    /// Color analysis for data series
    color_analysis: bool,
    
    /// Shape detection for markers
    shape_detection: bool,
}

/// Trend analyzer for time series and patterns
pub struct TrendAnalyzer {
    /// Minimum points for trend analysis
    min_points: usize,
    
    /// Smoothing window size
    smoothing_window: usize,
    
    /// Seasonality detection
    seasonality_detection: bool,
}

/// Chart description generator
pub struct ChartDescriptionGenerator {
    /// Template-based generation
    templates: std::collections::HashMap<ChartType, String>,
    
    /// Natural language generation
    nlg_enabled: bool,
}

/// Classification model for chart types
#[derive(Debug, Clone)]
pub struct ClassificationModel {
    /// Model type
    model_type: ModelType,
    
    /// Confidence threshold
    confidence_threshold: f32,
    
    /// Feature extractors
    features: Vec<FeatureType>,
}

/// Model types for classification
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    CNN,
    SVM,
    RandomForest,
    Ensemble,
}

/// Feature types for classification
#[derive(Debug, Clone, Copy)]
pub enum FeatureType {
    ColorHistogram,
    EdgeDetection,
    ShapeFeatures,
    TextFeatures,
    LayoutFeatures,
}

/// Chart analysis result
#[derive(Debug, Clone)]
pub struct ChartAnalysisResult {
    /// Identified chart type
    pub chart_type: ChartType,
    
    /// Classification confidence
    pub confidence: f32,
    
    /// Extracted data points
    pub data_points: Vec<DataPoint>,
    
    /// Chart elements
    pub elements: ChartElements,
    
    /// Visual properties
    pub visual_properties: VisualProperties,
}

/// Chart elements
#[derive(Debug, Clone)]
pub struct ChartElements {
    /// Chart title
    pub title: Option<String>,
    
    /// Axis labels
    pub axes: ChartAxes,
    
    /// Legend entries
    pub legend: Vec<LegendEntry>,
    
    /// Data series
    pub series: Vec<DataSeries>,
    
    /// Annotations
    pub annotations: Vec<ChartAnnotation>,
}

/// Legend entry
#[derive(Debug, Clone)]
pub struct LegendEntry {
    /// Legend text
    pub text: String,
    
    /// Associated color
    pub color: Option<(u8, u8, u8)>,
    
    /// Symbol/marker type
    pub symbol: Option<MarkerType>,
}

/// Data series in chart
#[derive(Debug, Clone)]
pub struct DataSeries {
    /// Series name
    pub name: String,
    
    /// Data points
    pub points: Vec<DataPoint>,
    
    /// Series color
    pub color: Option<(u8, u8, u8)>,
    
    /// Line style
    pub line_style: Option<LineStyle>,
}

/// Chart annotation
#[derive(Debug, Clone)]
pub struct ChartAnnotation {
    /// Annotation text
    pub text: String,
    
    /// Position
    pub position: (f64, f64),
    
    /// Annotation type
    pub annotation_type: AnnotationType,
}

/// Visual properties of chart
#[derive(Debug, Clone)]
pub struct VisualProperties {
    /// Chart area
    pub chart_area: ChartArea,
    
    /// Color scheme
    pub color_scheme: ColorScheme,
    
    /// Typography
    pub typography: Typography,
    
    /// Grid properties
    pub grid: Option<GridProperties>,
}

/// Chart area dimensions
#[derive(Debug, Clone)]
pub struct ChartArea {
    /// Chart bounds
    pub bounds: (f64, f64, f64, f64), // (x, y, width, height)
    
    /// Plot area
    pub plot_area: (f64, f64, f64, f64),
    
    /// Margins
    pub margins: (f64, f64, f64, f64), // (top, right, bottom, left)
}

/// Color scheme analysis
#[derive(Debug, Clone)]
pub struct ColorScheme {
    /// Primary colors
    pub primary_colors: Vec<(u8, u8, u8)>,
    
    /// Color palette type
    pub palette_type: PaletteType,
    
    /// Color accessibility score
    pub accessibility_score: f32,
}

/// Typography analysis
#[derive(Debug, Clone)]
pub struct Typography {
    /// Title font info
    pub title_font: Option<FontInfo>,
    
    /// Axis font info
    pub axis_font: Option<FontInfo>,
    
    /// Legend font info
    pub legend_font: Option<FontInfo>,
    
    /// Overall readability score
    pub readability_score: f32,
}

/// Font information
#[derive(Debug, Clone)]
pub struct FontInfo {
    /// Font family
    pub family: String,
    
    /// Font size
    pub size: f32,
    
    /// Font weight
    pub weight: FontWeight,
    
    /// Font color
    pub color: (u8, u8, u8),
}

/// Grid properties
#[derive(Debug, Clone)]
pub struct GridProperties {
    /// Grid type
    pub grid_type: GridType,
    
    /// Grid color
    pub color: (u8, u8, u8),
    
    /// Grid opacity
    pub opacity: f32,
    
    /// Grid line count
    pub line_count: (usize, usize), // (horizontal, vertical)
}

/// Marker types
#[derive(Debug, Clone, Copy)]
pub enum MarkerType {
    Circle,
    Square,
    Triangle,
    Diamond,
    Plus,
    Cross,
    Star,
}

/// Line styles
#[derive(Debug, Clone, Copy)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

/// Annotation types
#[derive(Debug, Clone, Copy)]
pub enum AnnotationType {
    Label,
    Arrow,
    Callout,
    Highlight,
}

/// Color palette types
#[derive(Debug, Clone, Copy)]
pub enum PaletteType {
    Sequential,
    Diverging,
    Categorical,
    Monochromatic,
}

/// Font weights
#[derive(Debug, Clone, Copy)]
pub enum FontWeight {
    Thin,
    Light,
    Regular,
    Medium,
    Bold,
    ExtraBold,
}

/// Grid types
#[derive(Debug, Clone, Copy)]
pub enum GridType {
    Major,
    Minor,
    Both,
    None,
}

impl DefaultChartProcessor {
    /// Create new chart processor
    pub fn new(config: ChartAnalysisConfig) -> RragResult<Self> {
        let type_classifier = ChartTypeClassifier::new()?;
        let data_extractor = ChartDataExtractor::new(true, true, true);
        let trend_analyzer = TrendAnalyzer::new(5, 3, true);
        let description_generator = ChartDescriptionGenerator::new();
        
        Ok(Self {
            config,
            type_classifier,
            data_extractor,
            trend_analyzer,
            description_generator,
        })
    }
    
    /// Comprehensive chart analysis
    pub fn analyze_comprehensive(&self, image_path: &Path) -> RragResult<ChartAnalysisResult> {
        // Classify chart type
        let (chart_type, confidence) = self.type_classifier.classify(image_path)?;
        
        // Extract data points
        let data_points = self.data_extractor.extract(image_path, chart_type)?;
        
        // Analyze chart elements
        let elements = self.analyze_elements(image_path, chart_type)?;
        
        // Analyze visual properties
        let visual_properties = self.analyze_visual_properties(image_path)?;
        
        Ok(ChartAnalysisResult {
            chart_type,
            confidence,
            data_points,
            elements,
            visual_properties,
        })
    }
    
    /// Analyze chart elements
    fn analyze_elements(&self, image_path: &Path, chart_type: ChartType) -> RragResult<ChartElements> {
        // Extract title
        let title = self.extract_title(image_path)?;
        
        // Extract axes information
        let axes = self.extract_axes(image_path)?;
        
        // Extract legend
        let legend = self.extract_legend(image_path)?;
        
        // Extract data series
        let series = self.extract_series(image_path, chart_type)?;
        
        // Extract annotations
        let annotations = self.extract_annotations(image_path)?;
        
        Ok(ChartElements {
            title,
            axes,
            legend,
            series,
            annotations,
        })
    }
    
    /// Extract chart title
    fn extract_title(&self, _image_path: &Path) -> RragResult<Option<String>> {
        // Simulate title extraction using OCR
        Ok(Some("Sample Chart Title".to_string()))
    }
    
    /// Extract axis information
    fn extract_axes(&self, _image_path: &Path) -> RragResult<ChartAxes> {
        // Simulate axis extraction
        Ok(ChartAxes {
            x_label: Some("Time".to_string()),
            y_label: Some("Value".to_string()),
            x_range: Some((0.0, 100.0)),
            y_range: Some((0.0, 50.0)),
        })
    }
    
    /// Extract legend information
    fn extract_legend(&self, _image_path: &Path) -> RragResult<Vec<LegendEntry>> {
        // Simulate legend extraction
        Ok(vec![
            LegendEntry {
                text: "Series 1".to_string(),
                color: Some((255, 0, 0)),
                symbol: Some(MarkerType::Circle),
            },
            LegendEntry {
                text: "Series 2".to_string(),
                color: Some((0, 255, 0)),
                symbol: Some(MarkerType::Square),
            },
        ])
    }
    
    /// Extract data series
    fn extract_series(&self, _image_path: &Path, chart_type: ChartType) -> RragResult<Vec<DataSeries>> {
        // Simulate series extraction based on chart type
        match chart_type {
            ChartType::Line => self.extract_line_series(),
            ChartType::Bar => self.extract_bar_series(),
            ChartType::Pie => self.extract_pie_series(),
            ChartType::Scatter => self.extract_scatter_series(),
            _ => Ok(vec![]),
        }
    }
    
    /// Extract line chart series
    fn extract_line_series(&self) -> RragResult<Vec<DataSeries>> {
        Ok(vec![
            DataSeries {
                name: "Series 1".to_string(),
                points: vec![
                    DataPoint { x: 0.0, y: 10.0, label: None, series: Some("Series 1".to_string()) },
                    DataPoint { x: 1.0, y: 15.0, label: None, series: Some("Series 1".to_string()) },
                    DataPoint { x: 2.0, y: 12.0, label: None, series: Some("Series 1".to_string()) },
                ],
                color: Some((255, 0, 0)),
                line_style: Some(LineStyle::Solid),
            }
        ])
    }
    
    /// Extract bar chart series
    fn extract_bar_series(&self) -> RragResult<Vec<DataSeries>> {
        Ok(vec![
            DataSeries {
                name: "Categories".to_string(),
                points: vec![
                    DataPoint { x: 0.0, y: 20.0, label: Some("Category A".to_string()), series: None },
                    DataPoint { x: 1.0, y: 35.0, label: Some("Category B".to_string()), series: None },
                    DataPoint { x: 2.0, y: 25.0, label: Some("Category C".to_string()), series: None },
                ],
                color: Some((0, 100, 200)),
                line_style: None,
            }
        ])
    }
    
    /// Extract pie chart series
    fn extract_pie_series(&self) -> RragResult<Vec<DataSeries>> {
        Ok(vec![
            DataSeries {
                name: "Pie Slices".to_string(),
                points: vec![
                    DataPoint { x: 0.0, y: 40.0, label: Some("Slice A".to_string()), series: None },
                    DataPoint { x: 1.0, y: 30.0, label: Some("Slice B".to_string()), series: None },
                    DataPoint { x: 2.0, y: 30.0, label: Some("Slice C".to_string()), series: None },
                ],
                color: None,
                line_style: None,
            }
        ])
    }
    
    /// Extract scatter plot series
    fn extract_scatter_series(&self) -> RragResult<Vec<DataSeries>> {
        Ok(vec![
            DataSeries {
                name: "Scatter Points".to_string(),
                points: vec![
                    DataPoint { x: 5.0, y: 10.0, label: None, series: None },
                    DataPoint { x: 15.0, y: 25.0, label: None, series: None },
                    DataPoint { x: 25.0, y: 20.0, label: None, series: None },
                ],
                color: Some((100, 100, 100)),
                line_style: None,
            }
        ])
    }
    
    /// Extract annotations
    fn extract_annotations(&self, _image_path: &Path) -> RragResult<Vec<ChartAnnotation>> {
        // Simulate annotation extraction
        Ok(vec![])
    }
    
    /// Analyze visual properties
    fn analyze_visual_properties(&self, _image_path: &Path) -> RragResult<VisualProperties> {
        // Simulate visual property analysis
        Ok(VisualProperties {
            chart_area: ChartArea {
                bounds: (0.0, 0.0, 800.0, 600.0),
                plot_area: (100.0, 100.0, 600.0, 400.0),
                margins: (50.0, 50.0, 50.0, 100.0),
            },
            color_scheme: ColorScheme {
                primary_colors: vec![(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                palette_type: PaletteType::Categorical,
                accessibility_score: 0.8,
            },
            typography: Typography {
                title_font: Some(FontInfo {
                    family: "Arial".to_string(),
                    size: 16.0,
                    weight: FontWeight::Bold,
                    color: (0, 0, 0),
                }),
                axis_font: Some(FontInfo {
                    family: "Arial".to_string(),
                    size: 12.0,
                    weight: FontWeight::Regular,
                    color: (100, 100, 100),
                }),
                legend_font: Some(FontInfo {
                    family: "Arial".to_string(),
                    size: 10.0,
                    weight: FontWeight::Regular,
                    color: (0, 0, 0),
                }),
                readability_score: 0.9,
            },
            grid: Some(GridProperties {
                grid_type: GridType::Major,
                color: (200, 200, 200),
                opacity: 0.3,
                line_count: (5, 10),
            }),
        })
    }
}

impl ChartProcessor for DefaultChartProcessor {
    fn analyze_chart(&self, image_path: &Path) -> RragResult<AnalyzedChart> {
        let analysis = self.analyze_comprehensive(image_path)?;
        
        // Generate description
        let description = if self.config.generate_descriptions {
            Some(self.description_generator.generate(&analysis)?)
        } else {
            None
        };
        
        // Analyze trends
        let trends = if self.config.analyze_trends && !analysis.data_points.is_empty() {
            Some(self.trend_analyzer.analyze(&analysis.data_points)?)
        } else {
            None
        };
        
        Ok(AnalyzedChart {
            id: format!("chart_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
            chart_type: analysis.chart_type,
            title: analysis.elements.title,
            axes: analysis.elements.axes,
            data_points: analysis.data_points,
            trends,
            description,
            embedding: None, // Would be generated by embedding service
        })
    }
    
    fn extract_data_points(&self, chart_image: &Path) -> RragResult<Vec<DataPoint>> {
        let analysis = self.analyze_comprehensive(chart_image)?;
        Ok(analysis.data_points)
    }
    
    fn identify_type(&self, chart_image: &Path) -> RragResult<ChartType> {
        let (chart_type, _confidence) = self.type_classifier.classify(chart_image)?;
        Ok(chart_type)
    }
    
    fn analyze_trends(&self, data_points: &[DataPoint]) -> RragResult<TrendAnalysis> {
        self.trend_analyzer.analyze(data_points)
    }
}

impl ChartTypeClassifier {
    /// Create new chart type classifier
    pub fn new() -> RragResult<Self> {
        let models = vec![
            ClassificationModel {
                model_type: ModelType::CNN,
                confidence_threshold: 0.8,
                features: vec![
                    FeatureType::ColorHistogram,
                    FeatureType::EdgeDetection,
                    FeatureType::ShapeFeatures,
                ],
            },
            ClassificationModel {
                model_type: ModelType::SVM,
                confidence_threshold: 0.7,
                features: vec![
                    FeatureType::LayoutFeatures,
                    FeatureType::TextFeatures,
                ],
            },
        ];
        
        Ok(Self { models })
    }
    
    /// Classify chart type
    pub fn classify(&self, image_path: &Path) -> RragResult<(ChartType, f32)> {
        // Simulate classification based on image analysis
        let filename = image_path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        
        // Simple heuristic based on filename for demonstration
        let (chart_type, confidence) = if filename.contains("line") {
            (ChartType::Line, 0.95)
        } else if filename.contains("bar") {
            (ChartType::Bar, 0.90)
        } else if filename.contains("pie") {
            (ChartType::Pie, 0.85)
        } else if filename.contains("scatter") {
            (ChartType::Scatter, 0.80)
        } else {
            (ChartType::Unknown, 0.5)
        };
        
        Ok((chart_type, confidence))
    }
    
    /// Extract features for classification
    pub fn extract_features(&self, image_path: &Path) -> RragResult<Vec<f32>> {
        // Simulate feature extraction
        let mut features = Vec::new();
        
        // Color histogram features
        features.extend(vec![0.1, 0.2, 0.3, 0.4]); // RGB histogram
        
        // Edge detection features
        features.extend(vec![0.5, 0.6]); // Edge density, direction
        
        // Shape features
        features.extend(vec![0.7, 0.8, 0.9]); // Rectangularity, circularity, linearity
        
        // Layout features
        features.extend(vec![0.2, 0.4]); // Symmetry, balance
        
        // Text features
        features.push(0.3); // Text density
        
        Ok(features)
    }
}

impl ChartDataExtractor {
    /// Create new data extractor
    pub fn new(ocr_enabled: bool, color_analysis: bool, shape_detection: bool) -> Self {
        Self {
            ocr_enabled,
            color_analysis,
            shape_detection,
        }
    }
    
    /// Extract data points from chart
    pub fn extract(&self, image_path: &Path, chart_type: ChartType) -> RragResult<Vec<DataPoint>> {
        match chart_type {
            ChartType::Line => self.extract_line_data(image_path),
            ChartType::Bar => self.extract_bar_data(image_path),
            ChartType::Pie => self.extract_pie_data(image_path),
            ChartType::Scatter => self.extract_scatter_data(image_path),
            ChartType::Area => self.extract_area_data(image_path),
            ChartType::Histogram => self.extract_histogram_data(image_path),
            _ => Ok(vec![]),
        }
    }
    
    /// Extract line chart data
    fn extract_line_data(&self, _image_path: &Path) -> RragResult<Vec<DataPoint>> {
        // Simulate line data extraction
        Ok(vec![
            DataPoint { x: 0.0, y: 10.0, label: None, series: Some("Line 1".to_string()) },
            DataPoint { x: 1.0, y: 15.0, label: None, series: Some("Line 1".to_string()) },
            DataPoint { x: 2.0, y: 12.0, label: None, series: Some("Line 1".to_string()) },
            DataPoint { x: 3.0, y: 18.0, label: None, series: Some("Line 1".to_string()) },
        ])
    }
    
    /// Extract bar chart data
    fn extract_bar_data(&self, _image_path: &Path) -> RragResult<Vec<DataPoint>> {
        // Simulate bar data extraction
        Ok(vec![
            DataPoint { x: 0.0, y: 25.0, label: Some("Q1".to_string()), series: None },
            DataPoint { x: 1.0, y: 30.0, label: Some("Q2".to_string()), series: None },
            DataPoint { x: 2.0, y: 35.0, label: Some("Q3".to_string()), series: None },
            DataPoint { x: 3.0, y: 40.0, label: Some("Q4".to_string()), series: None },
        ])
    }
    
    /// Extract pie chart data
    fn extract_pie_data(&self, _image_path: &Path) -> RragResult<Vec<DataPoint>> {
        // Simulate pie data extraction (percentages)
        Ok(vec![
            DataPoint { x: 0.0, y: 40.0, label: Some("Category A".to_string()), series: None },
            DataPoint { x: 1.0, y: 30.0, label: Some("Category B".to_string()), series: None },
            DataPoint { x: 2.0, y: 20.0, label: Some("Category C".to_string()), series: None },
            DataPoint { x: 3.0, y: 10.0, label: Some("Category D".to_string()), series: None },
        ])
    }
    
    /// Extract scatter plot data
    fn extract_scatter_data(&self, _image_path: &Path) -> RragResult<Vec<DataPoint>> {
        // Simulate scatter data extraction
        Ok(vec![
            DataPoint { x: 5.0, y: 10.0, label: None, series: None },
            DataPoint { x: 15.0, y: 25.0, label: None, series: None },
            DataPoint { x: 25.0, y: 20.0, label: None, series: None },
            DataPoint { x: 35.0, y: 40.0, label: None, series: None },
        ])
    }
    
    /// Extract area chart data
    fn extract_area_data(&self, image_path: &Path) -> RragResult<Vec<DataPoint>> {
        // Area charts similar to line charts
        self.extract_line_data(image_path)
    }
    
    /// Extract histogram data
    fn extract_histogram_data(&self, _image_path: &Path) -> RragResult<Vec<DataPoint>> {
        // Simulate histogram data extraction
        Ok(vec![
            DataPoint { x: 0.0, y: 5.0, label: Some("0-10".to_string()), series: None },
            DataPoint { x: 1.0, y: 15.0, label: Some("10-20".to_string()), series: None },
            DataPoint { x: 2.0, y: 25.0, label: Some("20-30".to_string()), series: None },
            DataPoint { x: 3.0, y: 10.0, label: Some("30-40".to_string()), series: None },
        ])
    }
}

impl TrendAnalyzer {
    /// Create new trend analyzer
    pub fn new(min_points: usize, smoothing_window: usize, seasonality_detection: bool) -> Self {
        Self {
            min_points,
            smoothing_window,
            seasonality_detection,
        }
    }
    
    /// Analyze trends in data points
    pub fn analyze(&self, data_points: &[DataPoint]) -> RragResult<TrendAnalysis> {
        if data_points.len() < self.min_points {
            return Err(RragError::validation("data_points", format!("minimum {} points", self.min_points), format!("{} points", data_points.len())));
        }
        
        // Calculate trend direction
        let direction = self.calculate_trend_direction(data_points);
        
        // Calculate trend strength
        let strength = self.calculate_trend_strength(data_points);
        
        // Detect seasonality if enabled
        let seasonality = if self.seasonality_detection {
            self.detect_seasonality(data_points)
        } else {
            None
        };
        
        // Detect outliers
        let outliers = self.detect_outliers(data_points);
        
        // Generate forecast if enough data
        let forecast = if data_points.len() >= 10 {
            Some(self.generate_forecast(data_points, 5)?)
        } else {
            None
        };
        
        Ok(TrendAnalysis {
            direction,
            strength,
            seasonality,
            outliers,
            forecast,
        })
    }
    
    /// Calculate trend direction
    fn calculate_trend_direction(&self, data_points: &[DataPoint]) -> TrendDirection {
        if data_points.len() < 2 {
            return TrendDirection::Stable;
        }
        
        let first_y = data_points[0].y;
        let last_y = data_points[data_points.len() - 1].y;
        let change = last_y - first_y;
        
        // Calculate volatility
        let volatility = self.calculate_volatility(data_points);
        
        if change.abs() < volatility * 0.5 {
            TrendDirection::Stable
        } else if volatility > change.abs() * 2.0 {
            TrendDirection::Volatile
        } else if change > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        }
    }
    
    /// Calculate trend strength
    fn calculate_trend_strength(&self, data_points: &[DataPoint]) -> f32 {
        if data_points.len() < 2 {
            return 0.0;
        }
        
        // Linear regression coefficient of determination (R²)
        let n = data_points.len() as f64;
        let sum_x: f64 = data_points.iter().map(|p| p.x).sum();
        let sum_y: f64 = data_points.iter().map(|p| p.y as f64).sum();
        let sum_xy: f64 = data_points.iter().map(|p| p.x * p.y as f64).sum();
        let sum_x2: f64 = data_points.iter().map(|p| p.x * p.x).sum();
        let sum_y2: f64 = data_points.iter().map(|p| (p.y as f64) * (p.y as f64)).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            return 0.0;
        }
        
        let r = numerator / denominator;
        (r * r) as f32 // R²
    }
    
    /// Calculate volatility
    fn calculate_volatility(&self, data_points: &[DataPoint]) -> f64 {
        if data_points.len() < 2 {
            return 0.0;
        }
        
        let values: Vec<f64> = data_points.iter().map(|p| p.y as f64).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
    
    /// Detect seasonality patterns
    fn detect_seasonality(&self, data_points: &[DataPoint]) -> Option<Seasonality> {
        if data_points.len() < 12 {
            return None; // Need at least 12 points for seasonality detection
        }
        
        // Simplified seasonality detection using autocorrelation
        // In practice, would use FFT or more sophisticated methods
        
        Some(Seasonality {
            period: 12.0, // Assume monthly seasonality
            amplitude: 5.0,
            phase: 0.0,
        })
    }
    
    /// Detect outliers using IQR method
    fn detect_outliers(&self, data_points: &[DataPoint]) -> Vec<DataPoint> {
        if data_points.len() < 4 {
            return vec![];
        }
        
        let mut y_values: Vec<f32> = data_points.iter().map(|p| p.y as f32).collect();
        y_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let q1_idx = y_values.len() / 4;
        let q3_idx = 3 * y_values.len() / 4;
        let q1 = y_values[q1_idx];
        let q3 = y_values[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        data_points.iter()
            .filter(|p| (p.y as f32) < lower_bound || (p.y as f32) > upper_bound)
            .cloned()
            .collect()
    }
    
    /// Generate forecast using simple linear regression
    fn generate_forecast(&self, data_points: &[DataPoint], num_points: usize) -> RragResult<Vec<DataPoint>> {
        if data_points.len() < 2 {
            return Ok(vec![]);
        }
        
        // Calculate linear regression parameters
        let n = data_points.len() as f64;
        let sum_x: f64 = data_points.iter().map(|p| p.x).sum();
        let sum_y: f64 = data_points.iter().map(|p| p.y as f64).sum();
        let sum_xy: f64 = data_points.iter().map(|p| p.x * p.y as f64).sum();
        let sum_x2: f64 = data_points.iter().map(|p| p.x * p.x).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;
        
        // Generate forecast points
        let last_x = data_points.last().unwrap().x;
        let mut forecast = Vec::new();
        
        for i in 1..=num_points {
            let x = last_x + i as f64;
            let y = (slope * x + intercept) as f32;
            
            forecast.push(DataPoint {
                x,
                y: y as f64,
                label: Some(format!("Forecast {}", i)),
                series: Some("Forecast".to_string()),
            });
        }
        
        Ok(forecast)
    }
}

impl ChartDescriptionGenerator {
    /// Create new description generator
    pub fn new() -> Self {
        let mut templates = std::collections::HashMap::new();
        
        templates.insert(
            ChartType::Line,
            "This line chart shows {data_description}. The trend is {trend_direction} with a strength of {trend_strength:.2}.".to_string()
        );
        
        templates.insert(
            ChartType::Bar,
            "This bar chart displays {data_description}. The highest value is {max_value} and the lowest is {min_value}.".to_string()
        );
        
        templates.insert(
            ChartType::Pie,
            "This pie chart represents {data_description}. The largest segment is {largest_segment} at {largest_percentage:.1}%.".to_string()
        );
        
        Self {
            templates,
            nlg_enabled: false,
        }
    }
    
    /// Generate chart description
    pub fn generate(&self, analysis: &ChartAnalysisResult) -> RragResult<String> {
        if let Some(template) = self.templates.get(&analysis.chart_type) {
            let description = self.fill_template(template, analysis)?;
            Ok(description)
        } else {
            Ok(format!("Chart of type {:?} with {} data points", 
                      analysis.chart_type, analysis.data_points.len()))
        }
    }
    
    /// Fill template with analysis data
    fn fill_template(&self, template: &str, analysis: &ChartAnalysisResult) -> RragResult<String> {
        let mut description = template.to_string();
        
        // Basic substitutions
        description = description.replace("{data_description}", 
                                        &format!("{} data points", analysis.data_points.len()));
        
        if !analysis.data_points.is_empty() {
            let max_y = analysis.data_points.iter()
                .map(|p| p.y as f32)
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let min_y = analysis.data_points.iter()
                .map(|p| p.y as f32)
                .fold(f32::INFINITY, |a, b| a.min(b));
            
            description = description.replace("{max_value}", &max_y.to_string());
            description = description.replace("{min_value}", &min_y.to_string());
        }
        
        Ok(description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_chart_processor_creation() {
        let config = ChartAnalysisConfig::default();
        let processor = DefaultChartProcessor::new(config).unwrap();
        
        assert!(processor.config.extract_data);
        assert!(processor.config.generate_descriptions);
        assert!(processor.config.analyze_trends);
    }
    
    #[test]
    fn test_chart_type_classification() {
        let classifier = ChartTypeClassifier::new().unwrap();
        
        // Create temporary file with line chart hint
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_file_name("line_chart.png");
        
        let (chart_type, confidence) = classifier.classify(&path).unwrap();
        assert_eq!(chart_type, ChartType::Line);
        assert!(confidence > 0.9);
    }
    
    #[test]
    fn test_trend_analysis() {
        let analyzer = TrendAnalyzer::new(3, 2, false);
        
        let data_points = vec![
            DataPoint { x: 0.0, y: 10.0, label: None, series: None },
            DataPoint { x: 1.0, y: 15.0, label: None, series: None },
            DataPoint { x: 2.0, y: 20.0, label: None, series: None },
            DataPoint { x: 3.0, y: 25.0, label: None, series: None },
        ];
        
        let trend = analyzer.analyze(&data_points).unwrap();
        assert_eq!(trend.direction, TrendDirection::Increasing);
        assert!(trend.strength > 0.8);
    }
    
    #[test]
    fn test_outlier_detection() {
        let analyzer = TrendAnalyzer::new(3, 2, false);
        
        let data_points = vec![
            DataPoint { x: 0.0, y: 10.0, label: None, series: None },
            DataPoint { x: 1.0, y: 12.0, label: None, series: None },
            DataPoint { x: 2.0, y: 100.0, label: None, series: None }, // Outlier
            DataPoint { x: 3.0, y: 11.0, label: None, series: None },
        ];
        
        let outliers = analyzer.detect_outliers(&data_points);
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0].y, 100.0);
    }
    
    #[test]
    fn test_data_extraction() {
        let extractor = ChartDataExtractor::new(true, true, true);
        
        let temp_file = NamedTempFile::new().unwrap();
        let data_points = extractor.extract(temp_file.path(), ChartType::Line).unwrap();
        
        assert!(!data_points.is_empty());
    }
    
    #[test]
    fn test_description_generation() {
        let generator = ChartDescriptionGenerator::new();
        
        let analysis = ChartAnalysisResult {
            chart_type: ChartType::Line,
            confidence: 0.9,
            data_points: vec![
                DataPoint { x: 0.0, y: 10.0, label: None, series: None },
                DataPoint { x: 1.0, y: 15.0, label: None, series: None },
            ],
            elements: ChartElements {
                title: None,
                axes: ChartAxes {
                    x_label: None,
                    y_label: None,
                    x_range: None,
                    y_range: None,
                },
                legend: vec![],
                series: vec![],
                annotations: vec![],
            },
            visual_properties: VisualProperties {
                chart_area: ChartArea {
                    bounds: (0.0, 0.0, 100.0, 100.0),
                    plot_area: (0.0, 0.0, 100.0, 100.0),
                    margins: (0.0, 0.0, 0.0, 0.0),
                },
                color_scheme: ColorScheme {
                    primary_colors: vec![],
                    palette_type: PaletteType::Categorical,
                    accessibility_score: 1.0,
                },
                typography: Typography {
                    title_font: None,
                    axis_font: None,
                    legend_font: None,
                    readability_score: 1.0,
                },
                grid: None,
            },
        };
        
        let description = generator.generate(&analysis).unwrap();
        assert!(description.contains("line chart"));
        assert!(description.contains("2 data points"));
    }
}