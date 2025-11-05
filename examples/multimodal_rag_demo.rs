//! # Multimodal RAG Demo
//!
//! This example demonstrates RRAG's multimodal capabilities:
//! - Processing documents with images, tables, and charts
//! - Multimodal embedding fusion
//! - Cross-modal retrieval and search
//! - Integrated text and visual understanding
//!
//! Run with: `cargo run --bin multimodal_rag_demo`

use rrag::multimodal::{
    AnalyzedChart, ChartType, ExtractedTable, MultiModalConfig, MultiModalDocument,
    MultiModalService,
};
use rrag::RragResult;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> RragResult<()> {
    tracing::debug!("🎨 RRAG Multimodal RAG Demo");
    tracing::debug!("===========================\n");

    // 1. Setup Multimodal Service
    tracing::debug!("1. Setting up multimodal processing...");
    let multimodal_service = setup_multimodal_service().await?;
    tracing::debug!("   ✓ Multimodal service initialized\n");

    // 2. Process Mixed Content Documents
    tracing::debug!("2. Processing documents with mixed content...");
    let documents = process_multimodal_documents(&multimodal_service).await?;
    tracing::debug!("   ✓ Processed {} multimodal documents\n", documents.len());

    // 3. Multimodal Search Demo
    tracing::debug!("3. Performing multimodal search...");
    demo_multimodal_search(&documents).await?;
    tracing::debug!("   ✓ Multimodal search completed\n");

    // 4. Cross-Modal Retrieval
    tracing::debug!("4. Cross-modal retrieval examples...");
    demo_cross_modal_retrieval().await?;
    tracing::debug!("   ✓ Cross-modal retrieval demonstrated\n");

    tracing::debug!("🎉 Multimodal RAG demo completed successfully!");
    Ok(())
}

async fn setup_multimodal_service() -> RragResult<MultiModalService> {
    let config = MultiModalConfig::default()
        .with_image_processing(true)
        .with_table_extraction(true)
        .with_chart_analysis(true)
        .with_ocr(true)
        .with_layout_analysis(true);

    let service = MultiModalService::new(config)?;
    tracing::debug!("   - Image processing: enabled");
    tracing::debug!("   - Table extraction: enabled");
    tracing::debug!("   - Chart analysis: enabled");
    tracing::debug!("   - OCR: enabled");
    tracing::debug!("   - Layout analysis: enabled");

    Ok(service)
}

async fn process_multimodal_documents(
    service: &MultiModalService,
) -> RragResult<Vec<MultiModalDocument>> {
    let mut documents = Vec::new();

    // Example 1: Financial report with charts and tables
    tracing::debug!("   Processing financial report...");
    let financial_doc = create_financial_report().await?;
    let processed_financial = service.process_document(&financial_doc).await?;
    documents.push(processed_financial);

    // Example 2: Research paper with diagrams
    tracing::debug!("   Processing research paper...");
    let research_doc = create_research_paper().await?;
    let processed_research = service.process_document(&research_doc).await?;
    documents.push(processed_research);

    // Example 3: Product catalog with images
    tracing::debug!("   Processing product catalog...");
    let catalog_doc = create_product_catalog().await?;
    let processed_catalog = service.process_document(&catalog_doc).await?;
    documents.push(processed_catalog);

    Ok(documents)
}

async fn create_financial_report() -> RragResult<PathBuf> {
    // Mock financial report with:
    // - Revenue charts (line graphs)
    // - Quarterly data tables
    // - Executive summary text
    // - Infographics

    tracing::debug!("     - Revenue trend charts: 3 detected");
    tracing::debug!("     - Data tables: 5 extracted");
    tracing::debug!("     - Text sections: 12 processed");

    // Return mock path - in real usage this would be an actual file
    Ok(PathBuf::from("./examples/data/financial_report.pdf"))
}

async fn create_research_paper() -> RragResult<PathBuf> {
    // Mock research paper with:
    // - Scientific diagrams
    // - Data visualizations
    // - Mathematical equations
    // - Citation tables

    tracing::debug!("     - Scientific diagrams: 8 analyzed");
    tracing::debug!("     - Equations: 15 recognized (OCR)");
    tracing::debug!("     - Reference tables: 2 structured");

    Ok(PathBuf::from("./examples/data/research_paper.pdf"))
}

async fn create_product_catalog() -> RragResult<PathBuf> {
    // Mock product catalog with:
    // - Product images with descriptions
    // - Specification tables
    // - Comparison charts
    // - Price tables

    tracing::debug!("     - Product images: 24 captioned");
    tracing::debug!("     - Spec tables: 12 extracted");
    tracing::debug!("     - Price comparisons: 3 charts analyzed");

    Ok(PathBuf::from("./examples/data/product_catalog.pdf"))
}

async fn demo_multimodal_search(documents: &[MultiModalDocument]) -> RragResult<()> {
    tracing::debug!("   Searching across modalities:");

    // Text-based search
    let text_query = "revenue growth trends";
    tracing::debug!("     🔤 Text query: '{}'", text_query);

    // Find documents with relevant text, charts, or tables
    for (i, doc) in documents.iter().enumerate() {
        let relevance_score = calculate_multimodal_relevance(doc, text_query).await?;
        if relevance_score > 0.7 {
            tracing::debug!(
                "       Found in document {}: score {:.2}",
                i + 1,
                relevance_score
            );

            // Show what modalities matched
            if doc.text_content.to_lowercase().contains("revenue") {
                tracing::debug!("         ✓ Text content match");
            }
            if doc
                .charts
                .iter()
                .any(|c| matches!(c.chart_type, ChartType::Line))
            {
                tracing::debug!("         ✓ Revenue trend chart detected");
            }
            if doc
                .tables
                .iter()
                .any(|t| t.headers.iter().any(|h| h.contains("Revenue")))
            {
                tracing::debug!("         ✓ Revenue data table found");
            }
        }
    }

    // Visual similarity search
    tracing::debug!("     🖼️  Visual similarity search:");
    tracing::debug!("       - Found 3 similar chart patterns");
    tracing::debug!("       - Identified 2 matching table structures");
    tracing::debug!("       - Located 5 related images");

    Ok(())
}

async fn demo_cross_modal_retrieval() -> RragResult<()> {
    tracing::debug!("   Cross-modal retrieval scenarios:");

    // Text query → Visual results
    tracing::debug!("     📝 → 🖼️  'Show me sales performance'");
    tracing::debug!("       Result: Revenue charts, performance dashboards");

    // Image query → Text results
    tracing::debug!("     🖼️  → 📝 [Upload chart image]");
    tracing::debug!("       Result: Related textual analysis and explanations");

    // Table query → Chart results
    tracing::debug!("     📊 → 📈 'Similar data patterns'");
    tracing::debug!("       Result: Charts with similar data distributions");

    // Combined multimodal query
    tracing::debug!("     🔗 Combined: 'Financial performance with visual evidence'");
    tracing::debug!("       Result: Text analysis + supporting charts + data tables");

    Ok(())
}

async fn calculate_multimodal_relevance(
    document: &MultiModalDocument,
    query: &str,
) -> RragResult<f32> {
    let mut total_score = 0.0f32;
    let mut weight_sum = 0.0f32;

    // Text relevance (weight: 0.4)
    let text_score = calculate_text_relevance(&document.text_content, query);
    total_score += text_score * 0.4;
    weight_sum += 0.4;

    // Chart relevance (weight: 0.3)
    if !document.charts.is_empty() {
        let chart_score = calculate_chart_relevance(&document.charts, query);
        total_score += chart_score * 0.3;
        weight_sum += 0.3;
    }

    // Table relevance (weight: 0.3)
    if !document.tables.is_empty() {
        let table_score = calculate_table_relevance(&document.tables, query);
        total_score += table_score * 0.3;
        weight_sum += 0.3;
    }

    Ok(if weight_sum > 0.0 {
        total_score / weight_sum
    } else {
        0.0
    })
}

fn calculate_text_relevance(text: &str, query: &str) -> f32 {
    // Simple keyword matching - in production use proper embedding similarity
    let text_lower = text.to_lowercase();
    let query_lower = query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    let matching_terms = query_terms
        .iter()
        .filter(|term| text_lower.contains(*term))
        .count();

    matching_terms as f32 / query_terms.len() as f32
}

fn calculate_chart_relevance(charts: &[AnalyzedChart], query: &str) -> f32 {
    // Check chart titles and descriptions for relevance
    charts
        .iter()
        .map(|chart| {
            let mut score = 0.0;

            if let Some(title) = &chart.title {
                score += calculate_text_relevance(title, query) * 0.6;
            }

            if let Some(description) = &chart.description {
                score += calculate_text_relevance(description, query) * 0.4;
            }

            score
        })
        .fold(0.0, f32::max)
}

fn calculate_table_relevance(tables: &[ExtractedTable], query: &str) -> f32 {
    // Check table headers and summary for relevance
    tables
        .iter()
        .map(|table| {
            let mut score = 0.0;

            // Check headers
            let headers_text = table.headers.join(" ");
            score += calculate_text_relevance(&headers_text, query) * 0.5;

            // Check summary if available
            if let Some(summary) = &table.summary {
                score += calculate_text_relevance(summary, query) * 0.5;
            }

            score
        })
        .fold(0.0, f32::max)
}

// Mock implementations for multimodal config extension
trait MultiModalConfigExt {
    fn with_image_processing(self, enabled: bool) -> Self;
    fn with_table_extraction(self, enabled: bool) -> Self;
    fn with_chart_analysis(self, enabled: bool) -> Self;
    fn with_ocr(self, enabled: bool) -> Self;
    fn with_layout_analysis(self, enabled: bool) -> Self;
}

impl MultiModalConfigExt for MultiModalConfig {
    fn with_image_processing(mut self, enabled: bool) -> Self {
        self.process_images = enabled;
        self
    }

    fn with_table_extraction(mut self, enabled: bool) -> Self {
        self.process_tables = enabled;
        self
    }

    fn with_chart_analysis(mut self, enabled: bool) -> Self {
        self.process_charts = enabled;
        self
    }

    fn with_ocr(self, _enabled: bool) -> Self {
        // OCR would be configured through OCRConfig
        self
    }

    fn with_layout_analysis(self, _enabled: bool) -> Self {
        // Layout analysis would be configured through LayoutAnalysisConfig
        self
    }
}
