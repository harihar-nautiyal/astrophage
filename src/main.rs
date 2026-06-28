use astrophage::{
    data::KoiDataset, evaluation::ModelEvaluator, features::FeatureEngineer, logger::Logger,
    report::generate_report, two_stage_model::TwoStageClassifier,
};
use tracing::info;

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    Logger::init(true).await;
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    🪐 ASTROPHAGE v0.2.0                       ║");
    println!("║     NASA KOI Exoplanet Classification System                 ║");
    println!("║     TWO-STAGE MODEL: CONFIRMED vs NOT → CANDIDATE vs FALSE  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Loading dataset
    info!("Step 1: Loading KOI dataset...");
    let dataset = KoiDataset::load("data/koi_dataset.csv")?;
    info!(
        "Loaded {} KOIs with {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    let class_dist = dataset.class_distribution();
    info!("Class distribution:");
    let mut l = String::new();
    for (class, count) in &class_dist {
        l.push_str(&format!("{}: {} ", class, count));
    }

    info!("{}", l);

    // Step 2: Feature engineering
    info!("Step 2: Engineering features...");
    let mut engineer = FeatureEngineer::new();
    let processed = engineer.process(&dataset)?;

    // Step 3: Train/test split (stratified)
    info!("Step 3: Splitting data (80/20 stratified)...");
    let (train, test) = processed.split(0.2, 42);
    info!("Train: {} samples", train.n_samples());
    info!("Test: {} samples", test.n_samples());

    // Step 4: Train TWO-STAGE model
    info!("Step 4: Training TWO-STAGE classifier...");
    let mut classifier = TwoStageClassifier::new();
    classifier.train(&train)?;

    // Step 5: Evaluate
    info!("Step 5: Evaluating model performance...");
    let evaluator = ModelEvaluator::new(&classifier, &test);
    let metrics = evaluator.evaluate()?;

    // Step 6: Feature importance
    info!("Step 6: Top astrophysical predictors:");
    for (i, (name, score)) in classifier.feature_importance().iter().take(10).enumerate() {
        info!("   {:2}. {:<25} {:.4}", i + 1, name, score);
    }

    // Step 7: Generate report
    info!("Step 7: Generating final report...");
    generate_report(&metrics, &classifier)?;

    info!("ASTROPHAGE two-stage classification complete!");
    info!("Check output/report.json for full results.");

    Ok(())
}
