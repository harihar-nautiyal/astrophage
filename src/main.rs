use astrophage::{
    data::KoiDataset, evaluation::ModelEvaluator, features::FeatureEngineer, logger::Logger,
    report::generate_report, two_stage_model::TwoStageClassifier,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Logger::init(true);
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    🪐 ASTROPHAGE v0.2.0                       ║");
    println!("║     NASA KOI Exoplanet Classification System                 ║");
    println!("║     TWO-STAGE MODEL: CONFIRMED vs NOT → CANDIDATE vs FALSE  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Load the dataset
    println!("📡 Step 1: Loading KOI dataset...");
    let dataset = KoiDataset::load("data/koi_dataset.csv")?;
    println!(
        "   ✓ Loaded {} KOIs with {} features",
        dataset.n_samples(),
        dataset.n_features()
    );

    let class_dist = dataset.class_distribution();
    println!("   ✓ Class distribution:");
    for (class, count) in &class_dist {
        println!("     • {}: {}", class, count);
    }
    println!();

    // Step 2: Feature engineering
    println!("🔧 Step 2: Engineering features...");
    let mut engineer = FeatureEngineer::new();
    let processed = engineer.process(&dataset)?;
    println!();

    // Step 3: Train/test split (stratified)
    println!("✂️  Step 3: Splitting data (80/20 stratified)...");
    let (train, test) = processed.split(0.2, 42);
    println!("   ✓ Train: {} samples", train.n_samples());
    println!("   ✓ Test: {} samples", test.n_samples());
    println!();

    // Step 4: Train TWO-STAGE model
    println!("🧠 Step 4: Training TWO-STAGE classifier...");
    let mut classifier = TwoStageClassifier::new();
    classifier.train(&train)?;
    println!();

    // Step 5: Evaluate
    println!("📊 Step 5: Evaluating model performance...");
    let evaluator = ModelEvaluator::new(&classifier, &test);
    let metrics = evaluator.evaluate()?;
    println!();

    // Step 6: Feature importance
    println!("🔬 Step 6: Top astrophysical predictors:");
    for (i, (name, score)) in classifier.feature_importance().iter().take(10).enumerate() {
        println!("   {:2}. {:<25} {:.4}", i + 1, name, score);
    }
    println!();

    // Step 7: Generate report
    println!("📝 Step 7: Generating final report...");
    generate_report(&metrics, &classifier, &engineer)?;
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("🎉 ASTROPHAGE two-stage classification complete!");
    println!("   Check output/report.json for full results.");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
