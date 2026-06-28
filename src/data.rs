use anyhow::Result;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use tracing::info;

pub struct KoiDataset {
    pub features: Array2<f64>,
    pub labels: Array1<u8>,
    pub feature_names: Vec<String>,
    pub label_map: HashMap<String, u8>,
    pub label_names: HashMap<u8, String>,
}

impl KoiDataset {
    pub fn load(path: &str) -> Result<Self> {
        use polars::prelude::*;

        info!("Reading CSV from: {}", path);

        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("data/koi_dataset.csv".into()))
            .unwrap()
            .finish()
            .unwrap();

        info!(
            "Loaded DataFrame: {} rows × {} columns",
            df.height(),
            df.width()
        );

        let target_col = df.column("koi_disposition")?;

        let mut label_map = HashMap::new();
        label_map.insert("CONFIRMED".to_string(), 0u8);
        label_map.insert("CANDIDATE".to_string(), 1u8);
        label_map.insert("FALSE POSITIVE".to_string(), 2u8);

        let mut label_names = HashMap::new();
        label_names.insert(0u8, "CONFIRMED".to_string());
        label_names.insert(1u8, "CANDIDATE".to_string());
        label_names.insert(2u8, "FALSE POSITIVE".to_string());

        let labels_vec: Vec<u8> = target_col
            .str()?
            .iter()
            .map(|opt| opt.and_then(|s| label_map.get(s).copied()).unwrap_or(255u8))
            .collect();

        let labels = Array1::from(labels_vec);

        let exclude_cols = [
            "koi_disposition",
            "koi_vet_stat",
            "koi_vet_date",
            "koi_pdisposition",
            "koi_disp_prov",
            "koi_comment",
            "koi_fittype",
            "koi_parm_prov",
            "koi_tce_delivname",
            "koi_datalink_dvr",
            "koi_datalink_dvs",
            "koi_sparprov",
            "koi_trans_mod",
            "kepoi_name",
            "kepler_name",
            "rowid",
            "kepid",
            "koi_tce_plnt_num",
            "koi_quarters",
        ];

        let feature_names: Vec<String> = df
            .get_column_names()
            .into_iter()
            .filter(|name| !exclude_cols.contains(&name.as_str()))
            .map(|s| s.to_string())
            .collect();

        info!("Selected {} feature columns", feature_names.len());

        let mut casted_columns = Vec::new();

        for name in &feature_names {
            let col = df.column(name)?;
            let casted = col.cast(&DataType::Float64)?;

            casted_columns.push(casted);
        }

        let feature_df = DataFrame::new_infer_height(casted_columns)?;

        let features: Array2<f64> = feature_df.to_ndarray::<Float64Type>(IndexOrder::C)?;

        println!(
            "Feature matrix: {} × {}",
            features.nrows(),
            features.ncols()
        );

        Ok(Self {
            features,
            labels,
            feature_names,
            label_map,
            label_names,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    pub fn features(&self) -> &Array2<f64> {
        &self.features
    }

    pub fn labels(&self) -> &Array1<u8> {
        &self.labels
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn class_distribution(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for &label in self.labels.iter() {
            let name = self
                .label_names
                .get(&label)
                .unwrap_or(&"UNKNOWN".to_string())
                .clone();
            *counts.entry(name).or_insert(0) += 1;
        }
        counts
    }

    pub fn label_name(&self, code: u8) -> Option<&String> {
        self.label_names.get(&code)
    }

    pub fn features_mut(&mut self) -> &mut Array2<f64> {
        &mut self.features
    }

    pub fn feature_index(&self, name: &str) -> Option<usize> {
        self.feature_names.iter().position(|n| n == name)
    }
}

pub struct KoiDisposition;

impl KoiDisposition {
    pub const CONFIRMED: u8 = 0;
    pub const CANDIDATE: u8 = 1;
    pub const FALSE_POSITIVE: u8 = 2;

    pub fn to_string(code: u8) -> &'static str {
        match code {
            Self::CONFIRMED => "CONFIRMED",
            Self::CANDIDATE => "CANDIDATE",
            Self::FALSE_POSITIVE => "FALSE_POSITIVE",
            _ => "UNKNOWN",
        }
    }
}
