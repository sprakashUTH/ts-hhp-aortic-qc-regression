# Separate data repository specification

The public code archive does not include real participant-level analytic data. A separate controlled-access or repository-approved data deposit should contain de-identified analysis-ready files sufficient to reproduce the manuscript results, subject to IRB, consent, and institutional governance.

## Minimum data files for full manuscript recapitulation

### QC / slope-generation inputs and outputs
- `aorta_data_long_from_AllHHP2025.csv`
- `aorta_data_long_from_AllHHP2025_mapped.csv`
- `aorta_data_long_with_outliers.csv`
- `qc_summary_by_patient_segment.csv`
- `flagged_measurements.csv`
- `patient_slopes_ols_vs_mixed.csv`
- `patient_progression_categories.csv`
- `patient_best_model_slopes_by_series.csv`
- `patient_best_model_slopes_pooled_by_patient_segment.csv`
- `high_risk_combined_delta_AHI.csv`
- `ranked_participants.csv`

### Regression datasets and outputs
- `regression_analysis_dataset.csv`
- `mixed_model_long_dataset.csv`
- `early_growth_dataset.csv`
- `linear_models_summary.csv`
- `tobit_models_summary.csv`
- `mixed_model_interactions.csv`
- `logistic_models_summary.csv`

## Relationship to this code archive
After download, users place approved files in `data/controlled/` or `data/raw/` and run the scripts/notebooks described in `MASTER_ANALYSIS_REPRODUCIBILITY_GUIDE.md`.
