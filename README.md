# HHP Aortic Zenodo Data Archive — Code-Aligned Deidentified Release

This archive is aligned to the companion public code archive for the HHP aortic QC and regression analyses. It contains deidentified input files and frozen analysis outputs with filenames that match the public code archive and master notebooks.

## Privacy and deidentification

Original identifiers were replaced with randomized numeric `patient_id` values. Exact dates of service were replaced with synthetic per-participant relative dates anchored at 2000-01-01, preserving within-person intervals needed to rerun longitudinal code while removing actual dates.

## Drop-in files for QC scripts

The public code archive expects the following files. They are included here with exact names:

```text
data/raw/aorta_data_long_from_AllHHP2025.csv
data/raw/aorta_data_long_from_AllHHP2025_mapped.csv
```

The enriched Top40 cohort is included within
aorta_data_long_from_AllHHP2025_mapped.csv.

Participants are identified using the column:
- cohort_group (HHP vs Top40)

This dataset is archived at:
https://doi.org/10.5281/zenodo.XXXXXXX

It is designed to be used with:
- hhp-aortic-qc-pipeline (code)
- hhp-aortic-regression-analyses (code)

To rerun the QC scripts from the public code archive:

```bash
# from the root of the public code archive
cp /path/to/zenodo_archive/data/raw/aorta_data_long_from_AllHHP2025.csv .
python analysis/01_hhp_full_cohort_qc.py

cp /path/to/zenodo_archive/data/raw/aorta_data_long_from_AllHHP2025_mapped.csv .
python analysis/02_enriched_bestmodel_qc.py
```

## Drop-in files for the master notebook

The master notebook in the public code archive reads controlled data from `data/controlled/` after setting `USE_SYNTHETIC_DEMO = False`. This archive includes the expected files:

```text
data/controlled/aorta_data_long_with_outliers.csv
data/controlled/qc_summary_by_patient_segment.csv
data/controlled/patient_slopes_ols_vs_mixed.csv
data/controlled/patient_progression_categories.csv
data/controlled/patient_best_model_slopes_pooled_by_patient_segment.csv
data/controlled/high_risk_combined_delta_AHI.csv
data/controlled/ranked_participants.csv
data/controlled/regression_analysis_dataset.csv
data/controlled/mixed_model_long_dataset.csv
data/controlled/early_growth_dataset.csv
```

To use them:

```bash
cp -R /path/to/zenodo_archive/data/controlled/* /path/to/public_code_archive/data/controlled/
# Open notebooks/MASTER_HHP_AORTIC_REPRODUCIBILITY_NOTEBOOK.ipynb
# Set USE_SYNTHETIC_DEMO = False
```

## Validation

Run the included validation script:

```bash
python validation/validate_zenodo_data_against_public_code.py
```

It confirms required filenames and schemas for the public code archive.

## Additional processed files

`data/processed/` includes deidentified visit-level covariates and endpoint/covariate files used to document regression model construction and manuscript-specific analyses.

## Use restrictions

These data are provided for research reproducibility. Users must not attempt to reidentify participants or link these data to external identifiable records. These data and code are not intended for clinical decision-making.
