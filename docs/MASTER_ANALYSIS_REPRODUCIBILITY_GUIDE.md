# Master Reproducibility Guide
## HHP Turner Syndrome Aortic QC, Slope Estimation, Regression, and Figure Recapitulation

This document describes how to recapitulate the computational analyses for the manuscript *Longitudinal Echocardiographic Surveillance of Aortic Dilation in a Phenotype-Enriched Turner Syndrome Cohort*. It reconciles the QC repository, the supplementary QC notebook, and the regression archive into one traceable workflow.

## 1. Artifact map

### A. QC and slope-generation repository
Authoritative pipeline scripts:

- `analysis/01_hhp_full_cohort_qc.py`  
  Full HHP cohort quality-control and longitudinal modeling pipeline.

- `analysis/02_enriched_bestmodel_qc.py`  
  Enriched/top40 cohort best-model and source-weighted QC pipeline.

Execution helpers:

- `run_full_cohort.sh`
- `run_enriched_bestmodel.sh`

Repository documentation:

- `README.md`
- `docs/qc_flow_diagram.md`
- `docs/methods_summary.md`
- `manuscript/data_code_availability_statement.md`
- `manuscript/supplemental_caption.md`
- `environment/requirements.txt`
- `environment/environment.yml`
- `data/raw/input_data_dictionary.csv`

### B. Supplementary QC notebook

- `HHP_Aortic_QC_Supplementary_Methods_Reproducibility.ipynb`

This notebook is a reviewer-facing figure and summary notebook. It should not replace the pipeline scripts. Its role is to load already generated analytic outputs and recreate selected QC summaries, plots, and supplemental visualizations.

### C. Regression archive

The regression archive is downstream of the QC outputs. It documents the manuscript’s statistical inference layer: baseline diameter versus growth, Tobit/floored-slope analyses, mixed-effects interaction models, progression models, and figure-generation analyses.

Core regression files include:

- `regression_analysis_dataset.csv`
- `early_growth_dataset.csv`
- `mixed_model_long_dataset.csv`
- `code/regression_analyses.py`
- `docs/regression_analysis_spec.md`
- `outputs/regression_results/*.csv`
- a regression figure-reproduction notebook

## 2. Repository strategy

1. **QC scripts are authoritative.**  
   They generate outlier flags, fitted values, slope estimates, progression categories, and high-risk ranking inputs.

2. **Regression scripts are authoritative for inference.**  
   They generate model coefficients, confidence intervals, sensitivity analyses, and regression-specific outputs.

3. **Notebook(s) are reviewer-facing recapitulation tools.**  
   They should reproduce figures and show how outputs connect to manuscript claims without becoming the primary implementation.

## 3. Minimal execution order

### Step 0. Create computational environment

Using conda:

```bash
conda env create -f environment/environment.yml
conda activate hhp-aortic-qc
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

### Step 1. Run full HHP QC pipeline

Required input:

```text
data/raw/aorta_data_long_from_AllHHP2025.csv
```

Command:

```bash
cp data/raw/aorta_data_long_from_AllHHP2025.csv .
python analysis/01_hhp_full_cohort_qc.py
```

Primary outputs:

```text
aorta_data_long_with_outliers.csv
qc_summary_by_patient_segment.csv
flagged_measurements.csv
patient_slopes_by_segment.csv
patient_slopes_ols_vs_mixed.csv
patient_progression_categories.csv
patient_best_model_slopes_by_series.csv
patient_best_model_slopes_pooled_by_patient_segment.csv
high_risk_combined_delta_AHI.csv
ranked_participants.csv
plots/
```

### Step 2. Run enriched/top40 best-model QC pipeline

Required input:

```text
data/raw/aorta_data_long_from_AllHHP2025_mapped.csv
```

Command:

```bash
cp data/raw/aorta_data_long_from_AllHHP2025_mapped.csv .
python analysis/02_enriched_bestmodel_qc.py
```

This produces analogous enriched-cohort outputs and source-weighted best-model results.

### Step 3. Build regression-ready datasets

Use the deidentified QC outputs and approved HHP covariate annotations to generate:

```text
regression_analysis_dataset.csv
early_growth_dataset.csv
mixed_model_long_dataset.csv
```
### Step 4. Run regression analyses

Command from the regression archive root:

```bash
python code/regression_analyses.py
```

Expected outputs:

```text
outputs/regression_results/linear_models_summary.csv
outputs/regression_results/tobit_models_summary.csv
outputs/regression_results/mixed_model_interactions.csv
outputs/regression_results/logistic_models_summary.csv
outputs/regression_results/model_qc_summary.csv
```
These outputs correspond to the manuscript’s inferential analyses, including:

- baseline diameter versus longitudinal growth
- floored/censored slope analyses
- mixed-effects time × baseline diameter interaction models
- age and segment sensitivity analyses
- progression endpoint models
- AHI/delta visualization inputs

### Step 5. Run reviewer-facing notebooks

- QC notebook: `HHP_Aortic_QC_Supplementary_Methods_Reproducibility.ipynb`
- Regression notebook: `HHP_Aortic_Regression_Figure_Reproduction.ipynb`

## 4. Manuscript-to-file crosswalk

| Manuscript claim / output | Primary source file(s) | Script / notebook |
|---|---|---|
| QC flags and flagged-measurement counts | `aorta_data_long_with_outliers.csv`, `flagged_measurements.csv`, `qc_summary_by_patient_segment.csv` | `01_hhp_full_cohort_qc.py`, `02_enriched_bestmodel_qc.py`, QC notebook |
| Patient-level slopes | `patient_best_model_slopes_pooled_by_patient_segment.csv`, `patient_slopes_ols_vs_mixed.csv` | QC scripts |
| Progression categories | `patient_progression_categories.csv` | QC scripts |
| AHI and observed-minus-expected diameter | `high_risk_combined_delta_AHI.csv` | QC scripts |
| Full HHP diameter-growth regressions | `regression_analysis_dataset.csv` | `regression_analyses.py`, regression notebook |
| Early growth / regression-to-mean analysis | `early_growth_dataset.csv` | `regression_analyses.py`, regression notebook |
| Mixed-effects time × baseline-diameter interaction | `mixed_model_long_dataset.csv` | `regression_analyses.py` |
| Supplemental figures | frozen CSV outputs + regression outputs | QC notebook and/or master notebook |

