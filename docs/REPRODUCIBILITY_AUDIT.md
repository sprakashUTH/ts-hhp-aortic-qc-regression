# Reproducibility audit against manuscript and supplement

This audit maps the manuscript/supplement analytic claims to archive artifacts.

## Fully supported by public archive code plus controlled data repository

| Manuscript / supplement component | Required artifacts | Archive status |
|---|---|---|
| Long-format HHP QC and outlier workflow | `analysis/01_hhp_full_cohort_qc.py`, `docs/qc_flow_diagram.md`, `data/controlled/aorta_data_long_from_AllHHP2025.csv` | Code included; data external |
| Enriched/best-model QC with source weighting and ranking | `analysis/02_enriched_bestmodel_qc.py`, `run_enriched_bestmodel.sh`, `data/controlled/aorta_data_long_from_AllHHP2025_mapped.csv` | Code included; data external |
| QC summary, flagged measurements, progression categories | QC scripts and generated CSV outputs | Code included; outputs external |
| Supplemental QC workflow figure/caption | `docs/qc_flow_diagram.md`, `manuscript/supplemental_caption.md` | Included |
| QC supplementary methods notebook | `notebooks/HHP_Aortic_QC_Supplementary_Methods_Reproducibility.ipynb` | Included verbatim |
| Regression analyses: baseline diameter vs slope, age vs slope, Tobit/floored-slope models, mixed-effects interaction models, progression logistic models | `analysis/03_regression_analyses.py`, `docs/regression_analysis_spec.md`, `notebooks/HHP_regression_figure_reproduction.ipynb`, controlled regression datasets | Code included; data external |
| Figures S1-S4: baseline diameter vs growth | `regression_analysis_dataset.csv`, regression notebook/master notebook | Notebook included; data external |
| Figures S7-S12: age vs growth | `regression_analysis_dataset.csv` with age variables | Notebook included; data external |
| Figures S13-S15: modeled baseline-diameter effects and mixed-model interactions | `mixed_model_long_dataset.csv`, regression script outputs | Code included; data external |
| Figure S16: early growth vs baseline maximum diameter | `early_growth_dataset.csv` | Code included; data external |
| Figures S18-S19: observed-minus-expected diameter vs AHI | `high_risk_combined_delta_AHI.csv` | Notebook support included; data external |

## Partially supported / requires endpoint dataset
Clinical endpoint analyses involving all-cause death, aortic surgery, dissection, or composite endpoint require a de-identified endpoint dataset not included in the public code archive. If endpoint analyses are retained in the manuscript, the separate data repository should include a de-identified endpoint table or the manuscript data availability statement should explicitly state that endpoint variables are available only through controlled access.

## HIPAA/public-release audit
The regenerated public archive excludes:
- individual-level analytic CSV outputs
- individual-level plots
- direct identifiers, linking identifiers, MRNs, accession numbers, and unapproved dates

## Reproducibility conclusion
The archive is suitable as a public code/notebook/documentation repository. Full numerical reproduction requires a separate approved data repository or controlled-access data package containing the de-identified analysis-ready datasets listed in `docs/DATA_REPOSITORY_SPECIFICATION.md`.
