# HHP Aortic QC and Regression Public Reproducibility Archive

This public archive contains the code, notebooks, documentation, environment files, and synthetic smoke-test data for the manuscript analyses of longitudinal aortic dilation in the Turner Syndrome Healthy Heart Project cohort.

## Quick Start 
```bash
conda env create -f environment/environment.yml
conda activate hhp-aortic-qc
python analysis/01_hhp_full_cohort_qc.py
jupyter notebook notebooks/MASTER_HHP_AORTIC_REPRODUCIBILITY_NOTEBOOK.ipynb

## What is included

- `analysis/01_hhp_full_cohort_qc.py` — full HHP cohort QC and slope-generation pipeline
- `analysis/02_enriched_bestmodel_qc.py` — enriched/top40 best-model QC, source weighting, and high-risk ranking pipeline
- `analysis/03_regression_analyses.py` — regression/inference layer
- `notebooks/HHP_Aortic_QC_Supplementary_Methods_Reproducibility.ipynb` — original QC supplementary notebook, included verbatim
- `notebooks/HHP_regression_figure_reproduction.ipynb` — regression figure notebook
- `notebooks/MASTER_HHP_AORTIC_REPRODUCIBILITY_NOTEBOOK.ipynb` — reviewer-facing orchestration notebook
- `docs/REPRODUCIBILITY_AUDIT.md` — manuscript/supplement-to-artifact audit
- `docs/DATA_REPOSITORY_SPECIFICATION.md` — separate data repository requirements
- `docs/NOTEBOOK_ROLES.md` — explanation of why notebooks are not merged
- synthetic smoke-test data in `data/synthetic/`

## What is not included

Real participant-level analytic data are excluded from the public archive.
This dataset is archived at:
https://doi.org/10.5281/zenodo.20004811

It is designed to be used with:
- hhp-aortic-qc-pipeline (code)
- hhp-aortic-regression-analyses (code)

## How to run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

Smoke-test the master notebook:

```bash
jupyter notebook notebooks/MASTER_HHP_AORTIC_REPRODUCIBILITY_NOTEBOOK.ipynb
```

Full manuscript recapitulation requires approved de-identified data files from the separate data repository placed in `data/controlled/` or `data/raw/`. See `docs/DATA_REPOSITORY_SPECIFICATION.md`.

## Authoritative computation

The `.py` scripts in `analysis/` are the authoritative computational implementation. Notebooks are for QC visualization, regression figure reproduction, and reviewer-facing recapitulation.

## Research-use disclaimer

This code is for research reproducibility only and is not intended for clinical decision-making.

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Disclaimer

This software is intended for research purposes only and is not intended for clinical decision-making.

## Data Availability

De-identified datasets are available at https:///doi.org/10.5281/zenodo.20004811
