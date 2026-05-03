#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []

def require_file(path):
    if not path.exists():
        errors.append(f"Missing file: {path.relative_to(ROOT)}")
    return path.exists()

def require_columns(path, cols):
    if not require_file(path):
        return
    df = pd.read_csv(path, nrows=5)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        errors.append(f"{path.relative_to(ROOT)} missing columns: {missing}")

qc_cols = ['patient_id','study_date','height_cm','segment','diameter_cm']
require_columns(ROOT/'data/raw/aorta_data_long_from_AllHHP2025.csv', qc_cols)
require_columns(ROOT/'data/raw/aorta_data_long_from_AllHHP2025_mapped.csv', qc_cols)
controlled = [
 'aorta_data_long_with_outliers.csv','qc_summary_by_patient_segment.csv','patient_slopes_ols_vs_mixed.csv',
 'patient_progression_categories.csv','patient_best_model_slopes_pooled_by_patient_segment.csv',
 'high_risk_combined_delta_AHI.csv','ranked_participants.csv','regression_analysis_dataset.csv',
 'mixed_model_long_dataset.csv','early_growth_dataset.csv'
]
for f in controlled:
    require_file(ROOT/'data/controlled'/f)
require_columns(ROOT/'data/controlled/regression_analysis_dataset.csv', ['patient_id','segment','baseline_diameter_cm','best_slope_mm_per_year','best_slope_floored_mm_per_year'])
require_columns(ROOT/'data/controlled/mixed_model_long_dataset.csv', ['patient_id','segment','time_years','diameter_cm','height_cm'])
require_columns(ROOT/'data/controlled/early_growth_dataset.csv', ['patient_id','segment','baseline_diameter_cm','early_rate_mm_per_year'])
if errors:
    print('VALIDATION FAILED')
    for e in errors: print('-', e)
    sys.exit(1)
print('VALIDATION PASSED: filenames and schemas are compatible with the public code archive.')
