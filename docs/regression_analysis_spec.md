# Regression Analysis Specification

## Primary regression dataset

The primary analytic unit is patient-segment. The key outcome is best-model longitudinal aortic growth rate in mm/year. A floored version is used for censored and binary progression models.

## Diameter-growth linear models

For each segment separately:

`slope_floored_mm_per_year ~ baseline_diameter_per_5mm`

Adjusted model:

`slope_floored_mm_per_year ~ baseline_diameter_per_5mm + age_first_hhp_years + height_baseline_cm`

## Tobit models

Left-censored Tobit models use the floored slope outcome censored at 0 mm/year.

## Mixed-effects interaction models

Long-format model by segment:

`diameter_cm ~ time_years * baseline_diameter_per_5mm + height_baseline_cm + (time_years | patient_id)`

The interaction term `time_years:baseline_diameter_per_5mm` estimates change in annual growth rate per 5 mm baseline diameter increment.

## Logistic models

Binary progression endpoint:

`progression_ge_1mm = 1 if best_slope_floored_mm_per_year >= 1.0`

Models include clinical covariates available in HHP:
- coarctation
- BAV
- hypertension
- karyotype_45x
- prior surgery coding
- baseline diameter
- baseline age
