# Methods summary

The analytic pipeline uses long-format aortic diameter data with one row per participant, imaging date, aortic segment, and measurement source. Dates are converted to time since each participant's baseline study. Segment-specific mixed-effects models estimate expected aortic diameter as a function of time and height, with participant-level random effects. Residual-based and interval growth-rate criteria are then used to flag potentially implausible measurements.

All measurements are retained in the audit dataset and diagnostic plots. Downstream slope estimation and model-selection analyses use a filtered dataset that excludes measurements flagged as outliers. Patient-level progression is summarized by aortic segment using OLS, mixed-model, and conservative best-model slope estimates. The best-model procedure compares linear and quadratic trajectories and selects a quadratic model only when it meaningfully improves model fit by BIC and leave-one-out cross-validation criteria.

For high-risk prioritization, the pipeline identifies each participant's maximum observed diameter by segment, compares it with the model-expected diameter at the same timepoint, calculates observed-minus-expected diameter, calculates aortic height index, and ranks participants using a size-first strategy in which AHI supersedes growth except within near-tied AHI bins.
