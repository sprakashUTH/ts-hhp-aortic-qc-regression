
# --- Source weighting (for model fitting) ---
# User-specified relative weights: HHP = 2× outside echo, and HHP = 4× CTMR.
# We implement this as:
#   outside: 1.0
#   HHP:     2.0
#   CTMR:    0.5
# For MixedLM (no native weights), we approximate frequency weights by expanding rows:
#   outside: 2, HHP: 4, CTMR: 1  (all integers; same relative ratios)
SOURCE_WEIGHT_FLOAT = {"outside": 1.0, "HHP": 2.0, "CTMR": 0.5}
SOURCE_WEIGHT_FREQ_INT = {"outside": 2, "HHP": 4, "CTMR": 1}

def normalize_source_label(x: str) -> str:
    if x is None:
        return "outside"
    s = str(x).strip()
    if s.lower() in ["hhp", "conference", "turner", "tssus"]:
        return "HHP"
    if s.lower() in ["ctmr", "ct", "cta", "mri", "cmr"]:
        return "CTMR"
    return "outside"

def add_source_weights(df, source_col="Source"):
    """Add source_weight (float) and source_weight_freq (int) columns."""
    if source_col not in df.columns:
        df["source_weight"] = 1.0
        df["source_weight_freq"] = 2
        return df
    src = df[source_col].map(normalize_source_label)
    df[source_col] = src
    df["source_weight"] = src.map(SOURCE_WEIGHT_FLOAT).fillna(1.0).astype(float)
    df["source_weight_freq"] = src.map(SOURCE_WEIGHT_FREQ_INT).fillna(2).astype(int)
    return df

def expand_by_frequency_weight(df, w_col="source_weight_freq"):
    """Approximate frequency weights by repeating rows (for MixedLM)."""
    if w_col not in df.columns:
        return df
    w = df[w_col].fillna(1).astype(int).clip(lower=1)
    return df.loc[df.index.repeat(w)].reset_index(drop=True)
#!/usr/bin/env python3
# Aortic diameter QC pipeline – LONG FORMAT (with Source-labeled plots + ranking)
#
# Input: long-format CSV with columns:
#   - patient_id
#   - study_date
#   - height_cm
#   - segment        ('sinus' or 'ascending')
#   - diameter_cm
#   - Source         (optional; used for plotting labels & summaries)
#
# Features:
# - Height as fixed-effect covariate
# - Segment-specific mixed-effects structures
# - Residual- and rate-based outlier detection
# - QC summaries & patient-level slopes with confidence labels
# - Patient×segment plots with Source categories labeled (CTMR / Outside / HHP)
# - Integrated "ranker.py" functionality to generate ranked participants list

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# MODEL SELECTION: linear vs quadratic per series (patient×segment×Source)
# Conservative rule:
#   - Fit quadratic only if n>=4
#   - Choose quadratic ONLY if ΔBIC < -2 AND LOOCV RMSE improves (n>=5)
# Classification remains based on OLS slopes elsewhere (clinical interpretability).
# ===============================

import statsmodels.api as sm  # for OLS design matrices used below

# Optional dependency: used only for constrained quadratic refits.
try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    minimize = None
    _HAVE_SCIPY = False

def _fit_linear_ols(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    X = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, X).fit()

def _fit_quadratic_ols(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([x, x**2])
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X).fit()

def _quad_design_matrix(x):
    """Design matrix for y = b0 + b1*x + b2*x^2."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    X = np.column_stack([x, x**2])
    X = sm.add_constant(X, has_constant="add")
    return X

def _predict_quadratic_params(params, x):
    """Predict quadratic using explicit params [b0, b1, b2]."""
    x = np.asarray(x, dtype=float)
    b0, b1, b2 = float(params[0]), float(params[1]), float(params[2])
    return b0 + b1 * x + b2 * (x ** 2)

def _quadratic_slope(params, t):
    """Derivative of quadratic at time t."""
    b1, b2 = float(params[1]), float(params[2])
    return b1 + 2.0 * b2 * float(t)

def fit_quadratic_constrained_nonnegative_slope(x, y):
    """
    Fit quadratic with constraint that slope is never negative over the observed interval.

    Since slope(t) = b1 + 2*b2*t is linear in t, enforcing non-negativity on
    [t_min, t_max] is equivalent to enforcing it at BOTH endpoints.

    Returns:
        params (np.ndarray shape (3,)): [b0, b1, b2]
        constrained (bool): whether an optimization refit was performed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        q = _fit_quadratic_ols(x, y)
        return np.asarray(q.params, dtype=float), False

    q = _fit_quadratic_ols(x, y)
    p0 = np.asarray(q.params, dtype=float)
    tmin, tmax = float(np.min(x)), float(np.max(x))

    # If already monotone-nondecreasing over observed span, keep OLS params.
    if (_quadratic_slope(p0, tmin) >= 0.0) and (_quadratic_slope(p0, tmax) >= 0.0):
        return p0, False

    if not _HAVE_SCIPY:
        return p0, False

    X = _quad_design_matrix(x)

    def obj(p):
        r = y - X.dot(p)
        return float(np.sum(r * r))

    cons = (
        {"type": "ineq", "fun": lambda p, t=tmin: _quadratic_slope(p, t)},
        {"type": "ineq", "fun": lambda p, t=tmax: _quadratic_slope(p, t)},
    )

    res = minimize(obj, p0, method="SLSQP", constraints=cons)
    if (not res.success) or (res.x is None) or (len(res.x) != 3):
        return p0, False

    p = np.asarray(res.x, dtype=float)
    # Numerical guard
    if (_quadratic_slope(p, tmin) < -1e-10) or (_quadratic_slope(p, tmax) < -1e-10):
        return p0, False
    return p, True

def _predict_linear(m, x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    X = sm.add_constant(x, has_constant="add")
    return np.asarray(m.predict(X), dtype=float)

def _predict_quadratic(m, x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    X = np.column_stack([x, x**2])
    X = sm.add_constant(X, has_constant="add")
    return np.asarray(m.predict(X), dtype=float)

def _loocv_rmse(model_func, pred_func, x, y, min_n):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < min_n:
        return float("nan")
    se = []
    for i in range(n):
        x_tr = np.delete(x, i)
        y_tr = np.delete(y, i)
        x_te = x[i:i+1]
        y_te = y[i:i+1]
        m = model_func(x_tr, y_tr)
        yhat = float(pred_func(m, x_te)[0])
        se.append((y_te[0] - yhat) ** 2)
    return float(np.sqrt(np.mean(se))) if se else float("nan")


def _fit_linear_wls(x, y, w=None):
    X = sm.add_constant(np.asarray(x, dtype=float))
    if w is None:
        return sm.OLS(np.asarray(y, dtype=float), X).fit()
    return sm.WLS(np.asarray(y, dtype=float), X, weights=np.asarray(w, dtype=float)).fit()

def _fit_quadratic_wls(x, y, w=None):
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x, x**2])
    if w is None:
        return sm.OLS(np.asarray(y, dtype=float), X).fit()
    return sm.WLS(np.asarray(y, dtype=float), X, weights=np.asarray(w, dtype=float)).fit()

def _weighted_rmse(resid, w=None):
    resid = np.asarray(resid, dtype=float)
    if w is None:
        return float(np.sqrt(np.nanmean(resid**2)))
    w = np.asarray(w, dtype=float)
    m = np.isfinite(resid) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float(np.sqrt(np.nanmean(resid**2)))
    return float(np.sqrt(np.average(resid[m]**2, weights=w[m])))
def choose_best_model_for_series(x_time_years, y_diameter_cm, bic_threshold=-2.0, require_cv_improvement=True):
    x = np.asarray(x_time_years, dtype=float)
    y = np.asarray(y_diameter_cm, dtype=float)
    n = len(y)

    out = {
        "n_points": int(n),
        "chosen_model": None,
        "chosen_slope_cm_per_year": float("nan"),
        "chosen_avg_slope_cm_per_year_over_span": float("nan"),
        "lin_aic": float("nan"),
        "lin_bic": float("nan"),
        "lin_adj_r2": float("nan"),
        "lin_slope_cm_per_year": float("nan"),
        "lin_loocv_rmse_cm": float("nan"),
        "quad_aic": float("nan"),
        "quad_bic": float("nan"),
        "quad_adj_r2": float("nan"),
        "quad_beta1_cm_per_year": float("nan"),
        "quad_beta2_cm_per_year2": float("nan"),
        "quad_term_p": float("nan"),
        "quad_loocv_rmse_cm": float("nan"),
        "delta_bic_quad_minus_lin": float("nan"),
        "delta_rmse_quad_minus_lin": float("nan"),
        "quadratic_eligible": bool(n >= 4),
        "cv_eligible_linear": bool(n >= 3),
        "cv_eligible_quadratic": bool(n >= 5),
    }

    if n < 2:
        return out

    lin = _fit_linear_ols(x, y)
    out["lin_aic"] = float(lin.aic)
    out["lin_bic"] = float(lin.bic)
    out["lin_adj_r2"] = float(lin.rsquared_adj)
    out["lin_slope_cm_per_year"] = float(lin.params[1]) if len(lin.params) > 1 else float("nan")
    out["lin_loocv_rmse_cm"] = _loocv_rmse(_fit_linear_ols, _predict_linear, x, y, min_n=3)

    chosen = "linear"
    chosen_slope = out["lin_slope_cm_per_year"]
    out["chosen_avg_slope_cm_per_year_over_span"] = float(chosen_slope)

    if n >= 4:
        quad = _fit_quadratic_ols(x, y)
        out["quad_aic"] = float(quad.aic)
        out["quad_bic"] = float(quad.bic)
        out["quad_adj_r2"] = float(quad.rsquared_adj)
        if len(quad.params) >= 3:
            out["quad_beta1_cm_per_year"] = float(quad.params[1])  # slope at baseline (t=0)
            out["quad_beta2_cm_per_year2"] = float(quad.params[2])
        if hasattr(quad, "pvalues") and len(quad.pvalues) >= 3:
            out["quad_term_p"] = float(quad.pvalues[2])

        out["quad_loocv_rmse_cm"] = _loocv_rmse(_fit_quadratic_ols, _predict_quadratic, x, y, min_n=5)

        out["delta_bic_quad_minus_lin"] = out["quad_bic"] - out["lin_bic"]
        out["delta_rmse_quad_minus_lin"] = out["quad_loocv_rmse_cm"] - out["lin_loocv_rmse_cm"]

        bic_ok = out["delta_bic_quad_minus_lin"] < bic_threshold
        cv_ok = True
        if require_cv_improvement:
            if not (out["cv_eligible_linear"] and out["cv_eligible_quadratic"]):
                cv_ok = False
            else:
                cv_ok = out["quad_loocv_rmse_cm"] < out["lin_loocv_rmse_cm"]

        if bic_ok and cv_ok:
            chosen = "quadratic"
            # For reporting/plots, refit a constrained quadratic so slope(t) is never negative
            # over the observed span. We then report the time-AVERAGED slope over the span.
            t0 = float(np.min(x))
            t1 = float(np.max(x))
            params_c, _ = fit_quadratic_constrained_nonnegative_slope(x, y)
            y0 = float(_predict_quadratic_params(params_c, t0))
            y1 = float(_predict_quadratic_params(params_c, t1))
            avg = (y1 - y0) / (t1 - t0) if (t1 > t0) else float("nan")
            chosen_slope = float(avg)
            out["chosen_avg_slope_cm_per_year_over_span"] = float(avg)

    out["chosen_model"] = chosen
    out["chosen_slope_cm_per_year"] = float(chosen_slope)
    return out


def _design_matrix_with_source_intercepts(t, source, quadratic=False):
    """Build a design matrix with per-source intercept offsets (dummy variables) and shared slope terms.

    Parameters
    ----------
    t : array-like
        Time in years since baseline.
    source : array-like
        Measurement source labels (e.g., HHP, outside, CTMR).
    quadratic : bool
        If True, include t^2 term.

    Returns
    -------
    X : np.ndarray
        Design matrix.
    col_names : list[str]
        Column names aligned to X.
    """
    t = np.asarray(t, dtype=float)
    src = pd.Series(source).astype(str)

    # baseline intercept
    cols = [np.ones_like(t), t]
    col_names = ["Intercept", "time_years"]

    if quadratic:
        cols.append(t**2)
        col_names.append("time_years2")

    # add source dummies (drop first to avoid perfect collinearity)
    dummies = pd.get_dummies(src, prefix="src", drop_first=True)
    for c in dummies.columns:
        cols.append(dummies[c].to_numpy(dtype=float))
        col_names.append(c)

    X = np.column_stack(cols)
    return X, col_names


def _fit_ols_with_source_intercepts(t, y, source, quadratic=False, enforce_nonneg_slope=False):
    """Fit OLS with shared slope terms and per-source intercepts."""
    X, col_names = _design_matrix_with_source_intercepts(t, source, quadratic=quadratic)
    y = np.asarray(y, dtype=float)

    # Optional slope constraints for quadratic: enforce non-negative derivative for all t>=0.
    # With t>=0, sufficient condition is beta_time >= 0 and beta_time2 >= 0.
    # We enforce by fallback: if violated, treat quadratic as ineligible.
    if enforce_nonneg_slope and quadratic:
        # First fit unconstrained; if violates, return None to signal ineligible.
        res = sm.OLS(y, X).fit()
        b1 = float(res.params[col_names.index("time_years")])
        b2 = float(res.params[col_names.index("time_years2")])
        if (b1 < 0) or (b2 < 0):
            return None, col_names
        return res, col_names

    res = sm.OLS(y, X).fit()
    return res, col_names


def _loocv_rmse_with_source_intercepts(t, y, source, quadratic=False, enforce_nonneg_slope=False):
    """Brute-force LOOCV RMSE for OLS models with source intercepts (small-n safe)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    src = np.asarray(source, dtype=object)

    n = len(y)
    if n < 3:
        return np.nan

    preds = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        res, col_names = _fit_ols_with_source_intercepts(
            t[mask], y[mask], src[mask], quadratic=quadratic, enforce_nonneg_slope=enforce_nonneg_slope
        )
        if res is None:
            return np.nan
        # Predict held-out
        X_i, col_i = _design_matrix_with_source_intercepts(np.array([t[i]]), np.array([src[i]]), quadratic=quadratic)
        # Ensure columns align (dummy presence may differ between train and test if a source appears only in held-out)
        # In that case, missing dummy columns are treated as 0.
        # We rebuild X_i to match train columns by name.
        train_cols = col_names
        x_vec = np.zeros((1, len(train_cols)), dtype=float)
        x_map = {name: j for j, name in enumerate(train_cols)}
        # Fill intercept/time(/time2)
        x_vec[0, x_map["Intercept"]] = 1.0
        x_vec[0, x_map["time_years"]] = float(t[i])
        if quadratic and "time_years2" in x_map:
            x_vec[0, x_map["time_years2"]] = float(t[i]**2)
        # Fill dummies if present in train
        # Build held-out dummies with same naming convention
        held_src = str(src[i])
        # Dummies are like 'src_<level>'; but with drop_first the baseline is implicit
        for name in train_cols:
            if name.startswith("src_"):
                level = name.split("src_", 1)[1]
                x_vec[0, x_map[name]] = 1.0 if held_src == level else 0.0

        preds.append(float(np.dot(x_vec, res.params)))

    rmse = float(np.sqrt(np.mean((y - np.array(preds)) ** 2)))
    return rmse


def choose_best_model_pooled_all_sources(df_series, time_col="time_years", y_col="diameter_cm", source_col="Source"):
    """Choose linear vs quadratic for a pooled (all-sources) patient×segment series,
    allowing per-source intercept shifts but a single shared trajectory (slope terms).

    Returns a dict with chosen model and slope metrics; selection uses BIC primarily and LOOCV RMSE secondarily.
    """
    df = df_series.dropna(subset=[time_col, y_col, source_col]).copy()
    if df.empty:
        return {}

    t = df[time_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    src = df[source_col].astype(str).to_numpy()

    n = len(df)
    out = {"n_points": int(n)}

    # Linear
    lin_res, lin_cols = _fit_ols_with_source_intercepts(t, y, src, quadratic=False)
    out.update({
        "lin_aic": float(lin_res.aic),
        "lin_bic": float(lin_res.bic),
        "lin_slope_cm_per_year": float(lin_res.params[lin_cols.index("time_years")]),
        "lin_loocv_rmse_cm": _loocv_rmse_with_source_intercepts(t, y, src, quadratic=False),
    })

    # Quadratic (eligible if >=4 points and >=3 unique times)
    quad_ok = (n >= 4) and (len(np.unique(t)) >= 3)
    out["quadratic_eligible"] = bool(quad_ok)

    quad_res = None
    if quad_ok:
        quad_res, quad_cols = _fit_ols_with_source_intercepts(
            t, y, src, quadratic=True, enforce_nonneg_slope=True
        )
        if quad_res is not None:
            b1 = float(quad_res.params[quad_cols.index("time_years")])
            b2 = float(quad_res.params[quad_cols.index("time_years2")])
            T = float(np.max(t) - np.min(t))
            avg_slope = b1 + b2 * T  # average derivative over span [0, T]
            out.update({
                "quad_aic": float(quad_res.aic),
                "quad_bic": float(quad_res.bic),
                "quad_beta1_cm_per_year": b1,
                "quad_beta2_cm_per_year2": b2,
                "quad_time_averaged_slope_cm_per_year": float(avg_slope),
                "quad_loocv_rmse_cm": _loocv_rmse_with_source_intercepts(t, y, src, quadratic=True, enforce_nonneg_slope=True),
            })
        else:
            # violated constraints -> treat as ineligible
            out["quadratic_eligible"] = False

    # Model choice
    chosen = "linear"
    chosen_slope = out.get("lin_slope_cm_per_year", np.nan)
    chosen_avg = chosen_slope

    if out.get("quadratic_eligible", False) and ("quad_bic" in out):
        delta_bic = out["quad_bic"] - out["lin_bic"]
        delta_rmse = out["quad_loocv_rmse_cm"] - out["lin_loocv_rmse_cm"]
        out["delta_bic_quad_minus_lin"] = float(delta_bic)
        out["delta_rmse_quad_minus_lin"] = float(delta_rmse)

        # Prefer quadratic only if meaningfully better BIC and not worse CV RMSE
        if (delta_bic <= -2.0) and (delta_rmse <= 0.0):
            chosen = "quadratic"
            chosen_slope = out["quad_beta1_cm_per_year"]  # baseline derivative at t=0
            chosen_avg = out["quad_time_averaged_slope_cm_per_year"]

    out["chosen_model"] = chosen
    out["chosen_slope_cm_per_year"] = float(chosen_slope) if chosen_slope is not None else np.nan
    out["chosen_avg_slope_cm_per_year_over_span"] = float(chosen_avg) if chosen_avg is not None else np.nan
    return out


def compute_best_model_pooled_with_source_intercepts(df_long, key_cols=None, time_col="time_years", y_col="diameter_cm", source_col="Source"):
    """Compute pooled best-model slopes for each patient×segment using all sources,
    allowing per-source intercept offsets but shared slope terms."""
    if key_cols is None:
        key_cols = (COL_PATIENT, "segment")
    rows = []
    for key, g in df_long.groupby(list(key_cols)):
        key = key if isinstance(key, tuple) else (key,)
        rec = {k: v for k, v in zip(key_cols, key)}
        m = choose_best_model_pooled_all_sources(g, time_col=time_col, y_col=y_col, source_col=source_col)
        if not m:
            continue
        rec.update(m)
        rows.append(rec)
    return pd.DataFrame(rows)
def choose_models_for_all_series(df_long, key_cols=None, time_col="time_years", y_col="diameter_cm",
                                 bic_threshold=-2.0, require_cv_improvement=True):
    # Default: per-series at patient×segment×Source level
    if key_cols is None:
        key_cols = ("patient_id", "segment", "Source")
    rows = []
    for key, g in df_long.groupby(list(key_cols), dropna=False):
        g = g.sort_values(time_col)
        x = g[time_col].to_numpy(float)
        y = g[y_col].to_numpy(float)
        d = choose_best_model_for_series(x, y, bic_threshold=bic_threshold, require_cv_improvement=require_cv_improvement)
        for c, v in zip(key_cols, key):
            d[c] = v
        rows.append(d)
    out = pd.DataFrame(rows)
    cols_front = list(key_cols) + ["n_points", "chosen_model", "chosen_slope_cm_per_year", "chosen_avg_slope_cm_per_year_over_span"]
    rest = [c for c in out.columns if c not in cols_front]
    return out[cols_front + rest]

def choose_models_for_all_series_pooled(df_long, key_cols=None, time_col="time_years", y_col="diameter_cm",
                                       bic_threshold=-2.0, require_cv_improvement=True):
    """Choose best model for pooled patient×segment trajectories (all Sources combined).

    This produces slopes that are directly comparable to pooled OLS/mixed slopes.
    """
    if key_cols is None:
        key_cols = ("patient_id", "segment")

    rows = []
    for key, g in df_long.groupby(list(key_cols), dropna=False):
        g = g.sort_values(time_col)
        x = g[time_col].to_numpy(float)
        y = g[y_col].to_numpy(float)
        d = choose_best_model_for_series(x, y, bic_threshold=bic_threshold, require_cv_improvement=require_cv_improvement)
        for c, v in zip(key_cols, key if isinstance(key, tuple) else (key,)):
            d[c] = v
        rows.append(d)

    out = pd.DataFrame(rows)
    cols_front = list(key_cols) + ["n_points", "chosen_model", "chosen_slope_cm_per_year", "chosen_avg_slope_cm_per_year_over_span"]
    rest = [c for c in out.columns if c not in cols_front]
    return out[cols_front + rest]


# ===============================
# CONFIG
# ===============================

# Input / output paths
INPUT_CSV = "aorta_data_long_from_AllHHP2025_mapped.csv"
OUTPUT_LONG_CSV = "aorta_data_long_with_outliers.csv"
OUTPUT_QC_SUMMARY_CSV = "qc_summary_by_patient_segment.csv"
OUTPUT_FLAGGED_CSV = "flagged_measurements.csv"
OUTPUT_SLOPE_CSV = "patient_slopes_by_segment.csv"
OUTPUT_SLOPE_COMPARISON_CSV = "patient_slopes_ols_vs_mixed.csv"
OUTPUT_PROGRESSORS_CSV = "patient_progression_categories.csv"
OUTPUT_BEST_MODEL_SERIES_CSV = "patient_best_model_slopes_by_series.csv"
OUTPUT_BEST_MODEL_POOLED_CSV = "patient_best_model_slopes_pooled_by_patient_segment.csv"

PLOTS_DIR = "plots"   # base directory for all patient/segment plots

# Ranking (merged from ranker.py)
RANK_INPUT_CSV = "high_risk_combined_delta_AHI.csv"
RANK_OUTPUT_CSV = "ranked_participants.csv"


OUTPUT_RANKED_PARTICIPANTS = RANK_OUTPUT_CSV  # backward-compatible alias
OUTPUT_CSV = RANK_OUTPUT_CSV
# Column names in long input
COL_PATIENT = "patient_id"
COL_DATE = "study_date"
COL_HEIGHT = "height_cm"     # after the one-time rename from HT_cm
COL_SEGMENT = "segment"
COL_DIAM = "diameter_cm"
COL_SOURCE = "Source"        # optional; used for plot labeling if present

# Segment-specific random-effects structures:
# "~time_years" = random intercept + random slope
# "1"           = random intercept only
SEGMENT_RE_FORMULA = {
    "sinus": "~time_years",      # TS: allow random slope
    "ascending": "~time_years",  # random intercept + slope
}

# Outlier thresholds (tailored for high-risk TS cohort)
MAX_GROWTH_CM_PER_YEAR = 1.0    # strong rate-based outlier threshold
MAX_DECREASE_CM = -0.3          # large negative change (cm)

RESID_Z_SUSPECT = 2.5
RESID_Z_OUTLIER = 3.0
MIN_ABS_RESID_CM = 0.3          # minimum absolute residual to be eligible as outlier

MIN_MEASUREMENTS_FOR_RATES = 2
MIN_PATIENTS_FOR_MIXEDLM = 2    # per segment

# "Review band" for elevated but possibly real growth
REVIEW_GROWTH_CM_PER_YEAR = 0.5


# ===============================
# CORE FUNCTIONS
# ===============================

def load_long(csv_path: str) -> pd.DataFrame:
    """Load long-format data and ensure required columns exist."""
    df_long = pd.read_csv(csv_path)

    if COL_DATE not in df_long.columns:
        raise ValueError(f"Expected date column '{COL_DATE}' not found.")
    df_long[COL_DATE] = pd.to_datetime(df_long[COL_DATE])

    required = [COL_PATIENT, COL_DATE, COL_HEIGHT, COL_SEGMENT, COL_DIAM]
    missing = [c for c in required if c not in df_long.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing diameters
    df_long = df_long.dropna(subset=[COL_DIAM])

    # Standardize names used in formulas
    if COL_DIAM != "diameter_cm":
        df_long["diameter_cm"] = df_long[COL_DIAM]
    if COL_SEGMENT != "segment":
        df_long["segment"] = df_long[COL_SEGMENT]
    if COL_HEIGHT != "height_cm":
        df_long["height_cm"] = df_long[COL_HEIGHT]

    # Sort for downstream operations
    df_long = df_long.sort_values([COL_PATIENT, "segment", COL_DATE])

    return df_long


def add_time_since_baseline(df_long: pd.DataFrame) -> pd.DataFrame:
    """Add time_years = years since first measurement per patient (shared across segments)."""
    df_long = df_long.copy()
    baseline = df_long.groupby(COL_PATIENT)[COL_DATE].transform("min")
    df_long["time_years"] = (df_long[COL_DATE] - baseline).dt.days / 365.25
    return df_long


def mixedlm_predict_conditional(mdf, df, group_col):
    """Conditional predictions for statsmodels MixedLMResults (fixed + random effects)."""
    import numpy as np
    from patsy import build_design_matrices

    design_info = mdf.model.data.design_info
    X = build_design_matrices([design_info], df, return_type="dataframe")[0]
    fe = mdf.fe_params
    X = X.reindex(columns=fe.index, fill_value=0.0)
    fixed = X.to_numpy() @ fe.to_numpy()

    preds = fixed.astype(float).copy()
    re_dict = mdf.random_effects
    groups = df[group_col].tolist()

    for i, g in enumerate(groups):
        re = re_dict.get(g)
        if re is None:
            continue
        if "Group" in re.index:
            preds[i] += float(re["Group"])
        elif "Intercept" in re.index:
            preds[i] += float(re["Intercept"])
        for name in re.index:
            if name in ("Group", "Intercept"):
                continue
            if name in df.columns:
                preds[i] += float(re[name]) * float(df.iloc[i][name])
    return preds

def fit_mixed_effects_per_segment(df_long: pd.DataFrame):
    """
    Fit segment-specific mixed-effects models with height as fixed covariate.

    Model for segment s:
      diameter ~ time_years + height_cm
      random structure given by SEGMENT_RE_FORMULA[s]
    """
    df_long = df_long.copy()
    models = {}

    for seg in df_long["segment"].unique():
        seg_mask = df_long["segment"] == seg
        df_seg = df_long.loc[seg_mask].copy()

        n_patients = df_seg[COL_PATIENT].nunique()
        if n_patients < MIN_PATIENTS_FOR_MIXEDLM:
            print(f"[WARN] Segment '{seg}': only {n_patients} patient(s); skipping mixed-effects fit.")
            df_long.loc[seg_mask, "fitted_diameter_cm"] = np.nan
            models[seg] = None
            continue

        re_formula = SEGMENT_RE_FORMULA.get(seg, "~time_years")
        print(f"[INFO] Fitting MixedLM for segment '{seg}' with re_formula='{re_formula}'")

        df_fit = expand_by_frequency_weight(df_seg, w_col='source_weight_freq')
        md = smf.mixedlm(
            "diameter_cm ~ time_years + height_cm",
            df_fit,
            groups=df_seg[COL_PATIENT],
            re_formula=re_formula
        )
        mdf = md.fit(method="lbfgs", maxiter=200, reml=True)
        models[seg] = mdf

        # NOTE: predict expects the same dataframe columns used in the formula.
        df_long.loc[seg_mask, "fitted_diameter_cm"] = mixedlm_predict_conditional(mdf, df_seg, group_col=COL_PATIENT)

    df_long["residual_cm"] = df_long["diameter_cm"] - df_long["fitted_diameter_cm"]
    return df_long, models


def add_residual_zscores(df_long: pd.DataFrame) -> pd.DataFrame:
    """Compute segment-level residual SD and standardized residuals."""
    df_long = df_long.copy()
    seg_sd = (
        df_long.groupby("segment")["residual_cm"]
        .agg(lambda x: np.nanstd(x.values, ddof=1))
        .rename("resid_sd")
    )
    df_long = df_long.merge(seg_sd, on="segment", how="left")
    df_long["resid_z"] = df_long["residual_cm"] / df_long["resid_sd"]
    return df_long


def add_interval_rates(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient/segment, compute:
      delta_cm, delta_years, rate_cm_per_year
    attached to the later timepoint of each interval.
    """
    df_long = df_long.copy()

    df_long["delta_cm"] = np.nan
    df_long["delta_years"] = np.nan
    df_long["rate_cm_per_year"] = np.nan

    df_long = df_long.sort_values([COL_PATIENT, "segment", COL_DATE])

    for (pid, seg), grp in df_long.groupby([COL_PATIENT, "segment"]):
        idx = grp.index
        if len(grp) < MIN_MEASUREMENTS_FOR_RATES:
            continue

        diam = grp["diameter_cm"].values
        time = grp["time_years"].values

        delta_d = np.diff(diam)
        delta_t = np.diff(time)

        df_long.loc[idx[1:], "delta_cm"] = delta_d
        df_long.loc[idx[1:], "delta_years"] = delta_t
        df_long.loc[idx[1:], "rate_cm_per_year"] = delta_d / delta_t

    return df_long


def build_patient_ols_slope_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Per-patient, per-segment OLS slopes using only that patient's data.

    Returns:
      - patient_id
      - segment
      - n_echoes
      - ols_intercept_cm
      - ols_slope_cm_per_year
      - ols_slope_confidence: 'high' if n_echoes >= 3, else 'low'
    """
    rows = []

    df = df_long.copy()
    df = df.sort_values([COL_PATIENT, "segment", "time_years"])

    for (pid, seg), grp in df.groupby([COL_PATIENT, "segment"]):
        g = grp.dropna(subset=["diameter_cm", "time_years"]).copy()
        n = len(g)
        if n == 0:
            continue

        ols_slope = np.nan
        ols_intercept = np.nan

        if n >= 2 and g["time_years"].nunique() >= 2:
            t = g["time_years"].values
            y = g["diameter_cm"].values

            A = np.vstack([t, np.ones_like(t)]).T
            b, a = np.linalg.lstsq(A, y, rcond=None)[0]  # y ≈ a + b*t

            ols_slope = b
            ols_intercept = a

        confidence = "high" if n >= 3 else "low"

        rows.append(
            {
                COL_PATIENT: pid,
                "segment": seg,
                "n_echoes": n,
                "ols_intercept_cm": ols_intercept,
                "ols_slope_cm_per_year": ols_slope,
                "ols_slope_confidence": confidence,
            }
        )

    ols_df = pd.DataFrame(rows).sort_values([COL_PATIENT, "segment"])
    return ols_df


def classify_progression(df_slopes: pd.DataFrame) -> pd.DataFrame:
    """
    Assign progression categories using a *single pooled* best-model slope per patient×segment,
    with clinical thresholds expressed in cm/year.

    Rules:
      - Use pooled best-model slope if available (best_model_slope_cm_per_year_pooled); otherwise fall back to OLS.
      - High-confidence requires >=3 usable studies after QC (n_echoes >= 3).
      - For classification only, negative slopes are floored to 0 for biological interpretability.
      - Categories: rapid >= 0.30 cm/yr (>=3 mm/yr); mild 0.10–<0.30 cm/yr (1–<3 mm/yr); stable <0.10 cm/yr (<1 mm/yr);
                   uncertain if <3 studies or missing slope.
    """
    df = df_slopes.copy()

    # Choose slope source for classification (pooled best-model preferred)
    slope_col = "best_model_slope_cm_per_year_pooled" if "best_model_slope_cm_per_year_pooled" in df.columns else "ols_slope_cm_per_year"
    df["progression_slope_source"] = slope_col

    # Confidence based on usable study count (after QC/outlier dropping)
    df["high_confidence"] = df.get("n_echoes", np.nan).fillna(0).astype(float) >= 3

    # Pull slope, floor negatives for classification only
    df["progression_slope_cm_per_year_raw"] = df.get(slope_col)
    df["progression_slope_cm_per_year"] = df["progression_slope_cm_per_year_raw"]
    df.loc[df["progression_slope_cm_per_year"].notna() & (df["progression_slope_cm_per_year"] < 0), "progression_slope_cm_per_year"] = 0.0

    # Thresholds in cm/year
    rapid_thr = 0.30  # 3 mm/yr
    mild_thr = 0.10   # 1 mm/yr

    cat = []
    for hc, s in zip(df["high_confidence"].tolist(), df["progression_slope_cm_per_year"].tolist()):
        if (not hc) or (s is None) or (isinstance(s, float) and np.isnan(s)):
            cat.append("uncertain")
        elif s >= rapid_thr:
            cat.append("rapid")
        elif s >= mild_thr:
            cat.append("mild")
        else:
            cat.append("stable")
    df["progression_category"] = cat
    return df
def flag_outliers(df_long: pd.DataFrame) -> pd.DataFrame:
    """Create residual-based and rate-based outlier flags, plus combined flag."""
    df_long = df_long.copy()

    df_long["outlier_resid_flag"] = "none"

    cond_suspect = (
        df_long["resid_z"].abs() >= RESID_Z_SUSPECT
    ) & (df_long["residual_cm"].abs() >= MIN_ABS_RESID_CM)

    cond_outlier = (
        df_long["resid_z"].abs() >= RESID_Z_OUTLIER
    ) & (df_long["residual_cm"].abs() >= MIN_ABS_RESID_CM)

    df_long.loc[cond_suspect, "outlier_resid_flag"] = "suspect"
    df_long.loc[cond_outlier, "outlier_resid_flag"] = "outlier"

    df_long["rate_review_flag"] = False
    df_long["outlier_rate_flag"] = False

    cond_rate_review = (
        (df_long["delta_years"] > 0)
        & (df_long["rate_cm_per_year"] > REVIEW_GROWTH_CM_PER_YEAR)
        & (df_long["rate_cm_per_year"] <= MAX_GROWTH_CM_PER_YEAR)
    )

    cond_rate_strong = (
        (df_long["delta_years"] > 0)
        & (
            (df_long["rate_cm_per_year"] > MAX_GROWTH_CM_PER_YEAR) |
            (df_long["delta_cm"] <= MAX_DECREASE_CM)
        )
    )

    df_long.loc[cond_rate_review, "rate_review_flag"] = True
    df_long.loc[cond_rate_strong, "outlier_rate_flag"] = True

    df_long["any_outlier_flag"] = (
        df_long["outlier_rate_flag"]
        | df_long["outlier_resid_flag"].isin(["suspect", "outlier"])
    )

    return df_long


def summarize_qc(df_long: pd.DataFrame) -> pd.DataFrame:
    """Per-patient / per-segment QC summary. If Source is present, stratify by Source as well."""
    df = df_long.copy()

    if COL_SOURCE in df.columns:
        group_cols = [COL_SOURCE, COL_PATIENT, "segment"]
    else:
        group_cols = [COL_PATIENT, "segment"]

    summary = (
        df.groupby(group_cols)
        .agg(
            n_measurements=("diameter_cm", "count"),
            n_resid_suspect=("outlier_resid_flag", lambda x: (x == "suspect").sum()),
            n_resid_outlier=("outlier_resid_flag", lambda x: (x == "outlier").sum()),
            n_rate_outlier=("outlier_rate_flag", lambda x: x.sum()),
            any_outlier=("any_outlier_flag", lambda x: x.any()),
            max_rate_cm_per_year=("rate_cm_per_year", "max"),
            min_rate_cm_per_year=("rate_cm_per_year", "min")
        )
        .reset_index()
    )
    return summary


def build_patient_mixed_slope_table(df_long: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Per-patient, per-segment mixed-model intercepts and slopes.

    If Source is present, tags dominant Source per patient/segment.
    """
    df = df_long.copy()
    rows = []

    n_echoes = (
        df.groupby([COL_PATIENT, "segment"])["diameter_cm"]
        .size()
        .rename("n_echoes")
    )

    source_map = None
    if COL_SOURCE in df.columns:
        def dominant_source(x):
            m = x.mode()
            if len(m) == 0:
                return np.nan
            if len(m) == 1:
                return m.iloc[0]
            return "mixed"

        source_map = (
            df.groupby([COL_PATIENT, "segment"])[COL_SOURCE]
            .agg(dominant_source)
        )

    for seg, mdf in models.items():
        if mdf is None:
            continue

        fe = mdf.fe_params
        beta0 = fe.get("Intercept", np.nan)
        beta1 = fe.get("time_years", np.nan)

        re_dict = mdf.random_effects

        for pid, re in re_dict.items():
            b0j = re.get("Intercept", 0.0)
            b1j = re.get("time_years", 0.0) if "time_years" in re.index else 0.0

            intercept_j = beta0 + b0j
            slope_j = beta1 + b1j

            n = n_echoes.get((pid, seg), 0)
            confidence = "high" if n >= 3 else "low"

            row = {
                COL_PATIENT: pid,
                "segment": seg,
                "n_echoes": n,
                "intercept_cm": intercept_j,
                "slope_cm_per_year": slope_j,
                "slope_confidence": confidence,
            }

            if source_map is not None:
                row[COL_SOURCE] = source_map.get((pid, seg), np.nan)

            rows.append(row)

    slopes_df = pd.DataFrame(rows)

    sort_cols = [COL_PATIENT, "segment"]
    if COL_SOURCE in slopes_df.columns:
        sort_cols = [COL_SOURCE] + sort_cols

    slopes_df = slopes_df.sort_values(sort_cols)
    return slopes_df


def apply_nonnegative_slope_floor(slopes_df: pd.DataFrame, floor=0.0) -> pd.DataFrame:
    """Preserve raw slopes, then floor negative slopes to zero."""
    df = slopes_df.copy()

    df["ols_slope_cm_per_year_raw"] = df["ols_slope_cm_per_year"]
    df["mixed_slope_cm_per_year_raw"] = df["slope_cm_per_year"]

    df["ols_slope_cm_per_year"] = df["ols_slope_cm_per_year"].clip(lower=floor)
    df["slope_cm_per_year"] = df["slope_cm_per_year"].clip(lower=floor)

    return df


# ===============================
# PLOTTING (UPDATED: Source categories labeled)
# ===============================

def _source_category(val) -> str:
    """
    Map raw Source strings into the 3 categories requested for plotting:
      - CTMR
      - Outside
      - HHP
    Everything else collapses to 'Other' (still shown if present).
    """
    if pd.isna(val):
        return "Other"
    s = str(val).strip().lower()
    if "ctmr" in s or "ct/mr" in s or "ct" in s or "mr" in s:
        # NOTE: we keep this broad so CTMR_ascending_out_cal is captured
        if "hhp" in s:
            return "HHP"
        return "CTMR"
    if "outside" in s or "external" in s or "clinic" in s:
        return "Outside"
    if "hhp" in s or "healthy heart" in s:
        return "HHP"
    return "Other"


def plot_patient(df_long: pd.DataFrame, patient_id, segment="sinus", save_dir=PLOTS_DIR):
    """
    Diagnostic plot for one patient & one segment:
      - Observed diameters, labeled by Source category (CTMR / Outside / HHP)
      - Mixed-effects fitted trend
      - Per-patient OLS line
      - Outliers marked

    Saves PNG to: {save_dir}/{segment}/patient_{patient_id}.png
    """
    mask = (df_long[COL_PATIENT] == patient_id) & (df_long["segment"] == segment)
    sub = df_long.loc[mask].sort_values("time_years").copy()

    # Credible data checks
    if sub.empty:
        return
    if sub["diameter_cm"].notna().sum() < 2:
        return
    if sub["time_years"].nunique() < 2:
        return
    if not sub["fitted_diameter_cm"].notna().any():
        return

    seg_dir = Path(save_dir) / segment
    seg_dir.mkdir(parents=True, exist_ok=True)
    out_path = seg_dir / f"patient_{patient_id}.png"

    # Build figure
    plt.figure(figsize=(6.6, 4.6))

    # Mixed-effects fitted line
    plt.plot(
        sub["time_years"],
        sub["fitted_diameter_cm"],
        "-",
        label="Mixed-effects fitted",
    )

    # Per-patient OLS line
    if len(sub) >= 2:
        t = sub["time_years"].values
        y = sub["diameter_cm"].values
        A = np.vstack([t, np.ones_like(t)]).T
        b, a = np.linalg.lstsq(A, y, rcond=None)[0]
        t_line = np.linspace(t.min(), t.max(), 50)
        y_line = a + b * t_line
        plt.plot(t_line, y_line, "--", label="Per-patient OLS")

    # Observed points: label by Source category with distinct markers
    if COL_SOURCE in sub.columns:
        sub["source_category"] = sub[COL_SOURCE].apply(_source_category)
    else:
        sub["source_category"] = "Other"

    marker_map = {
        "HHP": "o",
        "Outside": "^",
        "CTMR": "s",
        "Other": "o",
    }

    # Best-model overlay (POOLED across all sources): linear vs quadratic, conservative selection.
    # This is for visualization/robustness only; clinical progression classification remains OLS-based elsewhere.
    x_all = sub["time_years"].to_numpy(dtype=float)
    y_all = sub["diameter_cm"].to_numpy(dtype=float)
    if np.isfinite(x_all).sum() >= 2 and np.isfinite(y_all).sum() >= 2 and sub["time_years"].nunique() >= 2:
        bm = choose_best_model_for_series(x_all, y_all, bic_threshold=-2.0, require_cv_improvement=True)
        t_line = np.linspace(float(np.nanmin(x_all)), float(np.nanmax(x_all)), 80)

        if bm.get("chosen_model") == "quadratic" and np.isfinite(bm.get("quad_beta2_cm_per_year2", np.nan)):
            # Constrain the quadratic so slope(t) is never negative over the observed span.
            params_c, constrained = fit_quadratic_constrained_nonnegative_slope(x_all, y_all)
            y_line = _predict_quadratic_params(params_c, t_line)
            lbl = "Best-model fit (pooled: quadratic, constrained)" if constrained else "Best-model fit (pooled: quadratic)"
        else:
            lin = _fit_linear_ols(x_all, y_all)
            y_line = _predict_linear(lin, t_line)
            lbl = "Best-model fit (pooled: linear)"

        plt.plot(t_line, y_line, ":", label=lbl)

    # Plot categories separately so legend shows CTMR/Outside/HHP
    for cat, gcat in sub.groupby("source_category"):
        plt.scatter(
            gcat["time_years"],
            gcat["diameter_cm"],
            marker=marker_map.get(cat, "o"),
            label=f"Observed ({cat})",
        )

    # Overlay outliers with a clear marker (no color specification needed)
    out = sub[sub["any_outlier_flag"]].copy()
    if not out.empty:
        plt.scatter(out["time_years"], out["diameter_cm"], marker="x", label="Flagged outlier")

        # Optional text tags (kept from original)
        for _, row in out.iterrows():
            lbl = "R" if row["outlier_resid_flag"] in ["suspect", "outlier"] else ""
            if bool(row.get("outlier_rate_flag", False)):
                lbl = lbl + "Δ"
            plt.text(row["time_years"], row["diameter_cm"], lbl or "*", fontsize=8)

    plt.xlabel("Time since baseline (years)")
    plt.ylabel("Diameter (cm)")
    plt.title(f"Patient {patient_id} – {segment}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_all_plots(df_long, save_dir=PLOTS_DIR):
    """Generate plots for all patients × all segments with credible data."""
    patients = df_long[COL_PATIENT].unique()
    segments = df_long["segment"].unique()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        for pid in patients:
            plot_patient(df_long, pid, seg, save_dir)

    print(f"[INFO] Plots stored under: {save_dir}")

# ===============================
# HIGH-RISK DELTA vs AHI OUTPUTS
# ===============================

def make_high_risk_files_from_long(
    df_long: pd.DataFrame,
    out_prefix: str = "high_risk",
    include_segments=("ascending", "sinus"),
):
    """

    For each patient × segment:
      1) Identify the timepoint with the maximum observed diameter.
      2) Expected diameter = MixedLM fitted diameter at that same timepoint.
      3) delta = observed - expected (cm)
      4) AHI = observed / height_m (cm/m)  [height_m = height_cm/100]
      5) Rank within segment by delta (desc) and by AHI (desc), then combined rank = sum.
      6) 'high_confidence' = ✓ if n_meas >= 3, else blank.
    """
    d = df_long.copy()
    d["segment"] = d["segment"].astype(str).str.lower().str.strip()
    d = d[d["segment"].isin([s.lower() for s in include_segments])].copy()

    # Required columns (these are produced by the QC pipeline after MixedLM fit)
    req = ["patient_id", "segment", "study_date", "diameter_cm", "height_cm", "fitted_diameter_cm"]
    missing = [c for c in req if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns for high-risk module: {missing}")

    d["study_date"] = pd.to_datetime(d["study_date"], errors="coerce")
    d = d.dropna(subset=["patient_id", "segment", "diameter_cm", "height_cm", "fitted_diameter_cm", "study_date"])

    # n measurements (confidence indicator)
    n_meas = d.groupby(["patient_id", "segment"])["diameter_cm"].size().rename("n_meas")

    # Find row of maximum observed diameter per patient×segment
    idx = d.groupby(["patient_id", "segment"])["diameter_cm"].idxmax()
    top = d.loc[idx].copy()
    top = top.merge(n_meas.reset_index(), on=["patient_id", "segment"], how="left")

    top["expected_cm"] = top["fitted_diameter_cm"].astype(float)
    top["observed_cm"] = top["diameter_cm"].astype(float)
    top["delta_cm"] = top["observed_cm"] - top["expected_cm"]
        # Aortic Height Index (AHI): diameter (cm) / height (m)
    top["height_m"] = top["height_cm"].astype(float) / 100.0
    top["AHI_cm_per_m"] = top["observed_cm"] / top["height_m"]
    # Legacy scale (cm/cm) retained for backward compatibility
    top["AHI_cm_per_cm"] = top["AHI_cm_per_m"] / 100.0
    top["high_confidence"] = np.where(top["n_meas"] >= 3, "✓", "")

    # Ranks within segment (descending risk/priority)
    top["rank_delta_delta"] = top.groupby("segment")["delta_cm"].rank(ascending=False, method="dense").astype(int)
    top["rank_AHI_delta"] = top.groupby("segment")["AHI_cm_per_m"].rank(ascending=False, method="dense").astype(int)
    top["combined_rank"] = top["rank_delta_delta"] + top["rank_AHI_delta"]

    # Sort for export
    top = top.sort_values(["segment", "combined_rank", "rank_delta_delta", "rank_AHI_delta"]).reset_index(drop=True)

    ascending = top[top["segment"] == "ascending"].copy()
    sinus = top[top["segment"] == "sinus"].copy()
    combined = top.copy()

    keep_cols = [
        "patient_id", "segment",
        "observed_cm", "expected_cm", "delta_cm",
        "height_cm", "height_m", "AHI_cm_per_m", "AHI_cm_per_cm",
        "n_meas", "high_confidence",
        "rank_delta_delta", "rank_AHI_delta", "combined_rank",
    ]
    # Ensure all expected columns exist (robust to upstream changes)
    for _c in keep_cols:
        if _c not in ascending.columns:
            ascending[_c] = pd.NA
        if _c not in sinus.columns:
            sinus[_c] = pd.NA
        if _c not in combined.columns:
            combined[_c] = pd.NA

    asc_path = f"{out_prefix}_ascending_delta_AHI.csv"
    sin_path = f"{out_prefix}_sinus_delta_AHI.csv"
    comb_path = f"{out_prefix}_combined_delta_AHI.csv"

    ascending[keep_cols].to_csv(asc_path, index=False)
    sinus[keep_cols].to_csv(sin_path, index=False)
    combined[keep_cols].to_csv(comb_path, index=False)

    # Participant summaries (human-readable)
    def _fmt_row(r):
        pid = int(r["patient_id"])
        seg = r["segment"]
        return (
            f"Patient {pid} – {seg} segment: maximum observed diameter {r['observed_cm']:.2f} cm "
            f"vs expected {r['expected_cm']:.2f} cm (delta {r['delta_cm']:+.2f} cm), "
            f"height {r['height_cm']:.1f} cm, AHI {r['AHI_cm_per_m']:.2f} cm/m. "
            f"Within the {seg} group this patient ranks {int(r['rank_delta_delta'])} for observed–expected enlargement "
            f"and {int(r['rank_AHI_delta'])} for AHI, based on "
            f"{'high-confidence' if r['high_confidence']=='✓' else 'limited'} longitudinal data."
        )

    lines = []
    for _, r in ascending.iterrows():
        lines.append(_fmt_row(r))
    for _, r in sinus.iterrows():
        lines.append(_fmt_row(r))

    summary_path = f"{out_prefix}_participant_summaries.txt"
    Path(summary_path).write_text("\n".join(lines))

    # Scatter plot (delta vs AHI)
    plot_path = f"{out_prefix}_delta_vs_AHI_scatter.png"
    plt.figure()
    for seg, g in combined.groupby("segment"):
        plt.scatter(g["delta_cm"], g["AHI_cm_per_m"], label=seg)
        for _, r in g.iterrows():
            plt.text(r["delta_cm"], r["AHI_cm_per_m"], str(int(r["patient_id"])), fontsize=9)
    plt.title("High-risk participants by delta vs AHI")
    plt.xlabel("Observed – expected diameter (cm)")
    plt.ylabel("Aortic height index (cm/m)")
    plt.legend(title="Segment")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"[INFO] Wrote: {asc_path}, {sin_path}, {comb_path}, {summary_path}, {plot_path}")
    return {
        "ascending_csv": asc_path,
        "sinus_csv": sin_path,
        "combined_csv": comb_path,
        "summaries_txt": summary_path,
        "scatter_png": plot_path,
    }

# ===============================
# RANKING
# ===============================

def generate_ranked_participant_list(
    combined_csv: str,
    out_csv: str = OUTPUT_RANKED_PARTICIPANTS,
    ahi_tolerance_cm_per_m: float = 0.05,
):
    """
    Rank participants by *size-first* risk prioritization:
      1) Larger baseline AHI (cm/m) supersedes growth.
      2) For tied / near-tied AHI (within ahi_tolerance_cm_per_m), faster growth supersedes.

    Implementation:
      - Create an AHI "bin" by rounding to the nearest tolerance (default 0.05 cm/m).
      - Rank AHI bins descending (dense rank).
      - Within each AHI bin, rank growth (delta_cm) descending (dense rank).
      - Combine lexicographically by sorting (ahi_rank, delta_rank).

    Notes:
      - This is intentionally *not* a weighted sum: AHI dominates by design.
      - If delta_cm is missing, treat as worst (place last within bin).
    """
    df = pd.read_csv(combined_csv)

    required = {"patient_id", "segment", "AHI_cm_per_m", "delta_cm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for ranking: {sorted(missing)}")

    # Ensure numeric
    df["AHI_cm_per_m"] = pd.to_numeric(df["AHI_cm_per_m"], errors="coerce")
    df["delta_cm"] = pd.to_numeric(df["delta_cm"], errors="coerce")

    # AHI bin for "near-identical" handling
    tol = float(ahi_tolerance_cm_per_m)
    if tol <= 0:
        tol = 0.05
    df["ahi_bin"] = (df["AHI_cm_per_m"] / tol).round() * tol

    # Dense ranks (1 = highest risk)
    df["ahi_rank"] = df.groupby("segment")["ahi_bin"].rank(method="dense", ascending=False)
    # Put NaN delta last within-bin by filling with -inf for descending rank calc
    df["_delta_for_rank"] = df["delta_cm"].fillna(float("-inf"))
    df["delta_rank_within_ahi"] = df.groupby(["segment", "ahi_rank"])["_delta_for_rank"].rank(
        method="dense", ascending=False
    )
    df.drop(columns=["_delta_for_rank"], inplace=True)

    # Final sort: lexicographic (AHI first, then delta)
    df = df.sort_values(
        by=["segment", "ahi_rank", "delta_rank_within_ahi", "AHI_cm_per_m", "delta_cm"],
        ascending=[True, True, True, False, False],
        kind="mergesort",
    )

    # Add a convenient overall rank per segment
    df["rank_size_then_growth"] = df.groupby("segment").cumcount() + 1

    # Keep original columns plus ranks (do not drop anything that downstream might want)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved ranked participants to: {out_csv} (size-first ranking; AHI tol={tol:g} cm/m)")
    return df
def main():
    print("[INFO] Loading long-format data...")
    df_long = load_long(INPUT_CSV)
    print(f"[INFO] Loaded {len(df_long)} rows (long).")

    df_long = add_time_since_baseline(df_long)
    df_long, models = fit_mixed_effects_per_segment(df_long)
    df_long = add_residual_zscores(df_long)
    df_long = add_interval_rates(df_long)
    df_long = flag_outliers(df_long)

    # ------------------------------------------------------------------
    # IMPORTANT: downstream slope estimation and model selection should not be
    # dominated by measurements already flagged as outliers by QC.
    # We therefore keep *all* points for plotting/auditability, but we exclude
    # flagged outliers from:
    #   (1) per-series model selection (linear vs quadratic)
    #   (2) pooled (all-sources) best-model fitting
    #   (3) OLS slope estimation used for progression classification
    # MixedLM is still fit on the full dataset to support residual-based QC.
    # ------------------------------------------------------------------
    if "any_outlier_flag" in df_long.columns:
        df_fit = df_long.loc[~df_long["any_outlier_flag"].fillna(False)].copy()
    else:
        df_fit = df_long.copy()

    # Model fit summary (correct placement inside main)
    import numpy as np
    print("[INFO] Model fit summary (per segment):")
    for seg, m in models.items():
        if m is None:
            print(f"  - {seg}: model not fitted (too few patients)")
        else:
            n_patients = len(np.unique(m.model.groups))
            print(f"  - {seg}:")
            print(f"      converged: {m.converged}")
            print(f"      n_obs: {m.nobs}")
            print(f"      n_patients: {n_patients}")
            print(f"      fixed effects:\n{m.fe_params}")
            print(f"      random effects covariance:\n{m.cov_re}")
            
    # Rebuild high-risk delta vs AHI outputs (used by ranking)
    make_high_risk_files_from_long(df_long, out_prefix="high_risk")


    # Per-series "best model" slopes (linear vs quadratic) for robustness/sensitivity analyses.
    # NOTE: Clinical progression classification remains based on OLS slopes (see classify_progression()).
    best_model_series = choose_models_for_all_series(df_fit)

    # Pooled best-model is a single fit per patient×segment using *all sources*;
    # this is used for sensitivity/robustness summaries (not for classification).
    best_model_pooled = choose_models_for_all_series(df_fit, key_cols=(COL_PATIENT, "segment"))

    # QC summary
    qc_summary = summarize_qc(df_long)

    # Attach per-series best-model slopes summarized at patient×segment level (median across sources).
    try:
        bm_agg = best_model_series.groupby([COL_PATIENT, "segment"], dropna=False)["chosen_slope_cm_per_year"].median().reset_index()
        bm_agg = bm_agg.rename(columns={"chosen_slope_cm_per_year": "best_model_slope_cm_per_year_median_across_sources"})
        qc_summary = qc_summary.merge(bm_agg, on=[COL_PATIENT, "segment"], how="left")
    except Exception as e:
        print(f"[WARN] Could not merge best-model slopes into QC summary: {e}")

    # Patient-level slopes: mixed-model and OLS
    mixed_slopes_df = build_patient_mixed_slope_table(df_long, models)
    # OLS slopes (used for clinical progression classification) exclude flagged outliers.
    ols_slopes_df = build_patient_ols_slope_table(df_fit)

    # Merge mixed and OLS slopes
    combined_slopes = pd.merge(
        mixed_slopes_df,
        ols_slopes_df[[COL_PATIENT, "segment", "ols_intercept_cm", "ols_slope_cm_per_year", "ols_slope_confidence"]],
        on=[COL_PATIENT, "segment"],
        how="outer",
        suffixes=("", "_ols_only")
    )

    combined_slopes["slope_diff_ols_minus_mixed"] = (
        combined_slopes["ols_slope_cm_per_year"] - combined_slopes["slope_cm_per_year"]
    )


    # Attach best-model slope summary (median across sources) for symmetry in downstream tables
    try:
        bm_pool2 = best_model_pooled[[COL_PATIENT, "segment", "chosen_slope_cm_per_year"]].rename(
            columns={"chosen_slope_cm_per_year": "best_model_slope_cm_per_year_pooled"}
        )
        combined_slopes = combined_slopes.merge(
            bm_pool2,
            on=[COL_PATIENT, "segment"],
            how="left"
        )
        # keep median-across-sources as a sensitivity metric
        bm_med2 = best_model_series.groupby([COL_PATIENT, "segment"], dropna=False)["chosen_slope_cm_per_year"].median().reset_index()
        bm_med2 = bm_med2.rename(columns={"chosen_slope_cm_per_year": "best_model_slope_cm_per_year_median_across_sources"})
        combined_slopes = combined_slopes.merge(bm_med2, on=[COL_PATIENT, "segment"], how="left")
    except Exception as e:
        print(f"[WARN] Could not merge best-model slopes into OLS vs mixed comparison table: {e}")

    # Apply non-negative floor
    combined_slopes = apply_nonnegative_slope_floor(combined_slopes, floor=0.0)

    # Classify progression using floored OLS slopes
    classified_slopes = classify_progression(combined_slopes)

    # Save outputs
    Path(".").mkdir(exist_ok=True, parents=True)

    df_long.to_csv(OUTPUT_LONG_CSV, index=False)
    qc_summary.to_csv(OUTPUT_QC_SUMMARY_CSV, index=False)
    mixed_slopes_df.to_csv(OUTPUT_SLOPE_CSV, index=False)
    combined_slopes.to_csv(OUTPUT_SLOPE_COMPARISON_CSV, index=False)
    best_model_series.to_csv(OUTPUT_BEST_MODEL_SERIES_CSV, index=False)
    best_model_pooled.to_csv(OUTPUT_BEST_MODEL_POOLED_CSV, index=False)
    classified_slopes.to_csv(OUTPUT_PROGRESSORS_CSV, index=False)

    flagged = df_long[df_long["any_outlier_flag"]].copy()
    flagged.to_csv(OUTPUT_FLAGGED_CSV, index=False)

    print(f"[INFO] Saved long data with QC flags to: {OUTPUT_LONG_CSV}")
    print(f"[INFO] Saved QC summary to: {OUTPUT_QC_SUMMARY_CSV}")
    print(f"[INFO] Saved flagged measurements to: {OUTPUT_FLAGGED_CSV}")
    print(f"[INFO] Saved mixed-model slopes to: {OUTPUT_SLOPE_CSV}")
    print(f"[INFO] Saved OLS vs mixed slope comparison to: {OUTPUT_SLOPE_COMPARISON_CSV}")
    print(f"[INFO] Saved pooled best-model slopes to: {OUTPUT_BEST_MODEL_POOLED_CSV}")
    print(f"[INFO] Saved progression categories to: {OUTPUT_PROGRESSORS_CSV}")
        
    # Plots (UPDATED: Source categories in legend)
    generate_all_plots(df_long, save_dir=PLOTS_DIR)

    # Ranking (MERGED FROM ranker.py)
    try:
        generate_ranked_participant_list(RANK_INPUT_CSV, RANK_OUTPUT_CSV)
    except FileNotFoundError:
        print(f"[INFO] Ranking skipped: '{RANK_INPUT_CSV}' not found. The high-risk module should generate it; otherwise place the file next to the script or set RANK_INPUT_CSV.")
    except Exception as e:
        print(f"[WARN] Ranking step skipped/failed: {e}")


if __name__ == "__main__":
    main(
)