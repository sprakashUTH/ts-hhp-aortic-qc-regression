#!/usr/bin/env python3
"""
Regression analyses for HHP aortic growth manuscript.

Inputs:
    data/regression_analysis_dataset.csv
    data/mixed_model_long_dataset.csv
Outputs:
    outputs/regression_results/*.csv

This script uses only deidentified data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "regression_results"
OUT.mkdir(parents=True, exist_ok=True)

def ols_model_summary(df, y, xvars, model_name):
    d = df[[y] + xvars].dropna()
    if len(d) < len(xvars) + 3:
        return pd.DataFrame([{"model": model_name, "term": "INSUFFICIENT_DATA", "n": len(d)}])
    X = sm.add_constant(d[xvars])
    m = sm.OLS(d[y], X).fit()
    rows = []
    ci = m.conf_int()
    for param in m.params.index:
        rows.append({
            "model": model_name,
            "term": param,
            "estimate": m.params[param],
            "std_error": m.bse[param],
            "ci_lower": ci.loc[param, 0],
            "ci_upper": ci.loc[param, 1],
            "p_value": m.pvalues[param],
            "n": int(m.nobs),
            "r_squared": m.rsquared,
            "aic": m.aic,
            "bic": m.bic,
        })
    return pd.DataFrame(rows)

def fit_tobit_left(df, y, xvars, model_name, left=0.0):
    d = df[[y] + xvars].dropna().copy()
    if len(d) < len(xvars) + 5:
        return pd.DataFrame([{"model": model_name, "term": "INSUFFICIENT_DATA", "n": len(d)}])
    X = sm.add_constant(d[xvars]).astype(float).values
    yy = d[y].astype(float).values
    cens = yy <= left + 1e-12
    ols = sm.OLS(yy, X).fit()
    init = np.r_[np.asarray(ols.params), np.log(max(np.std(ols.resid), 1e-3))]

    def negll(theta):
        beta = theta[:-1]
        sigma = np.exp(theta[-1])
        mu = X.dot(beta)
        z = (left - mu) / sigma
        ll = np.empty_like(yy, dtype=float)
        ll[cens] = norm.logcdf(z[cens])
        ll[~cens] = norm.logpdf((yy[~cens] - mu[~cens]) / sigma) - np.log(sigma)
        return -np.sum(ll) if np.all(np.isfinite(ll)) else 1e100

    res = minimize(negll, init, method="BFGS")
    if not res.success:
        res = minimize(negll, init, method="Nelder-Mead", options={"maxiter": 10000})

    theta = res.x
    beta = theta[:-1]
    sigma = float(np.exp(theta[-1]))
    if hasattr(res, "hess_inv") and np.ndim(res.hess_inv) == 2:
        se = np.sqrt(np.diag(np.array(res.hess_inv)))[:len(beta)]
    else:
        se = np.repeat(np.nan, len(beta))

    terms = ["const"] + xvars
    ll = -negll(theta)
    k = len(theta)
    rows = []
    for i, term in enumerate(terms):
        z = beta[i] / se[i] if np.isfinite(se[i]) and se[i] > 0 else np.nan
        rows.append({
            "model": model_name,
            "term": term,
            "estimate": beta[i],
            "std_error": se[i],
            "ci_lower": beta[i] - 1.96 * se[i] if np.isfinite(se[i]) else np.nan,
            "ci_upper": beta[i] + 1.96 * se[i] if np.isfinite(se[i]) else np.nan,
            "p_value": 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan,
            "n": len(yy),
            "n_censored": int(cens.sum()),
            "sigma": sigma,
            "logLik": ll,
            "aic": 2 * k - 2 * ll,
            "bic": np.log(len(yy)) * k - 2 * ll,
            "converged": bool(res.success),
        })
    return pd.DataFrame(rows)

def main():
    reg = pd.read_csv(DATA / "regression_analysis_dataset.csv")
    long = pd.read_csv(DATA / "mixed_model_long_dataset.csv")
    reg["progression_ge_1mm"] = (reg["best_slope_floored_mm_per_year"] >= 1.0).astype(float)
    reg.loc[reg["best_slope_floored_mm_per_year"].isna(), "progression_ge_1mm"] = np.nan

    linear = []
    tobit = []
    logistic = []

    for seg in ["ascending", "sinus"]:
        d = reg[reg["segment"] == seg]
        linear.append(ols_model_summary(d, "best_slope_floored_mm_per_year", ["baseline_diameter_per_5mm"], f"{seg}_diameter_growth_linear"))
        linear.append(ols_model_summary(d, "best_slope_floored_mm_per_year", ["baseline_diameter_per_5mm", "age_first_hhp_years", "height_baseline_cm"], f"{seg}_diameter_growth_linear_adj_age_height"))

        tobit.append(fit_tobit_left(d, "best_slope_floored_mm_per_year", ["baseline_diameter_per_5mm"], f"{seg}_diameter_growth_tobit"))
        tobit.append(fit_tobit_left(d, "best_slope_floored_mm_per_year", ["baseline_diameter_per_5mm", "age_first_hhp_years", "height_baseline_cm"], f"{seg}_diameter_growth_tobit_adj_age_height"))

        for xvars, suffix in [
            (["coarctation_ever"], "coarctation"),
            (["bav_ever", "hypertension_ever", "coarctation_ever", "karyotype_45x_ever", "prior_surgery_hhp_ever", "baseline_diameter_per_5mm", "age_first_hhp_years"], "clinical_multivariable"),
        ]:
            dd = d[["progression_ge_1mm"] + xvars].dropna()
            model_name = f"{seg}_progression_ge_1mm_{suffix}"
            if len(dd) < len(xvars) + 10 or dd["progression_ge_1mm"].nunique() < 2:
                logistic.append(pd.DataFrame([{"model": model_name, "term": "INSUFFICIENT_DATA", "n": len(dd)}]))
                continue
            X = sm.add_constant(dd[xvars])
            try:
                m = sm.Logit(dd["progression_ge_1mm"], X).fit(disp=False, maxiter=100)
                ci = m.conf_int()
                rows = []
                for term in m.params.index:
                    rows.append({
                        "model": model_name,
                        "term": term,
                        "log_odds": m.params[term],
                        "odds_ratio": np.exp(m.params[term]),
                        "std_error": m.bse[term],
                        "ci_lower_or": np.exp(ci.loc[term, 0]),
                        "ci_upper_or": np.exp(ci.loc[term, 1]),
                        "p_value": m.pvalues[term],
                        "n": int(m.nobs),
                        "aic": m.aic,
                        "bic": m.bic,
                    })
                logistic.append(pd.DataFrame(rows))
            except Exception as e:
                logistic.append(pd.DataFrame([{"model": model_name, "term": "MODEL_FAILED", "note": str(e), "n": len(dd)}]))

    pd.concat(linear, ignore_index=True).to_csv(OUT / "linear_models_summary.csv", index=False)
    pd.concat(tobit, ignore_index=True).to_csv(OUT / "tobit_models_summary.csv", index=False)
    pd.concat(logistic, ignore_index=True).to_csv(OUT / "logistic_models_summary.csv", index=False)

    # Mixed-effects interaction models
    baseline = reg[["patient_id", "segment", "baseline_diameter_cm", "baseline_diameter_per_5mm", "height_baseline_cm"]].drop_duplicates()
    ml = long.merge(baseline, on=["patient_id", "segment"], how="left")
    mixed_rows = []
    for seg in ["ascending", "sinus"]:
        d = ml[ml["segment"] == seg].dropna(subset=["diameter_cm", "time_years", "baseline_diameter_per_5mm", "height_baseline_cm", "patient_id"])
        model_name = f"{seg}_time_by_baseline_diameter_mixedlm"
        try:
            md = smf.mixedlm("diameter_cm ~ time_years * baseline_diameter_per_5mm + height_baseline_cm", d, groups=d["patient_id"], re_formula="~time_years")
            mdf = md.fit(method="lbfgs", maxiter=200, reml=True)
            ci = mdf.conf_int()
            for term in mdf.params.index:
                mixed_rows.append({
                    "model": model_name,
                    "term": term,
                    "estimate": mdf.params[term],
                    "std_error": mdf.bse[term] if term in mdf.bse.index else np.nan,
                    "ci_lower": ci.loc[term, 0] if term in ci.index else np.nan,
                    "ci_upper": ci.loc[term, 1] if term in ci.index else np.nan,
                    "p_value": mdf.pvalues[term] if term in mdf.pvalues.index else np.nan,
                    "n_obs": int(mdf.nobs),
                    "n_patients": int(d["patient_id"].nunique()),
                    "converged": bool(mdf.converged),
                    "aic": mdf.aic,
                    "bic": mdf.bic,
                })
        except Exception as e:
            mixed_rows.append({"model": model_name, "term": "MODEL_FAILED", "note": str(e), "n_obs": len(d), "n_patients": d["patient_id"].nunique()})
    pd.DataFrame(mixed_rows).to_csv(OUT / "mixed_model_interactions.csv", index=False)

if __name__ == "__main__":
    main()
