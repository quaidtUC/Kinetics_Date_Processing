#!/usr/bin/env python3
"""
MLR.py ― Reproducible multiple‑linear‑regression (MLR) analysis for Zn‑ligand
catalytic rates.

Features
--------
* Reads **CSV, TSV, or Excel** descriptor tables
* Converts *k* → ln *k* if only k is given
* Handles k = 0 by substituting a small ε (configurable) or dropping the rows
* Fits an **ordinary least‑squares** model
* Computes **leave‑one‑out** cross‑validation metrics
* Writes coefficients & diagnostics to `coeffs.json`
* Prints a worked example for k = 800 M⁻¹ s⁻¹

Usage
-----
python MLR.py -p "/path/to/descriptors.xlsx"
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

# ----------------------------------------------------------------------
# Descriptor columns expected in the data table
DESCRIPTORS = ["sigmaE2", "delta_q_Zn", "percent_Vbur"]

# ----------------------------------------------------------------------
# Aliases that map messy spreadsheet headers to the canonical names above
DESCRIPTOR_ALIASES = {
    "sigmaE2": ["sigma_e2", "sigma e2", "sigmae2"],
    "delta_q_Zn": ["zn_q", "deltaqzn", "delta_qzn", "delta q zn", "znq"],
    "percent_Vbur": ["vbur_aqua", "delta_vbur", "vbur", "%vbur", "percentvbur"],
}


def load_dataset(file_path: Path, epsilon: float):
    """
    Load descriptor data from CSV/TSV or Excel and return:

    X : np.ndarray  (n_samples × n_descriptors)
    y : np.ndarray  (n_samples,)  ln k values
    names : list[str]  descriptor column names
    """
    ext = file_path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            # Fallback for odd encodings
            df = pd.read_csv(file_path, encoding_errors="replace")

    # Strip whitespace in headers
    df.columns = [c.strip() for c in df.columns]

    # Identify ln k or k column
    lnk_aliases = ["lnk", "ln_k", "logk", "log_k"]
    k_aliases   = ["k", "rate", "k_ms", "k_m-1s-1"]

    lnk_col = next((c for c in lnk_aliases if c in df.columns), None)
    k_col   = next((c for c in k_aliases if c in df.columns), None)

    if lnk_col is None and k_col is None:
        raise KeyError(
            "Could not find a rate column. Expected one of "
            f"{lnk_aliases + k_aliases}. Headers present: {list(df.columns)}"
        )

    if lnk_col is None:
        bad_mask = df[k_col] <= 0
        n_bad = int(bad_mask.sum())
        if n_bad:
            if epsilon > 0:
                print(
                    f"⚠️  Replacing {n_bad} non‑positive k values with ε = {epsilon} "
                    f"before taking ln(k)."
                )
                df.loc[bad_mask, k_col] = epsilon
            else:
                print(
                    f"⚠️  Dropping {n_bad} rows with non‑positive k values "
                    f"(ε = 0)."
                )
                df = df.loc[~bad_mask].copy()
        df["lnk"] = np.log(df[k_col].astype(float))
        lnk_col = "lnk"

    # ------------------------------------------------------------------
    # 4) Bring spreadsheet headers to canonical descriptor names
    df_lower = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    mapped_cols = {}
    for canonical in DESCRIPTORS:
        if canonical in df.columns:
            mapped_cols[canonical] = canonical
            continue
        # Try aliases
        aliases = DESCRIPTOR_ALIASES.get(canonical, [])
        hit = next(
            (df_lower[a.lower().replace(" ", "").replace("_", "")]
             for a in aliases
             if a.lower().replace(" ", "").replace("_", "") in df_lower),
            None,
        )
        if hit:
            mapped_cols[canonical] = hit
        else:
            raise KeyError(
                f"Missing descriptor column for '{canonical}'. "
                f"Looked for { [canonical] + aliases }"
            )

    X = df[[mapped_cols[d] for d in DESCRIPTORS]].astype(float).values
    y = df[lnk_col].astype(float).values
    return X, y, DESCRIPTORS


def fit_mlr(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Fit OLS and return the model."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def loo_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Generate leave‑one‑out predictions."""
    preds = np.zeros_like(y)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    return preds


def main(file_path: Path, epsilon: float):
    print(f"Using data file: {file_path}")

    X, y, names = load_dataset(file_path, epsilon)
    model = fit_mlr(X, y)
    y_hat = model.predict(X)

    # Training metrics
    r2 = r2_score(y, y_hat)
    try:
        rmse = mean_squared_error(y, y_hat, squared=False)
    except TypeError:
        # Older scikit‑learn versions (<0.22) don’t support 'squared'
        rmse = np.sqrt(mean_squared_error(y, y_hat))

    # LOO‑CV metrics
    loo_preds = loo_predictions(X, y)
    r2_cv = r2_score(y, loo_preds)
    try:
        rmse_cv = mean_squared_error(y, loo_preds, squared=False)
    except TypeError:
        rmse_cv = np.sqrt(mean_squared_error(y, loo_preds))

    # Pretty print model
    equation = (
        f"ln k = {model.intercept_: .3f}"
        + "".join(
            f" + {coef: .3f}·{name}"
            for coef, name in zip(model.coef_, names)
        )
    )
    print("\nFitted model:")
    print("  " + equation)

    print(f"\nMetrics on training set : R² = {r2: .3f}, RMSE = {rmse: .3f}")
    print(f"Leave‑one‑out CV        : R²_CV = {r2_cv: .3f}, RMSE_CV = {rmse_cv: .3f}")

    # Save coefficients for provenance
    coeffs = {
        "intercept": float(model.intercept_),
        **{name: float(c) for name, c in zip(names, model.coef_)},
        "R2_train": float(r2),
        "RMSE_train": float(rmse),
        "R2_CV": float(r2_cv),
        "RMSE_CV": float(rmse_cv),
    }
    with open("coeffs.json", "w") as fh:
        json.dump(coeffs, fh, indent=2)
    print("\nCoefficients written to coeffs.json")

    # Example conversion
    ln_k_800 = np.log(800.0)
    print(f"\nExample: k = 800  →  ln k = {ln_k_800: .3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zn‑ligand MLR analysis")
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="Path to descriptor table (.csv, .tsv, .xlsx)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Small positive value to substitute for k ≤ 0 before taking ln(k). "
             "Set to 0 to drop those rows instead (default: 1e-6).",
    )
    args = parser.parse_args()
    main(args.path, args.epsilon)