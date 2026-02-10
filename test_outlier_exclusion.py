#!/usr/bin/env python3
"""
Test if excluding outliers brings us closer to user's values.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib

Q = 0.402
CO2_CONC = 0.0338
T_MIN, T_MAX = 0.02, 40.0

def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_trial(time, absorbance, t_min=T_MIN, t_max=T_MAX):
    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 20:
        return None

    n_window = max(1, len(y_fit) // 20)
    y_start = np.median(y_fit[:n_window])
    y_end = np.median(y_fit[-n_window:])
    A0_guess = y_end
    total_amp = y_start - y_end

    try:
        p0 = [A0_guess, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
        bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                  [np.inf, np.inf, 100, np.inf, 100])

        popt, _ = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)
        A0, A1, k1, A2, k2 = popt
        dydt0 = -A1 * k1 - A2 * k2
        return dydt0
    except:
        return None


def analyze_with_outlier_handling(xl, sheet, exclude_outliers=True, zscore_thresh=2.0):
    """Analyze one date with optional outlier exclusion."""
    df = pd.read_excel(xl, sheet_name=sheet, header=None)

    row0 = df.iloc[0].values
    conc_cols = [i for i, v in enumerate(row0)
                 if isinstance(v, str) and 'concentration' in v.lower()]

    conc_rates = []

    for i, col_idx in enumerate(conc_cols):
        import re
        header = str(row0[col_idx])
        match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
        conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

        next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
        time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

        dydt0_values = []
        for tc in range(col_idx + 1, next_idx):
            col_data = df.iloc[2:, tc]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue
            try:
                trial_data = pd.to_numeric(col_data, errors='coerce').values
                dydt0 = fit_trial(time, trial_data)
                if dydt0 is not None:
                    dydt0_values.append(dydt0)
            except:
                continue

        if dydt0_values:
            dydt0_arr = np.array(dydt0_values)

            if exclude_outliers and len(dydt0_arr) > 2:
                # Z-score based outlier removal
                mean = np.mean(dydt0_arr)
                std = np.std(dydt0_arr, ddof=1)
                if std > 0:
                    zscores = np.abs((dydt0_arr - mean) / std)
                    mask = zscores < zscore_thresh
                    if mask.sum() >= 2:  # Keep at least 2 points
                        dydt0_arr = dydt0_arr[mask]

            mean_dydt0 = np.mean(dydt0_arr)
            rate = abs(mean_dydt0) / CO2_CONC * Q
            conc_rates.append((conc_mM, rate))

    # Regression
    concs_M = np.array([c/1000 for c, _ in conc_rates])
    rates = np.array([r for _, r in conc_rates])

    slope, intercept, r_value, _, _ = linregress(concs_M, rates)
    return slope, intercept, r_value**2


def main():
    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    xl = pd.ExcelFile(file_path)

    print("User's target values: 164, 210, 251 /M/s with k_uncat ~ 0.14")
    print()

    print("="*70)
    print("Method 1: All trials (no outlier exclusion)")
    print("="*70)
    for sheet in ['12_05', '12_13', '01_14']:
        k_cat, k_uncat, r2 = analyze_with_outlier_handling(xl, sheet, exclude_outliers=False)
        print(f"  {sheet}: k_cat = {k_cat:.1f}, k_uncat = {k_uncat:.4f}, R² = {r2:.4f}")

    print()
    print("="*70)
    print("Method 2: Exclude outliers (Z > 2)")
    print("="*70)
    for sheet in ['12_05', '12_13', '01_14']:
        k_cat, k_uncat, r2 = analyze_with_outlier_handling(xl, sheet, exclude_outliers=True, zscore_thresh=2.0)
        print(f"  {sheet}: k_cat = {k_cat:.1f}, k_uncat = {k_uncat:.4f}, R² = {r2:.4f}")

    print()
    print("="*70)
    print("Method 3: Exclude outliers (Z > 1.5)")
    print("="*70)
    for sheet in ['12_05', '12_13', '01_14']:
        k_cat, k_uncat, r2 = analyze_with_outlier_handling(xl, sheet, exclude_outliers=True, zscore_thresh=1.5)
        print(f"  {sheet}: k_cat = {k_cat:.1f}, k_uncat = {k_uncat:.4f}, R² = {r2:.4f}")

    # Let's manually exclude the 1mm_ch1_003 trial from 12_05
    print()
    print("="*70)
    print("Method 4: Manual exclusion of 1mm_ch1_003 from 12_05")
    print("="*70)

    df = pd.read_excel(xl, sheet_name='12_05', header=None)
    row0 = df.iloc[0].values
    row1 = df.iloc[1].values

    conc_cols = [i for i, v in enumerate(row0)
                 if isinstance(v, str) and 'concentration' in v.lower()]

    conc_rates = []
    for i, col_idx in enumerate(conc_cols):
        import re
        header = str(row0[col_idx])
        match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
        conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

        next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
        time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

        dydt0_values = []
        for tc in range(col_idx + 1, next_idx):
            trial_name = str(row1[tc]) if tc < len(row1) else ""

            # Skip the outlier trial
            if '003' in trial_name and conc_mM == 1.0:
                continue

            col_data = df.iloc[2:, tc]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue
            try:
                trial_data = pd.to_numeric(col_data, errors='coerce').values
                dydt0 = fit_trial(time, trial_data)
                if dydt0 is not None:
                    dydt0_values.append(dydt0)
            except:
                continue

        if dydt0_values:
            mean_dydt0 = np.mean(dydt0_values)
            rate = abs(mean_dydt0) / CO2_CONC * Q
            conc_rates.append((conc_mM, rate))
            print(f"    {conc_mM} mM: rate = {rate:.4f} (n={len(dydt0_values)})")

    concs_M = np.array([c/1000 for c, _ in conc_rates])
    rates = np.array([r for _, r in conc_rates])

    slope, intercept, r_value, _, _ = linregress(concs_M, rates)
    print(f"  12_05 (manual exclusion): k_cat = {slope:.1f}, k_uncat = {intercept:.4f}, R² = {r_value**2:.4f}")


if __name__ == "__main__":
    main()
