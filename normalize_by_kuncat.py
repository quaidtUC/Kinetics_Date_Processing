#!/usr/bin/env python3
"""
Test if normalizing by k_uncat improves consistency using the Working Method.

If day-to-day variation is due to a systematic experimental factor (like CO2 saturation
or temperature), then normalizing each day's k_cat by its k_uncat should reduce CV.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib

CO2_CONC = 0.0338
Q = 0.402
T_MIN, T_MAX = 0.02, 40.0
EXPECTED_KUNCAT = 0.15  # User's measured value


def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_double_exp(time, absorbance):
    mask = (time >= T_MIN) & (time <= T_MAX)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit, y_fit = t_fit[valid], y_fit[valid]

    if len(t_fit) < 50:
        return None

    n = max(1, len(y_fit) // 20)
    y_start, y_end = np.median(y_fit[:n]), np.median(y_fit[-n:])
    total_amp = y_start - y_end

    try:
        p0 = [y_end, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
        bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                  [np.inf, np.inf, 100, np.inf, 100])
        popt, _ = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)

        A0, A1, k1, A2, k2 = popt
        dydt0 = -A1 * k1 - A2 * k2
        rate = abs(dydt0) / CO2_CONC * Q

        return rate
    except:
        return None


def analyze_sheet(xl, sheet):
    """Get rates for each concentration."""
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

        rates = []
        for tc in range(col_idx + 1, next_idx):
            col_data = df.iloc[2:, tc]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue

            trial_data = pd.to_numeric(col_data, errors='coerce').values
            rate = fit_double_exp(time, trial_data)
            if rate is not None:
                rates.append(rate)

        if rates:
            conc_rates.append((conc_mM, np.mean(rates)))

    return conc_rates


def main():
    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    xl = pd.ExcelFile(file_path)

    print("=" * 70)
    print("NORMALIZING BY k_uncat - WORKING METHOD")
    print("=" * 70)

    results = {}

    for sheet in ['12_05', '12_13', '01_14']:
        conc_rates = analyze_sheet(xl, sheet)
        concs_M = np.array([c / 1000 for c, _ in conc_rates])
        rates = np.array([r for _, r in conc_rates])

        slope, intercept, r_value, _, _ = linregress(concs_M, rates)

        results[sheet] = {
            'k_cat': slope,
            'k_uncat': intercept,
            'r2': r_value**2,
            'conc_rates': conc_rates
        }

    # Raw results
    print("\n1. RAW RESULTS (no normalization)")
    print("-" * 50)
    k_cats_raw = []
    for sheet in ['12_05', '12_13', '01_14']:
        r = results[sheet]
        print(f"  {sheet}: k_cat = {r['k_cat']:.1f}, k_uncat = {r['k_uncat']:.4f}")
        k_cats_raw.append(r['k_cat'])

    mean_raw = np.mean(k_cats_raw)
    std_raw = np.std(k_cats_raw, ddof=1)
    cv_raw = 100 * std_raw / mean_raw
    print(f"\n  Mean k_cat: {mean_raw:.1f} ± {std_raw:.1f} /M/s")
    print(f"  CV: {cv_raw:.1f}%")

    # Method 1: Normalize k_cat by (expected_kuncat / measured_kuncat)
    print("\n2. NORMALIZED BY k_uncat RATIO")
    print(f"   k_cat_norm = k_cat × ({EXPECTED_KUNCAT} / k_uncat)")
    print("-" * 50)
    k_cats_norm1 = []
    for sheet in ['12_05', '12_13', '01_14']:
        r = results[sheet]
        norm_factor = EXPECTED_KUNCAT / r['k_uncat']
        k_cat_norm = r['k_cat'] * norm_factor
        print(f"  {sheet}: k_cat = {r['k_cat']:.1f} × {norm_factor:.3f} = {k_cat_norm:.1f}")
        k_cats_norm1.append(k_cat_norm)

    mean_norm1 = np.mean(k_cats_norm1)
    std_norm1 = np.std(k_cats_norm1, ddof=1)
    cv_norm1 = 100 * std_norm1 / mean_norm1
    print(f"\n  Mean k_cat: {mean_norm1:.1f} ± {std_norm1:.1f} /M/s")
    print(f"  CV: {cv_norm1:.1f}%")

    # Method 2: Force intercept to expected k_uncat and recalculate slope
    print("\n3. FORCE INTERCEPT TO 0.15, RECALCULATE SLOPE")
    print("   Using: k_cat = (rate - 0.15) / [catalyst]")
    print("-" * 50)
    k_cats_forced = []
    for sheet in ['12_05', '12_13', '01_14']:
        conc_rates = results[sheet]['conc_rates']
        # For each concentration, calculate implied k_cat if k_uncat = 0.15
        implied_kcats = []
        for conc_mM, rate in conc_rates:
            conc_M = conc_mM / 1000
            implied_kcat = (rate - EXPECTED_KUNCAT) / conc_M
            implied_kcats.append(implied_kcat)
        mean_implied = np.mean(implied_kcats)
        print(f"  {sheet}: k_cat = {mean_implied:.1f} /M/s")
        k_cats_forced.append(mean_implied)

    mean_forced = np.mean(k_cats_forced)
    std_forced = np.std(k_cats_forced, ddof=1)
    cv_forced = 100 * std_forced / mean_forced
    print(f"\n  Mean k_cat: {mean_forced:.1f} ± {std_forced:.1f} /M/s")
    print(f"  CV: {cv_forced:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<40} {'Mean k_cat':<15} {'CV (%)':<10}")
    print("-" * 65)
    print(f"{'Raw (no normalization)':<40} {mean_raw:<15.1f} {cv_raw:<10.1f}")
    print(f"{'Normalized by k_uncat ratio':<40} {mean_norm1:<15.1f} {cv_norm1:<10.1f}")
    print(f"{'Force intercept = 0.15':<40} {mean_forced:<15.1f} {cv_forced:<10.1f}")

    improvement1 = (cv_raw - cv_norm1) / cv_raw * 100
    improvement2 = (cv_raw - cv_forced) / cv_raw * 100
    print(f"\nImprovement with k_uncat ratio normalization: {improvement1:+.1f}%")
    print(f"Improvement with forced intercept: {improvement2:+.1f}%")


if __name__ == "__main__":
    main()
