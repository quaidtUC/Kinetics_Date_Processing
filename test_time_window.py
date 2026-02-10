#!/usr/bin/env python3
"""
Test effect of including data prior to 0.02s (the mixing artifact region).
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib

CO2_CONC = 0.0338
Q = 0.402


def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_double_exp(time, absorbance, t_min, t_max):
    mask = (time >= t_min) & (time <= t_max)
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


def analyze_with_window(xl, t_min, t_max):
    """Analyze all sheets with given time window."""
    results = {}

    for sheet in ['12_05', '12_13', '01_14']:
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
                rate = fit_double_exp(time, trial_data, t_min, t_max)
                if rate is not None:
                    rates.append(rate)

            if rates:
                conc_rates.append((conc_mM, np.mean(rates)))

        if len(conc_rates) >= 2:
            concs_M = np.array([c / 1000 for c, _ in conc_rates])
            rates = np.array([r for _, r in conc_rates])
            slope, intercept, r_value, _, _ = linregress(concs_M, rates)
            results[sheet] = {'k_cat': slope, 'k_uncat': intercept, 'r2': r_value**2}

    return results


def main():
    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    xl = pd.ExcelFile(file_path)

    print("=" * 70)
    print("EFFECT OF TIME WINDOW ON RESULTS")
    print("=" * 70)

    # Test different t_min values
    t_min_values = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    t_max = 40.0

    print(f"\n{'t_min (s)':<12} {'12_05':<12} {'12_13':<12} {'01_14':<12} {'Mean':<12} {'CV (%)':<10}")
    print("-" * 70)

    for t_min in t_min_values:
        results = analyze_with_window(xl, t_min, t_max)

        k_cats = [results[s]['k_cat'] for s in ['12_05', '12_13', '01_14']]
        mean_kcat = np.mean(k_cats)
        std_kcat = np.std(k_cats, ddof=1)
        cv = 100 * std_kcat / mean_kcat

        print(f"{t_min:<12.3f} {k_cats[0]:<12.1f} {k_cats[1]:<12.1f} {k_cats[2]:<12.1f} {mean_kcat:<12.1f} {cv:<10.1f}")

    # Also show k_uncat for the key windows
    print("\n" + "=" * 70)
    print("k_uncat VALUES FOR KEY TIME WINDOWS")
    print("=" * 70)

    for t_min in [0.0, 0.02]:
        results = analyze_with_window(xl, t_min, t_max)
        print(f"\nt_min = {t_min} s:")
        for sheet in ['12_05', '12_13', '01_14']:
            r = results[sheet]
            print(f"  {sheet}: k_cat = {r['k_cat']:.1f}, k_uncat = {r['k_uncat']:.4f}")

        k_uncats = [results[s]['k_uncat'] for s in ['12_05', '12_13', '01_14']]
        print(f"  Mean k_uncat: {np.mean(k_uncats):.4f} /s (expected: 0.15 /s)")


if __name__ == "__main__":
    main()
