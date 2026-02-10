#!/usr/bin/env python3
"""
Try to match the user's original k_cat values: 164, 210, 251 /M/s
with k_uncat ~ 0.14 /s

Method:
1. dydt0 from double exponential
2. rate = |dydt0| / [CO2] * |Q|
3. Linear regression of rate vs [catalyst]

Testing different parameters to find what matches.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib

# Known values
Q = 0.402  # absolute value
CO2_CONC = 0.0338  # M

def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_double_exp_various_windows(time, absorbance, windows):
    """Try different time windows."""
    results = {}
    for t_min, t_max in windows:
        mask = (time >= t_min) & (time <= t_max)
        t_fit = time[mask]
        y_fit = absorbance[mask]

        valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
        t_fit = t_fit[valid]
        y_fit = y_fit[valid]

        if len(t_fit) < 20:
            continue

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
            results[(t_min, t_max)] = dydt0
        except:
            pass
    return results


def main():
    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    xl = pd.ExcelFile(file_path)

    date_sheets = ['12_05', '12_13', '01_14']
    target_kcats = [164, 210, 251]
    target_kuncat = 0.14

    # Standard window
    T_MIN, T_MAX = 0.02, 40.0

    print("Target k_cat values: 164, 210, 251 /M/s")
    print("Target k_uncat: ~0.14 /s")
    print()

    # Process each date
    for sheet in date_sheets:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)

        row0 = df.iloc[0].values
        conc_cols = [i for i, v in enumerate(row0)
                     if isinstance(v, str) and 'concentration' in v.lower()]

        print(f"\n{'='*60}")
        print(f"{sheet}")
        print(f"{'='*60}")

        # Get mean dydt0 per concentration
        conc_dydt0 = {}

        for i, col_idx in enumerate(conc_cols):
            import re
            header = str(row0[col_idx])
            match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
            conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

            next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
            time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

            dydt0_trials = []
            for tc in range(col_idx + 1, next_idx):
                col_data = df.iloc[2:, tc]
                if col_data.isna().sum() > len(col_data) * 0.5:
                    continue
                try:
                    trial_data = pd.to_numeric(col_data, errors='coerce').values

                    # Fit with standard window
                    mask = (time >= T_MIN) & (time <= T_MAX)
                    t_fit = time[mask]
                    y_fit = trial_data[mask]
                    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
                    t_fit = t_fit[valid]
                    y_fit = y_fit[valid]

                    if len(t_fit) < 20:
                        continue

                    n_window = max(1, len(y_fit) // 20)
                    y_start = np.median(y_fit[:n_window])
                    y_end = np.median(y_fit[-n_window:])
                    A0_guess = y_end
                    total_amp = y_start - y_end

                    p0 = [A0_guess, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
                    bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                              [np.inf, np.inf, 100, np.inf, 100])

                    popt, _ = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)
                    A0, A1, k1, A2, k2 = popt
                    dydt0 = -A1 * k1 - A2 * k2
                    dydt0_trials.append(dydt0)
                except:
                    continue

            if dydt0_trials:
                conc_dydt0[conc_mM] = np.mean(dydt0_trials)
                print(f"  {conc_mM} mM: dydt0 = {conc_dydt0[conc_mM]:.6f} AU/s ({len(dydt0_trials)} trials)")

        # Now do regression with user's conversion
        concs_M = np.array([c/1000 for c in sorted(conc_dydt0.keys())])
        dydt0s = np.array([conc_dydt0[c] for c in sorted(conc_dydt0.keys())])

        # rate = |dydt0| / [CO2] * |Q|
        rates = np.abs(dydt0s) / CO2_CONC * Q

        print(f"\n  Converted rates (|dydt0|/{CO2_CONC}*{Q}):")
        for c, r in zip(sorted(conc_dydt0.keys()), rates):
            print(f"    {c} mM: {r:.4f}")

        slope, intercept, r_value, _, _ = linregress(concs_M, rates)
        print(f"\n  k_cat = {slope:.1f}, k_uncat = {intercept:.4f}, R² = {r_value**2:.4f}")

    # Now let's try the REVERSE - what parameters would give us the user's values?
    print("\n" + "="*60)
    print("REVERSE ENGINEERING: What parameters give user's values?")
    print("="*60)

    # If k_cat for 12_05 should be 164 instead of 131, the ratio is 164/131 = 1.25
    # This suggests either:
    # - Q should be 0.502 instead of 0.402 (25% higher)
    # - [CO2] should be 0.027 instead of 0.0338 (20% lower)
    # - dydt0 values are 25% higher in user's calculation

    print("\nPossible explanations for difference:")
    print(f"  My k_cats: ~131, ~210, ~228")
    print(f"  User k_cats: 164, 210, 251")
    print()
    print("  Option 1: Different Q value")
    print(f"    To get 164 from 131: Q = {0.402 * 164/131:.3f}")
    print()
    print("  Option 2: Different [CO2] value")
    print(f"    To get 164 from 131: [CO2] = {0.0338 * 131/164:.4f} M")


if __name__ == "__main__":
    main()
