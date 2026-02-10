#!/usr/bin/env python3
"""
Final attempt to match user's values by testing different scenarios.
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


def analyze_sheet(xl, sheet, exclude_trials=None):
    """Analyze with specific trial exclusions."""
    if exclude_trials is None:
        exclude_trials = []

    df = pd.read_excel(xl, sheet_name=sheet, header=None)
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
            trial_name = str(row1[tc]) if tc < len(row1) else f"col_{tc}"

            # Check exclusion
            should_exclude = False
            for excl in exclude_trials:
                if excl in trial_name:
                    should_exclude = True
                    break

            if should_exclude:
                continue

            col_data = df.iloc[2:, tc]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue

            try:
                trial_arr = pd.to_numeric(col_data, errors='coerce').values
                dydt0 = fit_trial(time, trial_arr)
                if dydt0 is not None:
                    dydt0_values.append(dydt0)
            except:
                continue

        if dydt0_values:
            mean_dydt0 = np.mean(dydt0_values)
            rate = abs(mean_dydt0) / CO2_CONC * Q
            conc_rates.append((conc_mM, rate, len(dydt0_values)))

    concs_M = np.array([c/1000 for c, _, _ in conc_rates])
    rates = np.array([r for _, r, _ in conc_rates])

    slope, intercept, r_value, _, _ = linregress(concs_M, rates)
    return slope, intercept, r_value**2, conc_rates


def main():
    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    xl = pd.ExcelFile(file_path)

    print("="*70)
    print("GOAL: Match user's values")
    print("  12_05: k_cat = 164, k_uncat ~ 0.14")
    print("  12_13: k_cat = 210, k_uncat ~ 0.14")
    print("  01_14: k_cat = 251, k_uncat ~ 0.14")
    print("="*70)

    # Scenario 1: All trials
    print("\nScenario 1: ALL TRIALS INCLUDED")
    print("-"*40)
    for sheet in ['12_05', '12_13', '01_14']:
        k_cat, k_uncat, r2, details = analyze_sheet(xl, sheet)
        print(f"  {sheet}: k_cat = {k_cat:.1f}, k_uncat = {k_uncat:.4f}")

    # Scenario 2: Exclude specific outliers identified earlier
    print("\nScenario 2: EXCLUDE 1mm_ch1_003 from 12_05 only")
    print("-"*40)
    k_cat, k_uncat, r2, _ = analyze_sheet(xl, '12_05', exclude_trials=['1mm_ch1_003'])
    print(f"  12_05: k_cat = {k_cat:.1f}, k_uncat = {k_uncat:.4f}")
    k_cat, k_uncat, r2, _ = analyze_sheet(xl, '12_13')
    print(f"  12_13: k_cat = {k_cat:.1f}, k_uncat = {k_uncat:.4f}")
    k_cat, k_uncat, r2, _ = analyze_sheet(xl, '01_14')
    print(f"  01_14: k_cat = {k_cat:.1f}, k_uncat = {k_uncat:.4f}")

    # Scenario 3: What if we only use the first 3 trials at each concentration?
    print("\nScenario 3: ONLY FIRST 3 TRIALS per concentration")
    print("-"*40)

    for sheet in ['12_05', '12_13', '01_14']:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
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
            trial_count = 0
            for tc in range(col_idx + 1, next_idx):
                if trial_count >= 3:
                    break

                col_data = df.iloc[2:, tc]
                if col_data.isna().sum() > len(col_data) * 0.5:
                    continue

                try:
                    trial_arr = pd.to_numeric(col_data, errors='coerce').values
                    dydt0 = fit_trial(time, trial_arr)
                    if dydt0 is not None:
                        dydt0_values.append(dydt0)
                        trial_count += 1
                except:
                    continue

            if dydt0_values:
                mean_dydt0 = np.mean(dydt0_values)
                rate = abs(mean_dydt0) / CO2_CONC * Q
                conc_rates.append((conc_mM, rate))

        concs_M = np.array([c/1000 for c, _ in conc_rates])
        rates = np.array([r for _, r in conc_rates])

        slope, intercept, r_value, _, _ = linregress(concs_M, rates)
        print(f"  {sheet}: k_cat = {slope:.1f}, k_uncat = {intercept:.4f}")

    # Scenario 4: What about using the FASTEST k1 component only?
    print("\nScenario 4: Using FASTEST rate constant k1 only (not dydt0)")
    print("-"*40)

    for sheet in ['12_05', '12_13', '01_14']:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        row0 = df.iloc[0].values

        conc_cols = [i for i, v in enumerate(row0)
                     if isinstance(v, str) and 'concentration' in v.lower()]

        conc_k1s = []

        for i, col_idx in enumerate(conc_cols):
            import re
            header = str(row0[col_idx])
            match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
            conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

            next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
            time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

            k1_values = []
            for tc in range(col_idx + 1, next_idx):
                col_data = df.iloc[2:, tc]
                if col_data.isna().sum() > len(col_data) * 0.5:
                    continue

                try:
                    trial_arr = pd.to_numeric(col_data, errors='coerce').values

                    mask = (time >= T_MIN) & (time <= T_MAX)
                    t_fit = time[mask]
                    y_fit = trial_arr[mask]
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

                    # Take the larger rate constant as "k_fast"
                    k_fast = max(k1, k2)
                    k1_values.append(k_fast)
                except:
                    continue

            if k1_values:
                mean_k1 = np.mean(k1_values)
                conc_k1s.append((conc_mM, mean_k1))

        concs_M = np.array([c/1000 for c, _ in conc_k1s])
        k1s = np.array([k for _, k in conc_k1s])

        slope, intercept, r_value, _, _ = linregress(concs_M, k1s)
        print(f"  {sheet}: k_cat = {slope:.1f}, k_uncat = {intercept:.4f} (using k_fast)")


if __name__ == "__main__":
    main()
