#!/usr/bin/env python3
"""
Find all potential outlier trials across all dates.
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


def main():
    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    xl = pd.ExcelFile(file_path)

    excluded_trials = {
        '12_05': [],
        '12_13': [],
        '01_14': []
    }

    print("Looking for outliers (|Z| > 1.5) in each concentration group...")
    print()

    for sheet in ['12_05', '12_13', '01_14']:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)

        row0 = df.iloc[0].values
        row1 = df.iloc[1].values

        conc_cols = [i for i, v in enumerate(row0)
                     if isinstance(v, str) and 'concentration' in v.lower()]

        print(f"\n{'='*60}")
        print(f"{sheet}")
        print(f"{'='*60}")

        for i, col_idx in enumerate(conc_cols):
            import re
            header = str(row0[col_idx])
            match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
            conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

            next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
            time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

            trial_data_list = []
            for tc in range(col_idx + 1, next_idx):
                trial_name = str(row1[tc]) if tc < len(row1) else f"col_{tc}"
                col_data = df.iloc[2:, tc]

                if col_data.isna().sum() > len(col_data) * 0.5:
                    continue

                try:
                    trial_arr = pd.to_numeric(col_data, errors='coerce').values
                    dydt0 = fit_trial(time, trial_arr)
                    if dydt0 is not None:
                        trial_data_list.append((trial_name, dydt0))
                except:
                    continue

            if len(trial_data_list) >= 3:
                dydt0_values = np.array([d[1] for d in trial_data_list])
                mean = np.mean(dydt0_values)
                std = np.std(dydt0_values, ddof=1)

                print(f"\n  {conc_mM} mM: mean dydt0 = {mean:.6f}, std = {std:.6f}")

                for trial_name, dydt0 in trial_data_list:
                    zscore = (dydt0 - mean) / std if std > 0 else 0
                    outlier_flag = " **OUTLIER**" if abs(zscore) > 1.5 else ""
                    print(f"    {trial_name}: {dydt0:.6f} (Z={zscore:+.2f}){outlier_flag}")

                    if abs(zscore) > 1.5:
                        excluded_trials[sheet].append((trial_name, conc_mM))

    print("\n" + "="*60)
    print("Summary of potential outliers to exclude:")
    print("="*60)
    for sheet, trials in excluded_trials.items():
        if trials:
            print(f"  {sheet}: {trials}")
        else:
            print(f"  {sheet}: None")

    # Now compute k_cat with outliers excluded
    print("\n" + "="*60)
    print("k_cat values with outliers excluded:")
    print("="*60)

    for sheet in ['12_05', '12_13', '01_14']:
        exclude_list = [t[0] for t in excluded_trials[sheet]]

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

                if trial_name in exclude_list:
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
                conc_rates.append((conc_mM, rate))

        concs_M = np.array([c/1000 for c, _ in conc_rates])
        rates = np.array([r for _, r in conc_rates])

        slope, intercept, r_value, _, _ = linregress(concs_M, rates)
        print(f"  {sheet}: k_cat = {slope:.1f}, k_uncat = {intercept:.4f}, R² = {r_value**2:.4f}")

    print("\nUser's target: 164, 210, 251 with k_uncat ~ 0.14")


if __name__ == "__main__":
    main()
