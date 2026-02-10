#!/usr/bin/env python3
"""
Debug: Look at individual trial dydt0 values to understand variance.
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
    """Fit double exponential to one trial."""
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

        # Also calc amplitude-weighted k_obs
        k_obs_weighted = abs(A1*k1 + A2*k2) / abs(A1 + A2)

        return {
            'dydt0': dydt0,
            'k_obs_weighted': k_obs_weighted,
            'A1': A1, 'k1': k1,
            'A2': A2, 'k2': k2,
            'popt': popt
        }
    except Exception as e:
        return None


def main():
    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    xl = pd.ExcelFile(file_path)

    print("Detailed trial-by-trial analysis")
    print("="*70)

    for sheet in ['12_05', '12_13', '01_14']:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)

        row0 = df.iloc[0].values
        row1 = df.iloc[1].values  # trial names

        conc_cols = [i for i, v in enumerate(row0)
                     if isinstance(v, str) and 'concentration' in v.lower()]

        print(f"\n{'='*60}")
        print(f"{sheet}")
        print(f"{'='*60}")

        all_conc_data = []

        for i, col_idx in enumerate(conc_cols):
            import re
            header = str(row0[col_idx])
            match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
            conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

            next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
            time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

            print(f"\n  {conc_mM} mM:")
            print(f"  {'Trial':<25} {'dydt0':>12} {'k_obs_wt':>10} {'rate':>10}")
            print(f"  {'-'*60}")

            trial_dydt0s = []
            trial_rates = []

            for tc in range(col_idx + 1, next_idx):
                col_data = df.iloc[2:, tc]
                trial_name = str(row1[tc]) if tc < len(row1) else f"Trial {tc - col_idx}"

                if col_data.isna().sum() > len(col_data) * 0.5:
                    continue

                try:
                    trial_data = pd.to_numeric(col_data, errors='coerce').values
                    result = fit_trial(time, trial_data)

                    if result:
                        rate = abs(result['dydt0']) / CO2_CONC * Q
                        print(f"  {trial_name:<25} {result['dydt0']:>12.6f} {result['k_obs_weighted']:>10.4f} {rate:>10.4f}")
                        trial_dydt0s.append(result['dydt0'])
                        trial_rates.append(rate)
                except Exception as e:
                    print(f"  {trial_name:<25} ERROR: {e}")

            if trial_dydt0s:
                mean_dydt0 = np.mean(trial_dydt0s)
                mean_rate = np.mean(trial_rates)
                std_rate = np.std(trial_rates, ddof=1) if len(trial_rates) > 1 else 0
                print(f"  {'-'*60}")
                print(f"  {'MEAN':<25} {mean_dydt0:>12.6f} {'':>10} {mean_rate:>10.4f} ± {std_rate:.4f}")

                all_conc_data.append({
                    'conc_mM': conc_mM,
                    'mean_rate': mean_rate,
                    'n_trials': len(trial_rates)
                })

        # Regression
        if len(all_conc_data) >= 2:
            concs_M = np.array([d['conc_mM']/1000 for d in all_conc_data])
            rates = np.array([d['mean_rate'] for d in all_conc_data])

            slope, intercept, r_value, _, _ = linregress(concs_M, rates)
            print(f"\n  REGRESSION: k_cat = {slope:.1f}, k_uncat = {intercept:.4f}, R² = {r_value**2:.4f}")


if __name__ == "__main__":
    main()
