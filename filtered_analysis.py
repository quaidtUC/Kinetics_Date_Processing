#!/usr/bin/env python3
"""
Filtered analysis - keeping only MIDDLE amplitude range datasets.
Calculate k_cat using Working Method and Amplitude-Weighted k_obs.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib
import re

# Constants
CO2_CONC = 0.0338  # M
Q = 0.402
T_MIN, T_MAX = 0.02, 40.0

# Amplitude filter: MIDDLE range only (0.048 - 0.057)
# Slightly expanded lower bound to include 01_14 nBu/Hexyl
AMP_MIN = 0.048
AMP_MAX = 0.057


def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_trial(time, absorbance):
    """Fit double exp and return working method rate and weighted k_obs."""
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

    if total_amp < 0.01:
        return None

    try:
        p0 = [y_end, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
        bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                  [np.inf, np.inf, 100, np.inf, 100])
        popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)

        A0, A1, k1, A2, k2 = popt

        # Calculate R²
        y_pred = double_exp(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot

        if r2 < 0.99:
            return None

        # Working Method: rate from dydt0
        dydt0 = -A1 * k1 - A2 * k2
        rate_working = abs(dydt0) / CO2_CONC * Q

        # Amplitude-weighted k_obs
        k_weighted = abs(A1 * k1 + A2 * k2) / abs(A1 + A2)

        return {
            'amplitude': total_amp,
            'rate_working': rate_working,
            'k_weighted': k_weighted,
            'r2': r2
        }
    except:
        return None


def analyze_file(filepath, file_label):
    """Analyze a file and return results by concentration."""
    results = []

    try:
        xl = pd.ExcelFile(filepath)

        for sheet in xl.sheet_names:
            if sheet.lower() in ['results', 'summary', 'q', 'uncatalyzed']:
                continue

            df = pd.read_excel(xl, sheet_name=sheet, header=None)
            row0 = df.iloc[0].values

            has_conc = any('concentration' in str(v).lower() for v in row0 if isinstance(v, str))

            if has_conc:
                # Multi-concentration format (like Benz_Cyclen)
                conc_cols = [i for i, v in enumerate(row0)
                             if isinstance(v, str) and 'concentration' in str(v).lower()]

                for i, col_idx in enumerate(conc_cols):
                    # Parse concentration
                    header = str(row0[col_idx])
                    match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
                    conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

                    next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
                    time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

                    for tc in range(col_idx + 1, next_idx):
                        col_data = df.iloc[2:, tc]
                        if col_data.isna().sum() > len(col_data) * 0.5:
                            continue

                        trial_data = pd.to_numeric(col_data, errors='coerce').values
                        result = fit_trial(time, trial_data)

                        if result:
                            result['conc_mM'] = conc_mM
                            result['sheet'] = sheet
                            result['file'] = file_label
                            results.append(result)

            else:
                # Simple format with sheet name as concentration
                match = re.search(r'(\d+)p?(\d*)mm', sheet.lower())
                if match:
                    if match.group(2):
                        conc_mM = float(f"{match.group(1)}.{match.group(2)}")
                    else:
                        conc_mM = float(match.group(1))
                else:
                    continue

                time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values

                for col in range(1, df.shape[1]):
                    col_data = df.iloc[1:, col]
                    if col_data.isna().sum() > len(col_data) * 0.5:
                        continue

                    header = df.iloc[0, col] if col < len(df.iloc[0]) else None
                    if isinstance(header, str) and 'Q' in str(header).upper():
                        continue

                    trial_data = pd.to_numeric(col_data, errors='coerce').values
                    result = fit_trial(time, trial_data)

                    if result:
                        result['conc_mM'] = conc_mM
                        result['sheet'] = sheet
                        result['file'] = file_label
                        results.append(result)

    except Exception as e:
        print(f"Error with {filepath}: {e}")

    return results


def calc_kcat(results, method='rate_working'):
    """Calculate k_cat from concentration-rate data."""
    # Group by concentration
    conc_data = {}
    for r in results:
        conc = r['conc_mM']
        if conc not in conc_data:
            conc_data[conc] = []
        conc_data[conc].append(r[method])

    if len(conc_data) < 2:
        return None

    concs_M = np.array([c / 1000 for c in sorted(conc_data.keys())])
    means = np.array([np.mean(conc_data[c]) for c in sorted(conc_data.keys())])
    stds = np.array([np.std(conc_data[c], ddof=1) if len(conc_data[c]) > 1 else 0
                     for c in sorted(conc_data.keys())])

    slope, intercept, r_value, p_value, std_err = linregress(concs_M, means)

    return {
        'k_cat': slope,
        'k_cat_se': std_err,
        'k_uncat': intercept,
        'r2': r_value**2,
        'conc_data': [(c, np.mean(conc_data[c]), np.std(conc_data[c], ddof=1) if len(conc_data[c]) > 1 else 0)
                      for c in sorted(conc_data.keys())]
    }


def main():
    print("=" * 80)
    print("FILTERED ANALYSIS - MIDDLE AMPLITUDE RANGE ONLY")
    print(f"Amplitude filter: {AMP_MIN:.3f} - {AMP_MAX:.3f}")
    print("=" * 80)

    # Define datasets to analyze
    datasets = {
        # Benz_Cyclen - only 12_05 (12_13 is outlier, 01_14 is high)
        'Benz_Cyclen/12_05': ('Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx', '12_05'),

        # nBu_Cyclen
        'nBu_Cyclen/01_14': ('Data/Cyclen_Study/nBu_Cyclen/Butyl_01_14.xlsx', None),
        'nBu_Cyclen/02_02': ('Data/Cyclen_Study/nBu_Cyclen/nBu_cyc_02_02.xlsx', None),

        # Hexyl_Cyclen
        'Hexyl_Cyclen/01_14': ('Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_Cyclen_01_14.xlsx', None),
        'Hexyl_Cyclen/02_03': ('Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_cyc_02_03.xlsx', None),

        # Cyclen_baseline
        'Cyclen_baseline/01_13': ('Data/Cyclen_Study/Cyclen_baseline/Cyclen_01_13.xlsx', None),
        'Cyclen_baseline/5xDye': ('Data/Cyclen_Study/Cyclen_baseline/Cyclen_5xDye.xlsx', None),
        'Cyclen_baseline/original': ('Data/Cyclen_Study/Cyclen_baseline/Cyclen_original.xlsx', None),
    }

    all_results = {}

    for label, (filepath, sheet_filter) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {label}")
        print(f"{'='*60}")

        if sheet_filter:
            # For Benz_Cyclen, read specific sheet
            xl = pd.ExcelFile(filepath)
            df = pd.read_excel(xl, sheet_name=sheet_filter, header=None)
            row0 = df.iloc[0].values

            conc_cols = [i for i, v in enumerate(row0)
                         if isinstance(v, str) and 'concentration' in v.lower()]

            results = []
            for i, col_idx in enumerate(conc_cols):
                header = str(row0[col_idx])
                match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
                conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

                next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
                time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

                for tc in range(col_idx + 1, next_idx):
                    col_data = df.iloc[2:, tc]
                    if col_data.isna().sum() > len(col_data) * 0.5:
                        continue

                    trial_data = pd.to_numeric(col_data, errors='coerce').values
                    result = fit_trial(time, trial_data)

                    if result:
                        result['conc_mM'] = conc_mM
                        result['sheet'] = sheet_filter
                        result['file'] = label
                        results.append(result)
        else:
            results = analyze_file(filepath, label)

        if not results:
            print(f"  No valid results")
            continue

        # Check amplitudes
        amps = [r['amplitude'] for r in results]
        mean_amp = np.mean(amps)
        print(f"  Mean amplitude: {mean_amp:.4f} (range: {min(amps):.4f} - {max(amps):.4f})")

        # Filter by amplitude
        filtered = [r for r in results if AMP_MIN <= r['amplitude'] <= AMP_MAX]
        print(f"  Trials: {len(results)} total, {len(filtered)} in amplitude range")

        if len(filtered) < 3:
            print(f"  ⚠ Too few trials after filtering")
            # Try with unfiltered if amplitude is close
            if 0.045 <= mean_amp <= 0.065:
                print(f"  Using unfiltered (amplitude is borderline)")
                filtered = results
            else:
                continue

        # Calculate k_cat for both methods
        working_result = calc_kcat(filtered, 'rate_working')
        weighted_result = calc_kcat(filtered, 'k_weighted')

        if working_result:
            print(f"\n  Working Method (dydt0 → rate):")
            print(f"    k_cat = {working_result['k_cat']:.1f} ± {working_result['k_cat_se']:.1f} /M/s")
            print(f"    k_uncat = {working_result['k_uncat']:.4f} /s")
            print(f"    R² = {working_result['r2']:.4f}")
            print(f"    Conc data:")
            for conc, mean, std in working_result['conc_data']:
                print(f"      {conc}mM: {mean:.4f} ± {std:.4f}")

        if weighted_result:
            print(f"\n  Amplitude-Weighted k_obs:")
            print(f"    k_cat = {weighted_result['k_cat']:.1f} ± {weighted_result['k_cat_se']:.1f} /M/s")
            print(f"    k_uncat = {weighted_result['k_uncat']:.4f} /s")
            print(f"    R² = {weighted_result['r2']:.4f}")

        all_results[label] = {
            'mean_amp': mean_amp,
            'n_trials': len(filtered),
            'working': working_result,
            'weighted': weighted_result
        }

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - WORKING METHOD")
    print("=" * 80)
    print(f"\n{'Dataset':<25} {'Amplitude':<10} {'k_cat (/M/s)':<20} {'k_uncat (/s)':<12} {'R²':<8}")
    print("-" * 80)

    for label, data in all_results.items():
        if data['working']:
            w = data['working']
            kcat_str = f"{w['k_cat']:.1f} ± {w['k_cat_se']:.1f}"
            print(f"{label:<25} {data['mean_amp']:<10.4f} {kcat_str:<20} {w['k_uncat']:<12.4f} {w['r2']:<8.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE - AMPLITUDE-WEIGHTED k_obs")
    print("=" * 80)
    print(f"\n{'Dataset':<25} {'Amplitude':<10} {'k_cat (/M/s)':<20} {'k_uncat (/s)':<12} {'R²':<8}")
    print("-" * 80)

    for label, data in all_results.items():
        if data['weighted']:
            w = data['weighted']
            kcat_str = f"{w['k_cat']:.1f} ± {w['k_cat_se']:.1f}"
            print(f"{label:<25} {data['mean_amp']:<10.4f} {kcat_str:<20} {w['k_uncat']:<12.4f} {w['r2']:<8.4f}")

    # Group by catalyst type
    print("\n" + "=" * 80)
    print("GROUPED BY CATALYST - WORKING METHOD")
    print("=" * 80)

    catalyst_groups = {
        'Benz_Cyclen': [],
        'nBu_Cyclen': [],
        'Hexyl_Cyclen': [],
        'Cyclen_baseline': []
    }

    for label, data in all_results.items():
        if data['working']:
            for cat in catalyst_groups:
                if cat in label:
                    catalyst_groups[cat].append(data['working']['k_cat'])

    print(f"\n{'Catalyst':<20} {'k_cat values':<30} {'Mean ± Std':<20} {'CV (%)':<10}")
    print("-" * 80)

    for cat, kcats in catalyst_groups.items():
        if kcats:
            mean_k = np.mean(kcats)
            std_k = np.std(kcats, ddof=1) if len(kcats) > 1 else 0
            cv = 100 * std_k / mean_k if mean_k > 0 and len(kcats) > 1 else 0
            kcat_str = ', '.join([f'{k:.0f}' for k in kcats])
            mean_str = f"{mean_k:.0f} ± {std_k:.0f}" if len(kcats) > 1 else f"{mean_k:.0f}"
            print(f"{cat:<20} {kcat_str:<30} {mean_str:<20} {cv:<10.1f}")


if __name__ == "__main__":
    main()
