#!/usr/bin/env python3
"""
Filtered analysis v2 - keeping MIDDLE amplitude range datasets.
Discard only clear outliers (>0.065), keep borderline high (0.057-0.065).
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

# Amplitude filter: discard only clear outliers (>0.065)
AMP_MAX = 0.065


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


def analyze_multiconc_sheet(xl, sheet_name, file_label):
    """Analyze a sheet with multiple concentration columns."""
    results = []
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    row0 = df.iloc[0].values

    conc_cols = [i for i, v in enumerate(row0)
                 if isinstance(v, str) and 'concentration' in v.lower()]

    if not conc_cols:
        return results

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
                result['sheet'] = sheet_name
                result['file'] = file_label
                results.append(result)

    return results


def analyze_simple_file(filepath, file_label):
    """Analyze a file with sheet names as concentrations."""
    results = []

    try:
        xl = pd.ExcelFile(filepath)

        for sheet in xl.sheet_names:
            if sheet.lower() in ['results', 'summary', 'q', 'uncatalyzed', 'cyclen', 'old', 'co2 uncat']:
                continue

            # Parse concentration from sheet name
            match = re.search(r'(\d+)p?(\d*)mm', sheet.lower())
            if not match:
                continue

            if match.group(2):
                conc_mM = float(f"{match.group(1)}.{match.group(2)}")
            else:
                conc_mM = float(match.group(1))

            df = pd.read_excel(xl, sheet_name=sheet, header=None)
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

    slope, intercept, r_value, p_value, std_err = linregress(concs_M, means)

    return {
        'k_cat': slope,
        'k_cat_se': std_err,
        'k_uncat': intercept,
        'r2': r_value**2,
        'n_conc': len(conc_data),
        'conc_data': [(c, np.mean(conc_data[c]),
                       np.std(conc_data[c], ddof=1) if len(conc_data[c]) > 1 else 0,
                       len(conc_data[c]))
                      for c in sorted(conc_data.keys())]
    }


def main():
    print("=" * 85)
    print("FILTERED ANALYSIS - EXCLUDING AMPLITUDE OUTLIERS (>0.065)")
    print("=" * 85)

    all_results = {}

    # ============ BENZ_CYCLEN ============
    print("\n" + "=" * 70)
    print("BENZ_CYCLEN (excluding 12_13 outlier)")
    print("=" * 70)

    benz_file = 'Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx'
    xl = pd.ExcelFile(benz_file)

    for sheet in ['12_05', '01_14']:  # Skip 12_13
        results = analyze_multiconc_sheet(xl, sheet, f'Benz/{sheet}')
        if results:
            amps = [r['amplitude'] for r in results]
            print(f"\n  {sheet}: amp = {np.mean(amps):.4f} ± {np.std(amps):.4f}, n={len(results)}")

            working = calc_kcat(results, 'rate_working')
            weighted = calc_kcat(results, 'k_weighted')

            if working:
                print(f"    Working: k_cat = {working['k_cat']:.1f} ± {working['k_cat_se']:.1f}, k_uncat = {working['k_uncat']:.4f}, R² = {working['r2']:.4f}")
            if weighted:
                print(f"    Weighted: k_cat = {weighted['k_cat']:.1f} ± {weighted['k_cat_se']:.1f}, k_uncat = {weighted['k_uncat']:.4f}, R² = {weighted['r2']:.4f}")

            all_results[f'Benz_Cyclen/{sheet}'] = {
                'amp': np.mean(amps),
                'working': working,
                'weighted': weighted
            }

    # ============ nBu_CYCLEN ============
    print("\n" + "=" * 70)
    print("nBu_CYCLEN")
    print("=" * 70)

    nbu_files = [
        ('Data/Cyclen_Study/nBu_Cyclen/Butyl_01_14.xlsx', 'nBu/01_14'),
        ('Data/Cyclen_Study/nBu_Cyclen/nBu_cyc_02_02.xlsx', 'nBu/02_02'),
    ]

    for filepath, label in nbu_files:
        results = analyze_simple_file(filepath, label)
        if results:
            # Filter by amplitude
            filtered = [r for r in results if r['amplitude'] <= AMP_MAX]
            amps = [r['amplitude'] for r in filtered]

            print(f"\n  {label}: amp = {np.mean(amps):.4f} ± {np.std(amps):.4f}, n={len(filtered)}/{len(results)}")

            working = calc_kcat(filtered, 'rate_working')
            weighted = calc_kcat(filtered, 'k_weighted')

            if working:
                print(f"    Working: k_cat = {working['k_cat']:.1f} ± {working['k_cat_se']:.1f}, k_uncat = {working['k_uncat']:.4f}, R² = {working['r2']:.4f}")
            if weighted:
                print(f"    Weighted: k_cat = {weighted['k_cat']:.1f} ± {weighted['k_cat_se']:.1f}, k_uncat = {weighted['k_uncat']:.4f}, R² = {weighted['r2']:.4f}")

            all_results[label.replace('/', '_Cyclen/')] = {
                'amp': np.mean(amps),
                'working': working,
                'weighted': weighted
            }

    # ============ HEXYL_CYCLEN ============
    print("\n" + "=" * 70)
    print("HEXYL_CYCLEN")
    print("=" * 70)

    hexyl_files = [
        ('Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_Cyclen_01_14.xlsx', 'Hexyl/01_14'),
        ('Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_cyc_02_03.xlsx', 'Hexyl/02_03'),
    ]

    for filepath, label in hexyl_files:
        results = analyze_simple_file(filepath, label)
        if results:
            filtered = [r for r in results if r['amplitude'] <= AMP_MAX]
            amps = [r['amplitude'] for r in filtered]

            print(f"\n  {label}: amp = {np.mean(amps):.4f} ± {np.std(amps):.4f}, n={len(filtered)}/{len(results)}")

            working = calc_kcat(filtered, 'rate_working')
            weighted = calc_kcat(filtered, 'k_weighted')

            if working:
                print(f"    Working: k_cat = {working['k_cat']:.1f} ± {working['k_cat_se']:.1f}, k_uncat = {working['k_uncat']:.4f}, R² = {working['r2']:.4f}")
            if weighted:
                print(f"    Weighted: k_cat = {weighted['k_cat']:.1f} ± {weighted['k_cat_se']:.1f}, k_uncat = {weighted['k_uncat']:.4f}, R² = {weighted['r2']:.4f}")

            all_results[label.replace('/', '_Cyclen/')] = {
                'amp': np.mean(amps),
                'working': working,
                'weighted': weighted
            }

    # ============ CYCLEN_BASELINE ============
    print("\n" + "=" * 70)
    print("CYCLEN_BASELINE (unsubstituted cyclen)")
    print("=" * 70)

    cyclen_files = [
        ('Data/Cyclen_Study/Cyclen_baseline/Cyclen_01_13.xlsx', 'Cyclen/01_13'),
        ('Data/Cyclen_Study/Cyclen_baseline/Cyclen_5xDye.xlsx', 'Cyclen/5xDye'),
    ]

    for filepath, label in cyclen_files:
        results = analyze_simple_file(filepath, label)
        if results:
            filtered = [r for r in results if r['amplitude'] <= AMP_MAX]
            amps = [r['amplitude'] for r in filtered]

            print(f"\n  {label}: amp = {np.mean(amps):.4f} ± {np.std(amps):.4f}, n={len(filtered)}/{len(results)}")

            working = calc_kcat(filtered, 'rate_working')
            weighted = calc_kcat(filtered, 'k_weighted')

            if working:
                print(f"    Working: k_cat = {working['k_cat']:.1f} ± {working['k_cat_se']:.1f}, k_uncat = {working['k_uncat']:.4f}, R² = {working['r2']:.4f}")
                for conc, mean, std, n in working['conc_data']:
                    print(f"      {conc}mM: rate = {mean:.4f} ± {std:.4f} (n={n})")
            if weighted:
                print(f"    Weighted: k_cat = {weighted['k_cat']:.1f} ± {weighted['k_cat_se']:.1f}, k_uncat = {weighted['k_uncat']:.4f}, R² = {weighted['r2']:.4f}")

            all_results[label.replace('/', '_baseline/')] = {
                'amp': np.mean(amps),
                'working': working,
                'weighted': weighted
            }

    # ============ SUMMARY TABLES ============
    print("\n" + "=" * 85)
    print("SUMMARY TABLE - WORKING METHOD")
    print("=" * 85)
    print(f"\n{'Dataset':<25} {'Amp':<8} {'k_cat (/M/s)':<18} {'k_uncat (/s)':<12} {'R²':<8}")
    print("-" * 75)

    for label, data in sorted(all_results.items()):
        if data['working']:
            w = data['working']
            kcat_str = f"{w['k_cat']:.1f} ± {w['k_cat_se']:.1f}"
            print(f"{label:<25} {data['amp']:<8.4f} {kcat_str:<18} {w['k_uncat']:<12.4f} {w['r2']:<8.4f}")

    print("\n" + "=" * 85)
    print("SUMMARY TABLE - AMPLITUDE-WEIGHTED k_obs")
    print("=" * 85)
    print(f"\n{'Dataset':<25} {'Amp':<8} {'k_cat (/M/s)':<18} {'k_uncat (/s)':<12} {'R²':<8}")
    print("-" * 75)

    for label, data in sorted(all_results.items()):
        if data['weighted']:
            w = data['weighted']
            kcat_str = f"{w['k_cat']:.1f} ± {w['k_cat_se']:.1f}"
            print(f"{label:<25} {data['amp']:<8.4f} {kcat_str:<18} {w['k_uncat']:<12.4f} {w['r2']:<8.4f}")

    # ============ BY CATALYST TYPE ============
    print("\n" + "=" * 85)
    print("BY CATALYST TYPE - WORKING METHOD")
    print("=" * 85)

    groups = {
        'Benz_Cyclen': [],
        'nBu_Cyclen': [],
        'Hexyl_Cyclen': [],
        'Cyclen (unsubst)': []
    }

    for label, data in all_results.items():
        if not data['working']:
            continue
        if 'Benz' in label:
            groups['Benz_Cyclen'].append((label, data['working']['k_cat'], data['working']['k_cat_se']))
        elif 'nBu' in label:
            groups['nBu_Cyclen'].append((label, data['working']['k_cat'], data['working']['k_cat_se']))
        elif 'Hexyl' in label:
            groups['Hexyl_Cyclen'].append((label, data['working']['k_cat'], data['working']['k_cat_se']))
        elif 'Cyclen' in label:
            groups['Cyclen (unsubst)'].append((label, data['working']['k_cat'], data['working']['k_cat_se']))

    print(f"\n{'Catalyst':<18} {'Measurements':<40} {'Mean k_cat':<15} {'CV (%)':<8}")
    print("-" * 85)

    for cat, entries in groups.items():
        if entries:
            kcats = [k for _, k, _ in entries]
            labels = [l.split('/')[-1] for l, _, _ in entries]
            mean_k = np.mean(kcats)
            std_k = np.std(kcats, ddof=1) if len(kcats) > 1 else 0
            cv = 100 * std_k / mean_k if len(kcats) > 1 else 0

            meas_str = ', '.join([f"{l}:{k:.0f}" for l, k in zip(labels, kcats)])
            mean_str = f"{mean_k:.0f} ± {std_k:.0f}" if len(kcats) > 1 else f"{mean_k:.0f}"
            print(f"{cat:<18} {meas_str:<40} {mean_str:<15} {cv:<8.1f}")


if __name__ == "__main__":
    main()
