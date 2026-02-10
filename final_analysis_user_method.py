#!/usr/bin/env python3
"""
Final analysis using USER'S EXACT METHOD:
- rate = |dydt0| * Q / CO2
- Weighted least squares with weights = 1/sigma²
- Exclude amplitude outliers (>0.065)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import pathlib
import re

# Constants
CO2 = 0.0338  # M
Q = 0.402
T_MIN, T_MAX = 0.02, 40.0
AMP_MAX = 0.065  # Exclude outliers above this


def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_trial(time, absorbance):
    """Fit double exp and return dydt0."""
    mask = (time >= T_MIN) & (time <= T_MAX)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit, y_fit = t_fit[valid], y_fit[valid]

    if len(t_fit) < 50:
        return None

    # Initial guesses - same as user's OG script
    half_range = (y_fit.max() - y_fit.min()) / 2
    p0 = [y_fit.min(), half_range, 1.0, half_range, 0.1]

    try:
        popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=p0, maxfev=50000)
        A0, A1, k1, A2, k2 = popt

        # dydt0 = -A1*k1 - A2*k2
        dydt0 = -A1 * k1 - A2 * k2

        # Amplitude
        n = max(1, len(y_fit) // 20)
        amplitude = np.median(y_fit[:n]) - np.median(y_fit[-n:])

        # Rate using USER'S formula: rate = |dydt0| * Q / CO2
        rate = abs(dydt0) * Q / CO2

        return {
            'dydt0': dydt0,
            'rate': rate,
            'amplitude': amplitude,
            'A1': A1, 'k1': k1, 'A2': A2, 'k2': k2
        }
    except:
        return None


def weighted_least_squares(x, y, sigma):
    """User's weighted LS formula."""
    # Handle zero sigma
    sigma = np.where(sigma < 1e-10, 1e-10, sigma)

    w = 1.0 / (sigma ** 2)

    S = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)

    denom = S * Sxx - Sx * Sx

    slope = (S * Sxy - Sx * Sy) / denom
    intercept = (Sxx * Sy - Sx * Sxy) / denom

    se_slope = np.sqrt(S / denom)
    se_intercept = np.sqrt(Sxx / denom)

    return slope, se_slope, intercept, se_intercept


def analyze_multiconc_sheet(xl, sheet_name, label):
    """Analyze a sheet with concentration columns."""
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    row0 = df.iloc[0].values

    conc_cols = [i for i, v in enumerate(row0)
                 if isinstance(v, str) and 'concentration' in v.lower()]

    if not conc_cols:
        return None

    conc_data = []

    for i, col_idx in enumerate(conc_cols):
        header = str(row0[col_idx])
        match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
        conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

        next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
        time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

        rates = []
        amps = []

        for tc in range(col_idx + 1, next_idx):
            col_data = df.iloc[2:, tc]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue

            trial_data = pd.to_numeric(col_data, errors='coerce').values
            result = fit_trial(time, trial_data)

            if result and result['amplitude'] <= AMP_MAX:
                rates.append(result['rate'])
                amps.append(result['amplitude'])

        if rates:
            mean_rate = np.mean(rates)
            std_rate = np.std(rates, ddof=1) if len(rates) > 1 else 0.001
            conc_data.append({
                'conc_mM': conc_mM,
                'conc_M': conc_mM / 1000,
                'mean_rate': mean_rate,
                'std_rate': std_rate,
                'mean_amp': np.mean(amps),
                'n': len(rates)
            })

    if len(conc_data) < 2:
        return None

    # Weighted least squares
    x = np.array([d['conc_M'] for d in conc_data])
    y = np.array([d['mean_rate'] for d in conc_data])
    sigma = np.array([d['std_rate'] for d in conc_data])

    k_cat, se_kcat, k_w, se_kw = weighted_least_squares(x, y, sigma)

    return {
        'label': label,
        'k_cat': k_cat,
        'se_kcat': se_kcat,
        'k_uncat': k_w,
        'se_kuncat': se_kw,
        'mean_amp': np.mean([d['mean_amp'] for d in conc_data]),
        'conc_data': conc_data
    }


def analyze_simple_file(filepath, label):
    """Analyze a file with sheet names as concentrations."""
    try:
        xl = pd.ExcelFile(filepath)
    except:
        return None

    conc_data = []

    for sheet in xl.sheet_names:
        if sheet.lower() in ['results', 'summary', 'q', 'uncatalyzed', 'cyclen', 'old', 'co2 uncat']:
            continue

        match = re.search(r'(\d+)p?(\d*)mm', sheet.lower())
        if not match:
            continue

        if match.group(2):
            conc_mM = float(f"{match.group(1)}.{match.group(2)}")
        else:
            conc_mM = float(match.group(1))

        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values

        rates = []
        amps = []

        for col in range(1, df.shape[1]):
            col_data = df.iloc[1:, col]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue

            header = df.iloc[0, col] if col < len(df.iloc[0]) else None
            if isinstance(header, str) and 'Q' in str(header).upper():
                continue

            trial_data = pd.to_numeric(col_data, errors='coerce').values
            result = fit_trial(time, trial_data)

            if result and result['amplitude'] <= AMP_MAX:
                rates.append(result['rate'])
                amps.append(result['amplitude'])

        if rates:
            mean_rate = np.mean(rates)
            std_rate = np.std(rates, ddof=1) if len(rates) > 1 else 0.001
            conc_data.append({
                'conc_mM': conc_mM,
                'conc_M': conc_mM / 1000,
                'mean_rate': mean_rate,
                'std_rate': std_rate,
                'mean_amp': np.mean(amps),
                'n': len(rates)
            })

    if len(conc_data) < 2:
        return None

    x = np.array([d['conc_M'] for d in conc_data])
    y = np.array([d['mean_rate'] for d in conc_data])
    sigma = np.array([d['std_rate'] for d in conc_data])

    k_cat, se_kcat, k_w, se_kw = weighted_least_squares(x, y, sigma)

    return {
        'label': label,
        'k_cat': k_cat,
        'se_kcat': se_kcat,
        'k_uncat': k_w,
        'se_kuncat': se_kw,
        'mean_amp': np.mean([d['mean_amp'] for d in conc_data]),
        'conc_data': conc_data
    }


def main():
    print("=" * 85)
    print("FINAL ANALYSIS - USER'S EXACT METHOD")
    print("rate = |dydt0| * Q / CO2, Weighted Least Squares")
    print("Excluding amplitude outliers > 0.065")
    print("=" * 85)

    all_results = []

    # ============ BENZ_CYCLEN ============
    print("\n" + "=" * 70)
    print("BENZ_CYCLEN (excluding 12_13 outlier)")
    print("=" * 70)

    benz_file = 'Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx'
    xl = pd.ExcelFile(benz_file)

    for sheet in ['12_05', '01_14']:
        result = analyze_multiconc_sheet(xl, sheet, f'Benz/{sheet}')
        if result:
            print(f"\n  {sheet}:")
            print(f"    Amp = {result['mean_amp']:.4f}")
            for d in result['conc_data']:
                print(f"    {d['conc_mM']}mM: rate = {d['mean_rate']:.4f} ± {d['std_rate']:.4f} (n={d['n']})")
            print(f"    k_cat = {result['k_cat']:.1f} ± {result['se_kcat']:.1f}")
            print(f"    k_uncat = {result['k_uncat']:.4f} ± {result['se_kuncat']:.4f}")
            all_results.append(result)

    # ============ nBu_CYCLEN ============
    print("\n" + "=" * 70)
    print("nBu_CYCLEN")
    print("=" * 70)

    nbu_files = [
        ('Data/Cyclen_Study/nBu_Cyclen/Butyl_01_14.xlsx', 'nBu/01_14'),
        ('Data/Cyclen_Study/nBu_Cyclen/nBu_cyc_02_02.xlsx', 'nBu/02_02'),
    ]

    for filepath, label in nbu_files:
        result = analyze_simple_file(filepath, label)
        if result:
            print(f"\n  {label}:")
            print(f"    Amp = {result['mean_amp']:.4f}")
            for d in result['conc_data']:
                print(f"    {d['conc_mM']}mM: rate = {d['mean_rate']:.4f} ± {d['std_rate']:.4f} (n={d['n']})")
            print(f"    k_cat = {result['k_cat']:.1f} ± {result['se_kcat']:.1f}")
            print(f"    k_uncat = {result['k_uncat']:.4f} ± {result['se_kuncat']:.4f}")
            all_results.append(result)

    # ============ HEXYL_CYCLEN ============
    print("\n" + "=" * 70)
    print("HEXYL_CYCLEN")
    print("=" * 70)

    hexyl_files = [
        ('Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_Cyclen_01_14.xlsx', 'Hexyl/01_14'),
        ('Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_cyc_02_03.xlsx', 'Hexyl/02_03'),
    ]

    for filepath, label in hexyl_files:
        result = analyze_simple_file(filepath, label)
        if result:
            print(f"\n  {label}:")
            print(f"    Amp = {result['mean_amp']:.4f}")
            for d in result['conc_data']:
                print(f"    {d['conc_mM']}mM: rate = {d['mean_rate']:.4f} ± {d['std_rate']:.4f} (n={d['n']})")
            print(f"    k_cat = {result['k_cat']:.1f} ± {result['se_kcat']:.1f}")
            print(f"    k_uncat = {result['k_uncat']:.4f} ± {result['se_kuncat']:.4f}")
            all_results.append(result)

    # ============ CYCLEN_BASELINE ============
    print("\n" + "=" * 70)
    print("CYCLEN_BASELINE")
    print("=" * 70)

    cyclen_files = [
        ('Data/Cyclen_Study/Cyclen_baseline/Cyclen_01_13.xlsx', 'Cyclen/01_13'),
        ('Data/Cyclen_Study/Cyclen_baseline/Cyclen_5xDye.xlsx', 'Cyclen/5xDye'),
    ]

    for filepath, label in cyclen_files:
        result = analyze_simple_file(filepath, label)
        if result:
            print(f"\n  {label}:")
            print(f"    Amp = {result['mean_amp']:.4f}")
            for d in result['conc_data']:
                print(f"    {d['conc_mM']}mM: rate = {d['mean_rate']:.4f} ± {d['std_rate']:.4f} (n={d['n']})")
            print(f"    k_cat = {result['k_cat']:.1f} ± {result['se_kcat']:.1f}")
            print(f"    k_uncat = {result['k_uncat']:.4f} ± {result['se_kuncat']:.4f}")
            all_results.append(result)

    # ============ SUMMARY TABLE ============
    print("\n" + "=" * 85)
    print("SUMMARY TABLE")
    print("=" * 85)
    print(f"\n{'Dataset':<20} {'Amp':<8} {'k_cat':<12} {'±':<8} {'k_uncat':<10} {'±':<8}")
    print("-" * 70)

    for r in all_results:
        print(f"{r['label']:<20} {r['mean_amp']:<8.4f} {r['k_cat']:<12.1f} {r['se_kcat']:<8.1f} {r['k_uncat']:<10.4f} {r['se_kuncat']:<8.4f}")

    # ============ BY CATALYST TYPE ============
    print("\n" + "=" * 85)
    print("BY CATALYST TYPE")
    print("=" * 85)

    groups = {
        'Benzyl': [],
        'nButyl': [],
        'Hexyl': [],
        'Cyclen': []
    }

    for r in all_results:
        if 'Benz' in r['label']:
            groups['Benzyl'].append(r)
        elif 'nBu' in r['label']:
            groups['nButyl'].append(r)
        elif 'Hexyl' in r['label']:
            groups['Hexyl'].append(r)
        elif 'Cyclen' in r['label']:
            groups['Cyclen'].append(r)

    print(f"\n{'Catalyst':<12} {'Measurements':<35} {'Average k_cat':<15}")
    print("-" * 65)

    for cat, results in groups.items():
        if results:
            kcats = [r['k_cat'] for r in results]
            labels = [r['label'].split('/')[-1] for r in results]
            meas_str = ', '.join([f"{l}:{k:.0f}" for l, k in zip(labels, kcats)])
            avg = np.mean(kcats)
            print(f"{cat:<12} {meas_str:<35} {avg:<15.0f}")


if __name__ == "__main__":
    main()
