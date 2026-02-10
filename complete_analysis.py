#!/usr/bin/env python3
"""
Complete analysis with corrected Q values:
- Monodentate study: Q = -0.402
- Cyclen study (01_13 and later): Q = -0.452

Methods:
1. Working Method: rate = |dydt0| * |Q| / [CO2], then k_obs = rate
2. Weighted k: k_obs = |A1*k1 + A2*k2| / |A1 + A2|

Linear regression: k_obs = k_uncat + k_cat * [Catalyst]
Using weighted least squares with w = 1/σ²
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import pathlib
import re

# Constants
CO2 = 0.0338  # M
Q_MONO = 0.402  # For Monodentate study
Q_CYCLEN = 0.452  # For Cyclen study (01_13 and later)
T_MIN, T_MAX = 0.02, 40.0


def remove_outliers(values, n_sigma=1.0):
    """
    Remove outliers using a simple 1-std approach.

    This removes only the most extreme outliers - values that fall outside
    1 standard deviation from the mean. This is conservative and only removes
    clear outliers while preserving the natural variance in the data.

    For small datasets (n < 3), no outlier removal is performed.
    """
    if len(values) < 3:
        return values  # Need at least 3 points to detect outliers

    arr = np.array(values)

    # Calculate mean and std
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)

    # If std is essentially zero, all values are the same - keep all
    if std_val < 1e-10:
        return arr.tolist()

    # Keep values within n_sigma standard deviations of the mean
    mask = np.abs(arr - mean_val) <= n_sigma * std_val
    filtered = arr[mask].tolist()

    # Ensure we keep at least 2 points for regression
    return filtered if len(filtered) >= 2 else arr.tolist()


def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_trial(time, absorbance, Q):
    """Fit double exp and return both rate methods."""
    mask = (time >= T_MIN) & (time <= T_MAX)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit, y_fit = t_fit[valid], y_fit[valid]

    if len(t_fit) < 50:
        return None

    half_range = (y_fit.max() - y_fit.min()) / 2
    p0 = [y_fit.min(), half_range, 1.0, half_range, 0.1]

    try:
        popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=p0, maxfev=50000)
        A0, A1, k1, A2, k2 = popt

        # Amplitude
        n = max(1, len(y_fit) // 20)
        amplitude = np.median(y_fit[:n]) - np.median(y_fit[-n:])

        # Working Method: rate = |dydt0| * |Q| / [CO2]
        dydt0 = -A1 * k1 - A2 * k2
        k_obs_working = abs(dydt0) * abs(Q) / CO2

        # Weighted k: k_obs = |A1*k1 + A2*k2| / |A1 + A2|
        k_obs_weighted = abs(A1 * k1 + A2 * k2) / abs(A1 + A2)

        # R² for quality check
        y_pred = double_exp(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # QUALITY FILTERS:
        # 1. dydt0 should be negative (absorbance decreases)
        # 2. k_obs should be reasonable (< 2 /s for these systems)
        # 3. Rate constants k1, k2 should be positive and reasonable
        if dydt0 > 0:  # Wrong sign - bad fit
            return None
        if k_obs_working > 2.0:  # Way too high
            return None
        if k1 < 0 or k2 < 0:  # Negative rate constants
            return None
        if k1 > 50 or k2 > 50:  # Unreasonably fast
            return None

        return {
            'dydt0': dydt0,
            'k_obs_working': k_obs_working,
            'k_obs_weighted': k_obs_weighted,
            'amplitude': amplitude,
            'r2': r2,
            'A1': A1, 'k1': k1, 'A2': A2, 'k2': k2
        }
    except:
        return None


def weighted_least_squares(x, y, sigma):
    """Weighted LS with proper error handling."""
    sigma = np.where(sigma < 1e-10, 1e-10, sigma)
    w = 1.0 / (sigma ** 2)

    S = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)

    denom = S * Sxx - Sx * Sx
    if abs(denom) < 1e-20:
        return None, None, None, None

    slope = (S * Sxy - Sx * Sy) / denom
    intercept = (Sxx * Sy - Sx * Sxy) / denom
    se_slope = np.sqrt(S / denom)
    se_intercept = np.sqrt(Sxx / denom)

    return slope, se_slope, intercept, se_intercept


def analyze_monodentate():
    """Analyze Monodentate study with Q = -0.402."""
    print("=" * 100)
    print("MONODENTATE STUDY (Q = -0.402)")
    print("=" * 100)

    base_path = pathlib.Path("Data/Monodentate_Study")
    results = []

    # Find all xlsx files
    files = list(base_path.glob("**/*.xlsx"))

    for f in sorted(files):
        # Extract ligand equivalents from filename
        match = re.search(r'(\d+)x', f.stem, re.IGNORECASE)
        if not match:
            continue
        ligand_eq = int(match.group(1))

        # Extract date from path
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', str(f))
        date = date_match.group(1) if date_match else 'unknown'

        try:
            xl = pd.ExcelFile(f)

            conc_data_working = {}
            conc_data_weighted = {}

            for sheet in xl.sheet_names:
                if sheet.lower() in ['results', 'summary', 'q']:
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

                working_vals = []
                weighted_vals = []

                for col in range(1, df.shape[1]):
                    col_data = df.iloc[1:, col]
                    if col_data.isna().sum() > len(col_data) * 0.5:
                        continue

                    header = df.iloc[0, col] if col < len(df.iloc[0]) else None
                    if isinstance(header, str) and 'Q' in str(header).upper():
                        continue

                    trial_data = pd.to_numeric(col_data, errors='coerce').values
                    result = fit_trial(time, trial_data, Q_MONO)

                    if result and result['r2'] > 0.99 and result['amplitude'] > 0.01:
                        working_vals.append(result['k_obs_working'])
                        weighted_vals.append(result['k_obs_weighted'])

                if working_vals:
                    conc_data_working[conc_mM] = working_vals
                    conc_data_weighted[conc_mM] = weighted_vals

            # Do regression for both methods
            if len(conc_data_working) >= 2:
                # Remove outliers from each concentration group
                for c in conc_data_working:
                    conc_data_working[c] = remove_outliers(conc_data_working[c])
                    conc_data_weighted[c] = remove_outliers(conc_data_weighted[c])

                # Working method
                x = np.array([c / 1000 for c in sorted(conc_data_working.keys())])
                y_work = np.array([np.mean(conc_data_working[c]) for c in sorted(conc_data_working.keys())])
                sigma_work = np.array([np.std(conc_data_working[c], ddof=1) if len(conc_data_working[c]) > 1 else 0.001
                                       for c in sorted(conc_data_working.keys())])

                k_cat_work, se_work, k_uncat_work, se_uncat_work = weighted_least_squares(x, y_work, sigma_work)

                # Weighted method
                y_weight = np.array([np.mean(conc_data_weighted[c]) for c in sorted(conc_data_weighted.keys())])
                sigma_weight = np.array([np.std(conc_data_weighted[c], ddof=1) if len(conc_data_weighted[c]) > 1 else 0.001
                                         for c in sorted(conc_data_weighted.keys())])

                k_cat_weighted, se_weighted, k_uncat_weighted, se_uncat_weighted = weighted_least_squares(x, y_weight, sigma_weight)

                if k_cat_work is not None:
                    results.append({
                        'ligand_eq': ligand_eq,
                        'date': date,
                        'k_cat_working': k_cat_work,
                        'se_working': se_work,
                        'k_uncat_working': k_uncat_work,
                        'se_uncat_working': se_uncat_work,
                        'k_cat_weighted': k_cat_weighted,
                        'se_weighted': se_weighted,
                        'k_uncat_weighted': k_uncat_weighted,
                        'se_uncat_weighted': se_uncat_weighted,
                    })

        except Exception as e:
            print(f"Error with {f}: {e}")

    # Print table
    print(f"\n{'Ligand':<8} {'Date':<12} {'k_cat(work)':<12} {'±':<8} {'k_uncat':<10} {'±':<8} {'k_cat(wt)':<12} {'±':<8} {'k_uncat':<10} {'±':<8}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x['ligand_eq']):
        print(f"{r['ligand_eq']}x{'':<5} {r['date']:<12} "
              f"{r['k_cat_working']:<12.1f} {r['se_working']:<8.1f} "
              f"{r['k_uncat_working']:<10.4f} {r['se_uncat_working']:<8.4f} "
              f"{r['k_cat_weighted']:<12.1f} {r['se_weighted']:<8.1f} "
              f"{r['k_uncat_weighted']:<10.4f} {r['se_uncat_weighted']:<8.4f}")

    return results


def analyze_cyclen_study():
    """Analyze Cyclen study with Q = -0.452."""
    print("\n" + "=" * 100)
    print("CYCLEN STUDY (Q = -0.452)")
    print("Excluding: Benz 12_13, Benz 01_14, nBu 02_02")
    print("=" * 100)

    results = []

    # Datasets to analyze
    # All Cyclen study data uses Q = -0.452
    datasets = [
        # Benz_Cyclen - only 12_05
        {
            'name': 'Benz_Cyclen',
            'date': '12_05',
            'file': 'Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_12_05.xlsx',
            'type': 'simple',
            'Q': Q_CYCLEN
        },
        # nBu_Cyclen - only 01_14 (uses Q_CYCLEN because 01_14 >= 01_13)
        {
            'name': 'nBu_Cyclen',
            'date': '01_14',
            'file': 'Data/Cyclen_Study/nBu_Cyclen/Butyl_01_14.xlsx',
            'type': 'simple',
            'Q': Q_CYCLEN
        },
        # Hexyl_Cyclen - both dates (both use Q_CYCLEN because >= 01_13)
        {
            'name': 'Hexyl_Cyclen',
            'date': '01_14',
            'file': 'Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_Cyclen_01_14.xlsx',
            'type': 'simple',
            'Q': Q_CYCLEN
        },
        {
            'name': 'Hexyl_Cyclen',
            'date': '02_03',
            'file': 'Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_cyc_02_03.xlsx',
            'type': 'simple',
            'Q': Q_CYCLEN
        },
        # Cyclen baseline - 01_13 (uses Q_CYCLEN)
        {
            'name': 'Cyclen',
            'date': '01_13',
            'file': 'Data/Cyclen_Study/Cyclen_baseline/Cyclen_01_13.xlsx',
            'type': 'simple',
            'Q': Q_CYCLEN
        },
    ]

    for ds in datasets:
        try:
            conc_data_working = {}
            conc_data_weighted = {}
            Q = ds.get('Q', Q_CYCLEN)  # Use dataset-specific Q, default to Q_CYCLEN

            if ds['type'] == 'multiconc':
                # Multi-concentration sheet format
                xl = pd.ExcelFile(ds['file'])
                df = pd.read_excel(xl, sheet_name=ds['sheet'], header=None)
                row0 = df.iloc[0].values

                conc_cols = [i for i, v in enumerate(row0)
                             if isinstance(v, str) and 'concentration' in v.lower()]

                for i, col_idx in enumerate(conc_cols):
                    header = str(row0[col_idx])
                    match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
                    conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

                    next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
                    time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

                    working_vals = []
                    weighted_vals = []

                    for tc in range(col_idx + 1, next_idx):
                        col_data = df.iloc[2:, tc]
                        if col_data.isna().sum() > len(col_data) * 0.5:
                            continue

                        trial_data = pd.to_numeric(col_data, errors='coerce').values
                        result = fit_trial(time, trial_data, Q)

                        if result and result['r2'] > 0.99 and result['amplitude'] > 0.01:
                            working_vals.append(result['k_obs_working'])
                            weighted_vals.append(result['k_obs_weighted'])

                    if working_vals:
                        conc_data_working[conc_mM] = working_vals
                        conc_data_weighted[conc_mM] = weighted_vals

                # Remove outliers from multiconc data
                for c in list(conc_data_working.keys()):
                    conc_data_working[c] = remove_outliers(conc_data_working[c])
                    conc_data_weighted[c] = remove_outliers(conc_data_weighted[c])

            else:
                # Simple format with sheet names as concentrations
                xl = pd.ExcelFile(ds['file'])

                for sheet in xl.sheet_names:
                    if sheet.lower() in ['results', 'summary', 'q', 'cyclen', 'old', 'co2 uncat']:
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

                    working_vals = []
                    weighted_vals = []

                    for col in range(1, df.shape[1]):
                        col_data = df.iloc[1:, col]
                        if col_data.isna().sum() > len(col_data) * 0.5:
                            continue

                        header = df.iloc[0, col] if col < len(df.iloc[0]) else None
                        if isinstance(header, str) and 'Q' in str(header).upper():
                            continue

                        trial_data = pd.to_numeric(col_data, errors='coerce').values
                        result = fit_trial(time, trial_data, Q)

                        if result and result['r2'] > 0.99 and result['amplitude'] > 0.01:
                            working_vals.append(result['k_obs_working'])
                            weighted_vals.append(result['k_obs_weighted'])

                    if working_vals:
                        conc_data_working[conc_mM] = working_vals
                        conc_data_weighted[conc_mM] = weighted_vals

                # Remove outliers from simple format data
                for c in list(conc_data_working.keys()):
                    conc_data_working[c] = remove_outliers(conc_data_working[c])
                    conc_data_weighted[c] = remove_outliers(conc_data_weighted[c])

            # Do regression
            if len(conc_data_working) >= 2:
                x = np.array([c / 1000 for c in sorted(conc_data_working.keys())])

                y_work = np.array([np.mean(conc_data_working[c]) for c in sorted(conc_data_working.keys())])
                sigma_work = np.array([np.std(conc_data_working[c], ddof=1) if len(conc_data_working[c]) > 1 else 0.001
                                       for c in sorted(conc_data_working.keys())])

                y_weight = np.array([np.mean(conc_data_weighted[c]) for c in sorted(conc_data_weighted.keys())])
                sigma_weight = np.array([np.std(conc_data_weighted[c], ddof=1) if len(conc_data_weighted[c]) > 1 else 0.001
                                         for c in sorted(conc_data_weighted.keys())])

                k_cat_work, se_work, k_uncat_work, se_uncat_work = weighted_least_squares(x, y_work, sigma_work)
                k_cat_weighted, se_weighted, k_uncat_weighted, se_uncat_weighted = weighted_least_squares(x, y_weight, sigma_weight)

                if k_cat_work is not None:
                    results.append({
                        'name': ds['name'],
                        'date': ds['date'],
                        'Q': Q,
                        'k_cat_working': k_cat_work,
                        'se_working': se_work,
                        'k_uncat_working': k_uncat_work,
                        'se_uncat_working': se_uncat_work,
                        'k_cat_weighted': k_cat_weighted,
                        'se_weighted': se_weighted,
                        'k_uncat_weighted': k_uncat_weighted,
                        'se_uncat_weighted': se_uncat_weighted,
                        'n_conc': len(conc_data_working),
                    })

                    print(f"\n{ds['name']} ({ds['date']}) [Q={Q}]:")
                    for c in sorted(conc_data_working.keys()):
                        print(f"  {c}mM: work={np.mean(conc_data_working[c]):.4f}±{np.std(conc_data_working[c], ddof=1):.4f}, "
                              f"wt={np.mean(conc_data_weighted[c]):.4f}±{np.std(conc_data_weighted[c], ddof=1):.4f} (n={len(conc_data_working[c])})")

        except Exception as e:
            print(f"Error with {ds['name']} {ds['date']}: {e}")

    # Print table
    print(f"\n{'Catalyst':<15} {'Date':<8} {'k_cat(work)':<12} {'±':<8} {'k_uncat':<10} {'±':<8} {'k_cat(wt)':<12} {'±':<8} {'k_uncat':<10} {'±':<8}")
    print("-" * 105)

    for r in results:
        print(f"{r['name']:<15} {r['date']:<8} "
              f"{r['k_cat_working']:<12.1f} {r['se_working']:<8.1f} "
              f"{r['k_uncat_working']:<10.4f} {r['se_uncat_working']:<8.4f} "
              f"{r['k_cat_weighted']:<12.1f} {r['se_weighted']:<8.1f} "
              f"{r['k_uncat_weighted']:<10.4f} {r['se_uncat_weighted']:<8.4f}")

    return results


def main():
    print("=" * 100)
    print("COMPLETE KINETICS ANALYSIS")
    print("=" * 100)
    print("\nFormulas:")
    print("  Working Method: k_obs = |dydt0| × |Q| / [CO2]")
    print("  Weighted k:     k_obs = |A1·k1 + A2·k2| / |A1 + A2|")
    print("  Linear model:   k_obs = k_uncat + k_cat × [Catalyst]")
    print("  Regression:     Weighted least squares (w = 1/σ²)")
    print()
    print("Q values:")
    print("  Monodentate study: Q = -0.402")
    print("  Cyclen study:      Q = -0.452")

    mono_results = analyze_monodentate()
    cyclen_results = analyze_cyclen_study()

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print("\nMonodentate Study - Working Method averages by ligand equivalents:")
    ligand_groups = {}
    for r in mono_results:
        leq = r['ligand_eq']
        if leq not in ligand_groups:
            ligand_groups[leq] = []
        ligand_groups[leq].append(r['k_cat_working'])

    for leq in sorted(ligand_groups.keys()):
        vals = ligand_groups[leq]
        print(f"  {leq}x: {np.mean(vals):.1f} ± {np.std(vals, ddof=1):.1f} (n={len(vals)})" if len(vals) > 1 else f"  {leq}x: {vals[0]:.1f}")

    print("\nCyclen Study - Working Method:")
    for r in cyclen_results:
        print(f"  {r['name']} ({r['date']}): k_cat = {r['k_cat_working']:.1f} ± {r['se_working']:.1f}, k_uncat = {r['k_uncat_working']:.4f}")


if __name__ == "__main__":
    main()
