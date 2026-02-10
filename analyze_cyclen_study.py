#!/usr/bin/env python3
"""
Analyze all Cyclen study datasets and compute averages per ligand.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Constants
K_UNCAT_FIXED = 0.12
T_MIN = 0.02
T_MAX = 40.0


def extract_concentration(s: str) -> Optional[float]:
    import re
    s = str(s).lower().replace(' ', '')
    patterns = [r'(\d+)p(\d+)mm', r'(\d+\.?\d*)mm']
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return float(f"{groups[0]}.{groups[1]}")
            return float(groups[0])
    return None


def load_data(file_path: str) -> Dict[float, Tuple[np.ndarray, List[np.ndarray]]]:
    xl = pd.ExcelFile(file_path)
    data = {}

    for sheet in xl.sheet_names:
        conc = extract_concentration(sheet)
        if conc is None:
            continue

        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values

        trials = []
        for col in range(1, df.shape[1]):
            trial = pd.to_numeric(df.iloc[1:, col], errors='coerce').values
            valid = ~np.isnan(trial)
            if valid.sum() > 100:
                diffs = np.diff(trial[valid])
                if not ((diffs > 0).sum() / len(diffs) > 0.95):
                    trials.append(trial)

        if trials:
            data[conc] = (time, trials)

    return data


def prepare_data(time: np.ndarray, absorbance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (time >= T_MIN) & (time <= T_MAX)
    t = time[mask]
    y = absorbance[mask]
    valid = ~np.isnan(t) & ~np.isnan(y)
    return t[valid], y[valid]


def _linear_regression(concs_M: List[float], k_obs: List[float]) -> Tuple[float, float, float, float, float]:
    if len(concs_M) < 2:
        return (np.nan,) * 5
    x = np.array(concs_M)
    y = np.array(k_obs)
    slope, intercept, r_value, _, std_err = linregress(x, y)
    n = len(x)
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean)**2)
    y_pred = slope * x + intercept
    mse = np.sum((y - y_pred)**2) / (n - 2) if n > 2 else 0
    intercept_err = np.sqrt(mse * (1/n + x_mean**2 / ss_x)) if n > 2 else 0
    return (slope, std_err, intercept, intercept_err, r_value**2)


def method_single_exp(data):
    """Single exponential fitting."""
    concs_M = []
    k_obs_values = []

    for conc_mM in sorted(data.keys()):
        time, trials = data[conc_mM]
        k_vals = []

        for trial in trials:
            t, y = prepare_data(time, trial)
            if len(t) < 50:
                continue
            y_start = np.median(y[:20])
            y_end = np.median(y[-50:])

            def single_exp(t, A0, A, k):
                return A0 + A * np.exp(-k * t)

            try:
                popt, _ = curve_fit(single_exp, t, y, p0=[y_end, y_start - y_end, 0.3],
                                   bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 20]),
                                   maxfev=20000)
                if 0.05 < popt[2] < 10:
                    k_vals.append(popt[2])
            except:
                pass

        if k_vals:
            concs_M.append(conc_mM / 1000)
            k_obs_values.append(np.mean(k_vals))

    return _linear_regression(concs_M, k_obs_values)


def process_dataset(name: str, path: str):
    """Process a single dataset."""
    print(f"  Processing: {name}")
    data = load_data(path)
    if not data:
        print(f"    ERROR: No data loaded!")
        return None

    print(f"    Concentrations: {sorted(data.keys())} mM")
    result = method_single_exp(data)
    return {
        'name': name,
        'k_cat': result[0],
        'k_cat_err': result[1],
        'k_uncat': result[2],
        'k_uncat_err': result[3],
        'R2': result[4]
    }


# Define all datasets by ligand
datasets = {
    'Cyclen': [
        ("Cyclen_01_13", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Cyclen_baseline/Cyclen_01_13.xlsx"),
        ("Cyclen_5xDye", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Cyclen_baseline/Cyclen_5xDye.xlsx"),
    ],
    'nBu-Cyclen': [
        ("nBu_02_02", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/nBu_Cyclen/nBu_cyc_02_02.xlsx"),
        ("nBu_01_14", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/nBu_Cyclen/Butyl_01_14.xlsx"),
    ],
    'Hexyl-Cyclen': [
        ("Hexyl_02_03", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_cyc_02_03.xlsx"),
        ("Hexyl_01_14", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Hexyl_Cyclen/Hexyl_Cyclen_01_14.xlsx"),
    ],
}

# Process all datasets
all_results = {}
for ligand, files in datasets.items():
    print(f"\n{ligand}:")
    all_results[ligand] = []
    for name, path in files:
        result = process_dataset(name, path)
        if result:
            all_results[ligand].append(result)

# Print individual results
print("\n" + "=" * 100)
print("INDIVIDUAL DATASET RESULTS (Single Exponential)")
print("=" * 100)
print(f"{'Ligand':<15} | {'Dataset':<15} | {'k_cat (/M/s)':<20} | {'k_uncat (/s)':<15} | {'R²':<10}")
print("-" * 100)

for ligand, results in all_results.items():
    for r in results:
        k_cat_str = f"{r['k_cat']:.1f} ± {r['k_cat_err']:.1f}" if not np.isnan(r['k_cat']) else "N/A"
        k_uncat_str = f"{r['k_uncat']:.4f}" if not np.isnan(r['k_uncat']) else "N/A"
        r2_str = f"{r['R2']:.4f}" if not np.isnan(r['R2']) else "N/A"
        print(f"{ligand:<15} | {r['name']:<15} | {k_cat_str:<20} | {k_uncat_str:<15} | {r2_str:<10}")

# Calculate and print averages
print("\n" + "=" * 100)
print("AVERAGED RESULTS BY LIGAND")
print("=" * 100)
print(f"{'Ligand':<15} | {'n datasets':<12} | {'k_cat (/M/s)':<25} | {'k_uncat (/s)':<20} | {'Avg R²':<10}")
print("-" * 100)

summary = []
for ligand, results in all_results.items():
    if not results:
        continue

    k_cats = [r['k_cat'] for r in results if not np.isnan(r['k_cat'])]
    k_uncats = [r['k_uncat'] for r in results if not np.isnan(r['k_uncat'])]
    r2s = [r['R2'] for r in results if not np.isnan(r['R2'])]

    n = len(k_cats)
    if n > 0:
        k_cat_mean = np.mean(k_cats)
        k_cat_std = np.std(k_cats, ddof=1) if n > 1 else 0
        k_uncat_mean = np.mean(k_uncats)
        k_uncat_std = np.std(k_uncats, ddof=1) if n > 1 else 0
        r2_mean = np.mean(r2s)

        k_cat_str = f"{k_cat_mean:.1f} ± {k_cat_std:.1f}" if n > 1 else f"{k_cat_mean:.1f}"
        k_uncat_str = f"{k_uncat_mean:.4f} ± {k_uncat_std:.4f}" if n > 1 else f"{k_uncat_mean:.4f}"

        print(f"{ligand:<15} | {n:<12} | {k_cat_str:<25} | {k_uncat_str:<20} | {r2_mean:.4f}")

        summary.append({
            'ligand': ligand,
            'n': n,
            'k_cat_mean': k_cat_mean,
            'k_cat_std': k_cat_std,
            'k_uncat_mean': k_uncat_mean,
            'r2_mean': r2_mean
        })

print("=" * 100)

# Print ranked summary
print("\n" + "-" * 60)
print("RANKED BY CATALYTIC ACTIVITY (k_cat)")
print("-" * 60)
summary_sorted = sorted(summary, key=lambda x: x['k_cat_mean'], reverse=True)
for i, s in enumerate(summary_sorted, 1):
    print(f"{i}. {s['ligand']:<15} {s['k_cat_mean']:.1f} ± {s['k_cat_std']:.1f} /M/s")
