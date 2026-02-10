#!/usr/bin/env python3
"""
Analyze all monodentate study datasets with multiple fitting methods.
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


# Method 0: Single Exponential
def method0_single_exp(data):
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


# Method 1: Double Exp with amplitude-weighted rate
def method1_double_exp_weighted(data):
    concs_M = []
    k_obs_values = []

    for conc_mM in sorted(data.keys()):
        time, trials = data[conc_mM]
        k_vals = []

        for trial in trials:
            t, y = prepare_data(time, trial)
            if len(t) < 100:
                continue
            y_start = np.median(y[:20])
            y_end = np.median(y[-50:])
            total_amp = y_start - y_end

            if abs(total_amp) < 0.005:
                continue

            def objective(params):
                A0, A1, k1, A2, k2 = params
                pred = A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)
                return np.sum((y - pred)**2)

            best_result = None
            best_cost = np.inf

            for k1_init in [0.5, 1.0, 2.0]:
                for k2_init in [0.08, 0.12, 0.2]:
                    if k1_init <= k2_init:
                        continue
                    for frac in [0.5, 0.7]:
                        p0 = [y_end, total_amp * frac, k1_init, total_amp * (1 - frac), k2_init]
                        bounds = [(y_end - 0.05, y_end + 0.05), (0.001, total_amp * 2),
                                 (0.2, 15.0), (0.001, total_amp * 2), (0.01, 0.5)]
                        try:
                            res = minimize(objective, p0, method='L-BFGS-B', bounds=bounds)
                            if res.success and res.fun < best_cost:
                                best_cost = res.fun
                                best_result = res.x
                        except:
                            pass

            if best_result is not None:
                A0, A1, k1, A2, k2 = best_result
                if abs(A1) + abs(A2) > 0.001:
                    k_weighted = (abs(A1) * k1 + abs(A2) * k2) / (abs(A1) + abs(A2))
                    if 0.1 < k_weighted < 10:
                        k_vals.append(k_weighted)

        if k_vals:
            concs_M.append(conc_mM / 1000)
            k_obs_values.append(np.mean(k_vals))

    return _linear_regression(concs_M, k_obs_values)


# Method 2: Initial Rate
def method2_initial_rate(data):
    concs_M = []
    k_obs_values = []

    for conc_mM in sorted(data.keys()):
        time, trials = data[conc_mM]
        k_vals = []

        for trial in trials:
            t, y = prepare_data(time, trial)
            if len(t) < 100:
                continue
            y_start = np.median(y[:10])
            y_end = np.median(y[-50:])
            delta_A = y_start - y_end

            if abs(delta_A) < 0.005:
                continue

            threshold = y_start - 0.10 * delta_A
            idx = np.argmax(y < threshold) if delta_A > 0 else np.argmax(y > threshold)
            if idx < 20:
                idx = min(100, len(t) // 5)

            t_early = t[:idx] - t[0]
            y_early = y[:idx]

            if len(t_early) < 15:
                continue

            try:
                coeffs = np.polyfit(t_early, y_early, 2)
                dydt_0 = coeffs[1]
                k_obs = abs(dydt_0 / delta_A)
                if 0.05 < k_obs < 15:
                    k_vals.append(k_obs)
            except:
                pass

        if k_vals:
            concs_M.append(conc_mM / 1000)
            k_obs_values.append(np.mean(k_vals))

    return _linear_regression(concs_M, k_obs_values)


# Method 3: Global Fit
def method3_global_fit(data):
    prepared = {}
    for conc_mM in sorted(data.keys()):
        time, trials = data[conc_mM]
        t_base, _ = prepare_data(time, trials[0])
        y_stack = []
        for trial in trials:
            t, y = prepare_data(time, trial)
            if len(t) == len(t_base):
                y_stack.append(y)
        if y_stack:
            prepared[conc_mM / 1000] = (t_base, np.nanmean(y_stack, axis=0))

    if len(prepared) < 2:
        return (np.nan,) * 5

    concs = sorted(prepared.keys())

    def global_objective(params):
        k_cat = params[0]
        total_ssr = 0.0
        idx = 1
        for conc_M in concs:
            t, y = prepared[conc_M]
            A0, A = params[idx], params[idx + 1]
            idx += 2
            k_obs = K_UNCAT_FIXED + k_cat * conc_M
            pred = A0 + A * np.exp(-k_obs * t)
            total_ssr += np.sum((y - pred)**2)
        return total_ssr

    p0 = [200.0]
    bounds = [(10, 800)]
    for conc_M in concs:
        t, y = prepared[conc_M]
        y_end = np.median(y[-50:])
        y_start = np.median(y[:20])
        p0.extend([y_end, y_start - y_end])
        bounds.extend([(-0.5, 0.5), (-0.5, 0.5)])

    try:
        result = minimize(global_objective, p0, method='L-BFGS-B', bounds=bounds)
        if not result.success:
            return (np.nan,) * 5

        k_cat = result.x[0]
        eps = 0.5
        f0 = global_objective(result.x)
        x_plus = result.x.copy()
        x_plus[0] += eps
        x_minus = result.x.copy()
        x_minus[0] -= eps
        hess = (global_objective(x_plus) - 2*f0 + global_objective(x_minus)) / (eps**2)

        n_total = sum(len(prepared[c][0]) for c in concs)
        mse = result.fun / (n_total - len(result.x))
        k_cat_err = np.sqrt(2 * mse / max(hess, 1e-6))

        y_all = np.concatenate([prepared[c][1] for c in concs])
        ss_tot = np.sum((y_all - np.mean(y_all))**2)
        r_sq = 1 - result.fun / ss_tot

        return (k_cat, k_cat_err, K_UNCAT_FIXED, 0.0, r_sq)
    except:
        return (np.nan,) * 5


# Main
datasets = [
    ("10x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-07-01_10xNMeIm/MIm_10x.xlsx"),
    ("15x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-06-30_15x-37xNMeIm/2025-06-30_15xNMeIm/MIm_15x.xlsx"),
    ("25x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-06-26_25-50-75xNMeIm/2025-06-26_25xNMeIm/MIm_25x.xlsx"),
    ("37x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-06-30_15x-37xNMeIm/2025-06-30_37xNMeIm/MIm_37x.xlsx"),
    ("50x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-06-26_25-50-75xNMeIm/2025-06-26_50xNMeIm/MIm_50x.xlsx"),
    ("75x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-06-26_25-50-75xNMeIm/2025-06-26_75xNMeIm/MIm_75x.xlsx"),
    ("100x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-05-27_100xNMeIm/MIm_100x.xlsx"),
    ("175x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-05-27_175xNMeIm/MIm_175x.xlsx"),
    ("250x NMeIm", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Monodentate_Study/2025-05-27_250xNMeIm/MIm_250x.xlsx"),
]

all_results = []

for name, path in datasets:
    print(f"Processing: {name}")
    data = load_data(path)
    if not data:
        print(f"  ERROR: No data loaded!")
        continue

    print(f"  Concentrations: {sorted(data.keys())} mM")

    m0 = method0_single_exp(data)
    m1 = method1_double_exp_weighted(data)
    m2 = method2_initial_rate(data)
    m3 = method3_global_fit(data)

    all_results.append({
        'name': name,
        'single_exp': {'k_cat': m0[0], 'k_cat_err': m0[1], 'k_uncat': m0[2], 'R2': m0[4]},
        'double_exp': {'k_cat': m1[0], 'k_cat_err': m1[1], 'k_uncat': m1[2], 'R2': m1[4]},
        'initial_rate': {'k_cat': m2[0], 'k_cat_err': m2[1], 'k_uncat': m2[2], 'R2': m2[4]},
        'global_fit': {'k_cat': m3[0], 'k_cat_err': m3[1], 'k_uncat': m3[2], 'R2': m3[4]},
    })

# Print results
print("\n" + "=" * 120)
print("MONODENTATE STUDY - COMPARISON OF FITTING METHODS")
print("=" * 120)

print("\n" + "-" * 120)
print("k_cat VALUES (/M/s)")
print("-" * 120)
print(f"{'Ligand Equiv':<15} | {'Single Exp':<20} | {'Double Exp (weighted)':<22} | {'Initial Rate':<20} | {'Global Fit':<20}")
print("-" * 120)

for r in all_results:
    def fmt(d):
        if np.isnan(d['k_cat']):
            return "N/A"
        return f"{d['k_cat']:.1f} ± {d['k_cat_err']:.1f}"

    print(f"{r['name']:<15} | {fmt(r['single_exp']):<20} | {fmt(r['double_exp']):<22} | {fmt(r['initial_rate']):<20} | {fmt(r['global_fit']):<20}")

print("\n" + "-" * 120)
print("k_uncat VALUES (/s) - Expected: ~0.12")
print("-" * 120)
print(f"{'Ligand Equiv':<15} | {'Single Exp':<20} | {'Double Exp (weighted)':<22} | {'Initial Rate':<20} | {'Global Fit':<20}")
print("-" * 120)

for r in all_results:
    def fmt(d):
        if np.isnan(d['k_uncat']):
            return "N/A"
        return f"{d['k_uncat']:.4f}"

    print(f"{r['name']:<15} | {fmt(r['single_exp']):<20} | {fmt(r['double_exp']):<22} | {fmt(r['initial_rate']):<20} | {fmt(r['global_fit']):<20}")

print("\n" + "-" * 120)
print("R² VALUES")
print("-" * 120)
print(f"{'Ligand Equiv':<15} | {'Single Exp':<20} | {'Double Exp (weighted)':<22} | {'Initial Rate':<20} | {'Global Fit':<20}")
print("-" * 120)

for r in all_results:
    def fmt(d):
        if np.isnan(d['R2']):
            return "N/A"
        return f"{d['R2']:.4f}"

    print(f"{r['name']:<15} | {fmt(r['single_exp']):<20} | {fmt(r['double_exp']):<22} | {fmt(r['initial_rate']):<20} | {fmt(r['global_fit']):<20}")

print("=" * 120)
