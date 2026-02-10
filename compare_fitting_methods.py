#!/usr/bin/env python3
"""
Three-Method Kinetics Fitting Comparison
=========================================
Compares fitting approaches for stopped-flow CO2 hydration kinetics:
0. Single Exponential (original method)
1. Constrained Double Exponential (k_slow fixed to 0.12/s)
2. Initial Rate Method (polynomial fit to first 10% of reaction)
3. Global Fitting (shared k_cat across all concentrations)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────── Constants ─────────────────────────── #
K_UNCAT_FIXED = 0.12  # Fixed uncatalyzed rate constant (/s)
T_MIN = 0.02
T_MAX = 40.0


# ─────────────────────────── Data Loading ─────────────────────────── #

def extract_concentration(s: str) -> Optional[float]:
    """Extract concentration in mM from string."""
    import re
    s = str(s).lower().replace(' ', '')
    patterns = [
        r'(\d+)p(\d+)mm',
        r'(\d+\.?\d*)mm',
        r'concentration\s*=?\s*(\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return float(f"{groups[0]}.{groups[1]}")
            return float(groups[0])
    return None


def load_multi_sheet_data(file_path: str) -> Dict[float, Tuple[np.ndarray, List[np.ndarray]]]:
    """Load data from multi-sheet Excel (one sheet per concentration)."""
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


def load_benzyl_sheet(file_path: str, sheet_name: str) -> Dict[float, Tuple[np.ndarray, List[np.ndarray]]]:
    """Load data from Benzyl-Cyclen format (column groups by concentration)."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    data = {}

    row0 = df.iloc[0].values
    conc_cols = []
    for i, val in enumerate(row0):
        if isinstance(val, str) and 'concentration' in val.lower():
            conc = extract_concentration(val)
            conc_cols.append((i, conc))

    for idx, (col_idx, conc) in enumerate(conc_cols):
        next_idx = conc_cols[idx + 1][0] if idx + 1 < len(conc_cols) else df.shape[1]

        time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

        trials = []
        for col in range(col_idx + 1, next_idx):
            trial = pd.to_numeric(df.iloc[2:, col], errors='coerce').values
            if np.isnan(trial).sum() < len(trial) * 0.5:
                trials.append(trial)

        if trials:
            data[conc] = (time, trials)

    return data


def prepare_data(time: np.ndarray, absorbance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply time window and remove NaN."""
    mask = (time >= T_MIN) & (time <= T_MAX)
    t = time[mask]
    y = absorbance[mask]
    valid = ~np.isnan(t) & ~np.isnan(y)
    return t[valid], y[valid]


# ─────────────────────────── Method 0: Single Exponential ─────────────────────────── #

def method0_single_exp(data: Dict[float, Tuple[np.ndarray, List[np.ndarray]]]) -> Tuple[float, float, float, float, float]:
    """Original method: Single exponential A(t) = A0 + A*exp(-k*t)."""
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


# ─────────────────────────── Method 1: Double Exp with Amplitude-Weighted Rate ─────────────────────────── #

def method1_double_exp_weighted(data: Dict[float, Tuple[np.ndarray, List[np.ndarray]]]) -> Tuple[float, float, float, float, float]:
    """
    Double exponential with amplitude-weighted average rate.
    Model: A(t) = A0 + A1*exp(-k1*t) + A2*exp(-k2*t)

    k_obs = (A1*k1 + A2*k2) / (A1 + A2)

    This gives an effective observed rate that accounts for both components.
    """
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

            # Full double exponential (both rates free)
            def objective(params):
                A0, A1, k1, A2, k2 = params
                pred = A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)
                return np.sum((y - pred)**2)

            best_result = None
            best_cost = np.inf

            # Multiple starting points
            for k1_init in [0.5, 1.0, 2.0]:
                for k2_init in [0.08, 0.12, 0.2]:
                    if k1_init <= k2_init:
                        continue
                    for frac in [0.5, 0.7]:
                        p0 = [y_end, total_amp * frac, k1_init, total_amp * (1 - frac), k2_init]
                        bounds = [
                            (y_end - 0.05, y_end + 0.05),
                            (0.001, total_amp * 2),
                            (0.2, 15.0),  # k1 (fast)
                            (0.001, total_amp * 2),
                            (0.01, 0.5)   # k2 (slow)
                        ]
                        try:
                            res = minimize(objective, p0, method='L-BFGS-B', bounds=bounds,
                                          options={'maxiter': 5000})
                            if res.success and res.fun < best_cost:
                                best_cost = res.fun
                                best_result = res.x
                        except:
                            pass

            if best_result is not None:
                A0, A1, k1, A2, k2 = best_result
                # Amplitude-weighted rate
                if abs(A1) + abs(A2) > 0.001:
                    k_weighted = (abs(A1) * k1 + abs(A2) * k2) / (abs(A1) + abs(A2))
                    if 0.1 < k_weighted < 10:
                        k_vals.append(k_weighted)

        if k_vals:
            concs_M.append(conc_mM / 1000)
            k_obs_values.append(np.mean(k_vals))

    return _linear_regression(concs_M, k_obs_values)


# ─────────────────────────── Method 2: Initial Rate ─────────────────────────── #

def method2_initial_rate(data: Dict[float, Tuple[np.ndarray, List[np.ndarray]]]) -> Tuple[float, float, float, float, float]:
    """
    Initial rate method: fit polynomial to first 10% of reaction.
    Extract dA/dt at t=0, convert to k_obs.
    """
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

            # Find time when 10% of reaction is complete
            threshold = y_start - 0.10 * delta_A
            idx = np.argmax(y < threshold) if delta_A > 0 else np.argmax(y > threshold)
            if idx < 20:
                idx = min(100, len(t) // 5)

            t_early = t[:idx] - t[0]  # Shift to start at 0
            y_early = y[:idx]

            if len(t_early) < 15:
                continue

            try:
                # Quadratic fit: A(t) = a + b*t + c*t²
                coeffs = np.polyfit(t_early, y_early, 2)
                dydt_0 = coeffs[1]  # dA/dt at t=0

                # For exponential decay: dA/dt|_{t=0} = -A*k
                # k = -dA/dt / A = -dydt_0 / delta_A
                k_obs = abs(dydt_0 / delta_A)

                if 0.05 < k_obs < 15:
                    k_vals.append(k_obs)
            except:
                pass

        if k_vals:
            concs_M.append(conc_mM / 1000)
            k_obs_values.append(np.mean(k_vals))

    return _linear_regression(concs_M, k_obs_values)


# ─────────────────────────── Method 3: Global Fit ─────────────────────────── #

def method3_global_fit(data: Dict[float, Tuple[np.ndarray, List[np.ndarray]]]) -> Tuple[float, float, float, float, float]:
    """
    Global fitting: all traces fitted simultaneously.
    Enforces k_obs = k_uncat + k_cat * [catalyst] directly.
    k_uncat fixed to 0.12/s.
    """
    # Average trials per concentration
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

    # Parameters: [k_cat, A0_1, A_1, A0_2, A_2, ...]
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

    # Initial guess
    p0 = [200.0]  # k_cat
    bounds = [(10, 800)]

    for conc_M in concs:
        t, y = prepared[conc_M]
        y_end = np.median(y[-50:])
        y_start = np.median(y[:20])
        p0.extend([y_end, y_start - y_end])
        bounds.extend([(-0.5, 0.5), (-0.5, 0.5)])

    try:
        result = minimize(global_objective, p0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 10000})

        if not result.success:
            return (np.nan,) * 5

        k_cat = result.x[0]

        # Estimate uncertainty via finite difference Hessian
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

        # Global R²
        y_all = np.concatenate([prepared[c][1] for c in concs])
        ss_tot = np.sum((y_all - np.mean(y_all))**2)
        r_sq = 1 - result.fun / ss_tot

        return (k_cat, k_cat_err, K_UNCAT_FIXED, 0.0, r_sq)
    except:
        return (np.nan,) * 5


# ─────────────────────────── Helper ─────────────────────────── #

def _linear_regression(concs_M: List[float], k_obs: List[float]) -> Tuple[float, float, float, float, float]:
    """Perform linear regression and return (k_cat, k_cat_err, k_uncat, k_uncat_err, R²)."""
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


# ─────────────────────────── Main ─────────────────────────── #

def process_dataset(name: str, file_path: str, sheet: str = None) -> Dict:
    """Process a single dataset with all methods."""
    print(f"\nProcessing: {name}")

    if sheet:
        data = load_benzyl_sheet(file_path, sheet)
    else:
        data = load_multi_sheet_data(file_path)

    if not data:
        print(f"  ERROR: No data loaded!")
        return None

    print(f"  Concentrations: {sorted(data.keys())} mM")

    results = {'name': name}

    # Method 0: Single exponential
    m0 = method0_single_exp(data)
    results['single_exp'] = {'k_cat': m0[0], 'k_cat_err': m0[1], 'k_uncat': m0[2], 'k_uncat_err': m0[3], 'R2': m0[4]}

    # Method 1: Double exponential with amplitude-weighted rate
    m1 = method1_double_exp_weighted(data)
    results['double_exp'] = {'k_cat': m1[0], 'k_cat_err': m1[1], 'k_uncat': m1[2], 'k_uncat_err': m1[3], 'R2': m1[4]}

    # Method 2: Initial rate
    m2 = method2_initial_rate(data)
    results['initial_rate'] = {'k_cat': m2[0], 'k_cat_err': m2[1], 'k_uncat': m2[2], 'k_uncat_err': m2[3], 'R2': m2[4]}

    # Method 3: Global fit
    m3 = method3_global_fit(data)
    results['global_fit'] = {'k_cat': m3[0], 'k_cat_err': m3[1], 'k_uncat': m3[2], 'k_uncat_err': m3[3], 'R2': m3[4]}

    return results


def print_comparison_table(all_results: List[Dict]):
    """Print comparison table."""
    print("\n" + "=" * 110)
    print("COMPARISON OF FITTING METHODS")
    print("=" * 110)

    print("\n" + "-" * 110)
    print("k_cat VALUES (/M/s)")
    print("-" * 110)
    print(f"{'Ligand':<22} | {'Single Exp':<18} | {'Double Exp (k₂=0.12)':<20} | {'Initial Rate':<18} | {'Global Fit':<18}")
    print("-" * 110)

    for r in all_results:
        if r is None:
            continue
        name = r['name']

        def fmt(d):
            if np.isnan(d['k_cat']):
                return "N/A"
            return f"{d['k_cat']:.1f} ± {d['k_cat_err']:.1f}"

        print(f"{name:<22} | {fmt(r['single_exp']):<18} | {fmt(r['double_exp']):<20} | {fmt(r['initial_rate']):<18} | {fmt(r['global_fit']):<18}")

    print("\n" + "-" * 110)
    print("k_uncat VALUES (/s) - Expected: ~0.12")
    print("-" * 110)
    print(f"{'Ligand':<22} | {'Single Exp':<18} | {'Double Exp (k₂=0.12)':<20} | {'Initial Rate':<18} | {'Global Fit':<18}")
    print("-" * 110)

    for r in all_results:
        if r is None:
            continue
        name = r['name']

        def fmt(d):
            if np.isnan(d['k_uncat']):
                return "N/A"
            return f"{d['k_uncat']:.4f}"

        print(f"{name:<22} | {fmt(r['single_exp']):<18} | {fmt(r['double_exp']):<20} | {fmt(r['initial_rate']):<18} | {fmt(r['global_fit']):<18}")

    print("\n" + "-" * 110)
    print("R² VALUES")
    print("-" * 110)
    print(f"{'Ligand':<22} | {'Single Exp':<18} | {'Double Exp (k₂=0.12)':<20} | {'Initial Rate':<18} | {'Global Fit':<18}")
    print("-" * 110)

    for r in all_results:
        if r is None:
            continue

        def fmt(d):
            if np.isnan(d['R2']):
                return "N/A"
            return f"{d['R2']:.4f}"

        print(f"{r['name']:<22} | {fmt(r['single_exp']):<18} | {fmt(r['double_exp']):<20} | {fmt(r['initial_rate']):<18} | {fmt(r['global_fit']):<18}")

    print("=" * 110)


if __name__ == "__main__":
    datasets = [
        ("Cyclen", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Cyclen_baseline/Cyclen_01_13.xlsx", None),
        ("nBu-Cyclen", "/Users/thomasquaid/Downloads/2026-02-02_nBuCyclen/nBu_cyc_02_02.xlsx", None),
        ("Hexyl-Cyclen", "/Users/thomasquaid/Downloads/2026-02-03_hexylcyclen/Hexyl_cyc_02_03.xlsx", None),
        ("Benzyl-Cyclen (12/05)", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx", "12_05"),
        ("Benzyl-Cyclen (12/13)", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx", "12_13"),
        ("Benzyl-Cyclen (01/14)", "/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx", "01_14"),
    ]

    all_results = []
    for name, path, sheet in datasets:
        result = process_dataset(name, path, sheet)
        all_results.append(result)

    print_comparison_table(all_results)
