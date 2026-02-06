#!/usr/bin/env python3
"""
Sensitivity Analysis for Stopped-Flow Kinetics Data
====================================================
Systematically varies analysis parameters to identify sources of precision issues.

Tests:
1. Time window sensitivity (t_min, t_max)
2. Model choice (single vs double exponential)
3. Initial parameter estimation strategies
4. Outlier handling
5. Individual trial analysis to identify problematic data
"""

from __future__ import annotations
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import linregress, zscore
from scipy.signal import savgol_filter

# Import from main analysis module
from rate_constant_analysis import (
    AnalysisConfig, load_catalyzed_sheet, load_uncatalyzed_sheet,
    single_exp, double_exp, calc_dydt0_single, calc_dydt0_double,
    extract_concentration_mM
)


# ─────────────────────────── Alternative Fitting Strategies ─────────────────────────── #

def fit_with_global_optimization(time: np.ndarray, y: np.ndarray, model: str,
                                  t_min: float, t_max: float) -> Tuple[float, bool]:
    """
    Use differential evolution (global optimizer) to avoid local minima.
    """
    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    y_fit = y[mask]

    valid = ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 10:
        return np.nan, False

    y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
    y_end = np.median(y_fit[-max(1, len(y_fit)//20):])

    def objective(params):
        if model == "single":
            pred = single_exp(t_fit, *params)
        else:
            pred = double_exp(t_fit, *params)
        return np.sum((y_fit - pred)**2)

    try:
        if model == "single":
            bounds = [(y_end - 0.1, y_end + 0.1),  # A0
                      (y_start - y_end - 0.1, y_start - y_end + 0.1),  # A
                      (0.001, 50)]  # k
            result = differential_evolution(objective, bounds, seed=42,
                                            maxiter=500, tol=1e-8)
            if result.success:
                dydt0 = calc_dydt0_single(result.x[1], result.x[2])
                return dydt0, True
        else:
            total_amp = y_start - y_end
            bounds = [(y_end - 0.1, y_end + 0.1),  # A0
                      (total_amp * 0.1, total_amp * 0.9),  # A1
                      (0.01, 50),  # k1
                      (total_amp * 0.1, total_amp * 0.9),  # A2
                      (0.001, 5)]  # k2
            result = differential_evolution(objective, bounds, seed=42,
                                            maxiter=1000, tol=1e-8)
            if result.success:
                dydt0 = calc_dydt0_double(result.x[1], result.x[2],
                                          result.x[3], result.x[4])
                return dydt0, True

    except Exception:
        pass

    return np.nan, False


def fit_with_numerical_derivative(time: np.ndarray, y: np.ndarray,
                                   t_min: float, t_max: float,
                                   smoothing: bool = True) -> Tuple[float, bool]:
    """
    Extract initial rate directly from numerical derivative instead of model fitting.
    """
    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    y_fit = y[mask]

    valid = ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 20:
        return np.nan, False

    if smoothing:
        # Savitzky-Golay filter for smoothing
        window = min(51, len(y_fit) // 4 * 2 + 1)  # Must be odd
        if window >= 5:
            y_smooth = savgol_filter(y_fit, window, 3)
        else:
            y_smooth = y_fit
    else:
        y_smooth = y_fit

    # Compute derivative
    dy = np.gradient(y_smooth, t_fit)

    # Take average of first few points as initial rate
    n_early = max(1, len(dy) // 20)
    dydt0 = np.mean(dy[:n_early])

    return dydt0, True


def fit_linear_early_region(time: np.ndarray, y: np.ndarray,
                             t_min: float, t_linear_end: float) -> Tuple[float, bool]:
    """
    Fit only the early linear region to extract initial rate.
    For reactions that haven't progressed far, the decay is approximately linear.
    """
    mask = (time >= t_min) & (time <= t_linear_end)
    t_fit = time[mask]
    y_fit = y[mask]

    valid = ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 5:
        return np.nan, False

    slope, intercept, r_value, p_value, std_err = linregress(t_fit, y_fit)

    return slope, True


def fit_constrained_double_exp(time: np.ndarray, y: np.ndarray,
                                t_min: float, t_max: float,
                                enforce_k_order: bool = True) -> Tuple[np.ndarray, float, bool]:
    """
    Fit double exponential with constraint k1 > k2 to break symmetry.
    """
    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    y_fit = y[mask]

    valid = ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 10:
        return np.array([]), np.nan, False

    y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
    y_end = np.median(y_fit[-max(1, len(y_fit)//20):])
    total_amp = y_start - y_end

    def model_constrained(t, A0, A1, log_k1, A2, log_k_ratio):
        # k1 = exp(log_k1), k2 = k1 * exp(-log_k_ratio) = k1 / exp(log_k_ratio)
        # This ensures k1 > k2 when log_k_ratio > 0
        k1 = np.exp(log_k1)
        k2 = k1 * np.exp(-log_k_ratio)
        return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)

    try:
        # Initial guesses in transformed space
        p0 = [y_end, total_amp * 0.5, np.log(1.0), total_amp * 0.5, 1.0]
        bounds = ([-np.inf, -np.inf, np.log(0.01), -np.inf, 0],
                  [np.inf, np.inf, np.log(100), np.inf, 10])

        popt, pcov = curve_fit(model_constrained, t_fit, y_fit, p0=p0,
                               bounds=bounds, maxfev=50000)

        # Convert back to standard parameters
        k1 = np.exp(popt[2])
        k2 = k1 * np.exp(-popt[4])
        dydt0 = calc_dydt0_double(popt[1], k1, popt[3], k2)

        return np.array([popt[0], popt[1], k1, popt[3], k2]), dydt0, True

    except (RuntimeError, ValueError):
        return np.array([]), np.nan, False


# ─────────────────────────── Sensitivity Tests ─────────────────────────── #

def test_time_window_sensitivity(xl: pd.ExcelFile, sheet_name: str,
                                  model: str = "double") -> pd.DataFrame:
    """
    Vary t_min and t_max to see how sensitive results are to time window choice.
    """
    data = load_catalyzed_sheet(xl, sheet_name)

    t_mins = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    t_maxs = [20, 30, 40, 50, 60]

    results = []

    for conc_label, (time, trials) in data.items():
        conc_mM = extract_concentration_mM(conc_label)

        for t_min in t_mins:
            for t_max in t_maxs:
                if t_min >= t_max:
                    continue

                dydt0_values = []
                for trial_data in trials:
                    mask = (time >= t_min) & (time <= t_max)
                    t_fit = time[mask]
                    y_fit = trial_data[mask]

                    valid = ~np.isnan(y_fit)
                    if valid.sum() < 10:
                        continue

                    t_fit = t_fit[valid]
                    y_fit = y_fit[valid]

                    try:
                        if model == "single":
                            y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
                            y_end = np.median(y_fit[-max(1, len(y_fit)//20):])
                            p0 = [y_end, y_start - y_end, 0.1]
                            bounds = ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 100])
                            popt, _ = curve_fit(single_exp, t_fit, y_fit, p0=p0,
                                                bounds=bounds, maxfev=20000)
                            dydt0 = calc_dydt0_single(popt[1], popt[2])
                        else:
                            y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
                            y_end = np.median(y_fit[-max(1, len(y_fit)//20):])
                            total_amp = y_start - y_end
                            p0 = [y_end, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
                            bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                                      [np.inf, np.inf, 100, np.inf, 100])
                            popt, _ = curve_fit(double_exp, t_fit, y_fit, p0=p0,
                                                bounds=bounds, maxfev=20000)
                            dydt0 = calc_dydt0_double(popt[1], popt[2], popt[3], popt[4])

                        dydt0_values.append(dydt0)
                    except:
                        continue

                if dydt0_values:
                    results.append({
                        'sheet': sheet_name,
                        'concentration_mM': conc_mM,
                        't_min': t_min,
                        't_max': t_max,
                        'mean_dydt0': np.mean(dydt0_values),
                        'std_dydt0': np.std(dydt0_values, ddof=1) if len(dydt0_values) > 1 else 0,
                        'n_trials': len(dydt0_values)
                    })

    return pd.DataFrame(results)


def test_model_comparison(xl: pd.ExcelFile, config: AnalysisConfig) -> pd.DataFrame:
    """
    Compare single vs double exponential fits.
    """
    results = []

    for sheet_name in ['12_05', '12_13', '01_14']:
        data = load_catalyzed_sheet(xl, sheet_name)

        for conc_label, (time, trials) in data.items():
            conc_mM = extract_concentration_mM(conc_label)

            for trial_idx, trial_data in enumerate(trials):
                mask = (time >= config.t_min) & (time <= config.t_max)
                t_fit = time[mask]
                y_fit = trial_data[mask]

                valid = ~np.isnan(y_fit)
                if valid.sum() < 10:
                    continue

                t_fit = t_fit[valid]
                y_fit = y_fit[valid]

                y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
                y_end = np.median(y_fit[-max(1, len(y_fit)//20):])

                result = {
                    'sheet': sheet_name,
                    'concentration_mM': conc_mM,
                    'trial': trial_idx
                }

                # Single exponential
                try:
                    p0 = [y_end, y_start - y_end, 0.1]
                    bounds = ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 100])
                    popt_s, pcov_s = curve_fit(single_exp, t_fit, y_fit, p0=p0,
                                               bounds=bounds, maxfev=20000)
                    dydt0_s = calc_dydt0_single(popt_s[1], popt_s[2])
                    resid_s = y_fit - single_exp(t_fit, *popt_s)
                    ss_res_s = np.sum(resid_s**2)
                    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
                    r2_s = 1 - ss_res_s / ss_tot

                    result['dydt0_single'] = dydt0_s
                    result['k_single'] = popt_s[2]
                    result['k_single_err'] = np.sqrt(pcov_s[2, 2])
                    result['r2_single'] = r2_s
                except:
                    result['dydt0_single'] = np.nan
                    result['k_single'] = np.nan
                    result['k_single_err'] = np.nan
                    result['r2_single'] = np.nan

                # Double exponential
                try:
                    total_amp = y_start - y_end
                    p0 = [y_end, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
                    bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                              [np.inf, np.inf, 100, np.inf, 100])
                    popt_d, pcov_d = curve_fit(double_exp, t_fit, y_fit, p0=p0,
                                               bounds=bounds, maxfev=20000)
                    dydt0_d = calc_dydt0_double(popt_d[1], popt_d[2], popt_d[3], popt_d[4])
                    resid_d = y_fit - double_exp(t_fit, *popt_d)
                    ss_res_d = np.sum(resid_d**2)
                    r2_d = 1 - ss_res_d / ss_tot

                    result['dydt0_double'] = dydt0_d
                    result['k1_double'] = popt_d[2]
                    result['k2_double'] = popt_d[4]
                    result['k1_err'] = np.sqrt(pcov_d[2, 2])
                    result['k2_err'] = np.sqrt(pcov_d[4, 4])
                    result['r2_double'] = r2_d

                    # Check if k1 ≈ k2 (double exp may be overparameterized)
                    result['k_ratio'] = popt_d[2] / popt_d[4] if popt_d[4] > 0 else np.inf
                except:
                    result['dydt0_double'] = np.nan
                    result['k1_double'] = np.nan
                    result['k2_double'] = np.nan
                    result['k1_err'] = np.nan
                    result['k2_err'] = np.nan
                    result['r2_double'] = np.nan
                    result['k_ratio'] = np.nan

                results.append(result)

    return pd.DataFrame(results)


def test_fitting_methods(xl: pd.ExcelFile, config: AnalysisConfig) -> pd.DataFrame:
    """
    Compare different fitting methods on the same data.
    """
    results = []

    for sheet_name in ['12_05', '12_13', '01_14']:
        data = load_catalyzed_sheet(xl, sheet_name)

        for conc_label, (time, trials) in data.items():
            conc_mM = extract_concentration_mM(conc_label)

            for trial_idx, trial_data in enumerate(trials):
                result = {
                    'sheet': sheet_name,
                    'concentration_mM': conc_mM,
                    'trial': trial_idx
                }

                # Method 1: Standard curve_fit (single)
                mask = (time >= config.t_min) & (time <= config.t_max)
                t_fit = time[mask]
                y_fit = trial_data[mask]
                valid = ~np.isnan(y_fit)
                if valid.sum() < 10:
                    continue
                t_fit = t_fit[valid]
                y_fit = y_fit[valid]

                y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
                y_end = np.median(y_fit[-max(1, len(y_fit)//20):])

                try:
                    p0 = [y_end, y_start - y_end, 0.1]
                    bounds = ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 100])
                    popt, _ = curve_fit(single_exp, t_fit, y_fit, p0=p0,
                                        bounds=bounds, maxfev=20000)
                    result['curvefit_single'] = calc_dydt0_single(popt[1], popt[2])
                except:
                    result['curvefit_single'] = np.nan

                # Method 2: Global optimization
                dydt0, success = fit_with_global_optimization(
                    time, trial_data, "single", config.t_min, config.t_max
                )
                result['global_opt_single'] = dydt0 if success else np.nan

                # Method 3: Numerical derivative
                dydt0, success = fit_with_numerical_derivative(
                    time, trial_data, config.t_min, config.t_max, smoothing=True
                )
                result['numerical_deriv'] = dydt0 if success else np.nan

                # Method 4: Linear fit to early region
                dydt0, success = fit_linear_early_region(
                    time, trial_data, config.t_min, config.t_min + 1.0
                )
                result['linear_early'] = dydt0 if success else np.nan

                # Method 5: Constrained double exp
                _, dydt0, success = fit_constrained_double_exp(
                    time, trial_data, config.t_min, config.t_max
                )
                result['constrained_double'] = dydt0 if success else np.nan

                results.append(result)

    return pd.DataFrame(results)


def analyze_trial_variability(xl: pd.ExcelFile, config: AnalysisConfig) -> pd.DataFrame:
    """
    Analyze individual trial results to identify outliers and problematic data.
    """
    results = []

    for sheet_name in ['12_05', '12_13', '01_14']:
        data = load_catalyzed_sheet(xl, sheet_name)

        for conc_label, (time, trials) in data.items():
            conc_mM = extract_concentration_mM(conc_label)

            for trial_idx, trial_data in enumerate(trials):
                mask = (time >= config.t_min) & (time <= config.t_max)
                t_fit = time[mask]
                y_fit = trial_data[mask]

                valid = ~np.isnan(y_fit)
                if valid.sum() < 10:
                    continue
                t_fit = t_fit[valid]
                y_fit = y_fit[valid]

                y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
                y_end = np.median(y_fit[-max(1, len(y_fit)//20):])

                result = {
                    'sheet': sheet_name,
                    'concentration_mM': conc_mM,
                    'trial': trial_idx,
                    'y_start': y_start,
                    'y_end': y_end,
                    'amplitude': y_start - y_end,
                    'baseline_offset': y_end
                }

                try:
                    total_amp = y_start - y_end
                    p0 = [y_end, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
                    bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                              [np.inf, np.inf, 100, np.inf, 100])
                    popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=p0,
                                           bounds=bounds, maxfev=20000)

                    result['A0'] = popt[0]
                    result['A1'] = popt[1]
                    result['k1'] = popt[2]
                    result['A2'] = popt[3]
                    result['k2'] = popt[4]
                    result['dydt0'] = calc_dydt0_double(popt[1], popt[2], popt[3], popt[4])

                    # Parameter errors
                    perr = np.sqrt(np.diag(pcov))
                    result['k1_err'] = perr[2]
                    result['k2_err'] = perr[4]

                    # Residual analysis
                    resid = y_fit - double_exp(t_fit, *popt)
                    result['residual_rms'] = np.sqrt(np.mean(resid**2))
                    result['residual_max'] = np.max(np.abs(resid))

                    # R-squared
                    ss_res = np.sum(resid**2)
                    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
                    result['r2'] = 1 - ss_res / ss_tot

                except Exception as e:
                    result['dydt0'] = np.nan
                    result['r2'] = np.nan

                results.append(result)

    return pd.DataFrame(results)


# ─────────────────────────── Main Analysis ─────────────────────────── #

def run_sensitivity_analysis(file_path: pathlib.Path) -> None:
    """Run comprehensive sensitivity analysis."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)

    xl = pd.ExcelFile(file_path)
    config = AnalysisConfig()

    output_dir = file_path.parent / "sensitivity_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Model comparison
    print("\n[1] Comparing single vs double exponential models...")
    model_df = test_model_comparison(xl, config)
    model_df.to_csv(output_dir / "model_comparison.csv", index=False)

    print("\n  Single exponential summary:")
    print(f"    Mean R²: {model_df['r2_single'].mean():.4f}")
    print(f"    Mean |dA/dt|_0: {model_df['dydt0_single'].abs().mean():.6f} AU/s")

    print("\n  Double exponential summary:")
    print(f"    Mean R²: {model_df['r2_double'].mean():.4f}")
    print(f"    Mean |dA/dt|_0: {model_df['dydt0_double'].abs().mean():.6f} AU/s")

    # Check k1/k2 ratio
    k_ratios = model_df['k_ratio'].dropna()
    print(f"\n  k1/k2 ratio distribution:")
    print(f"    Median: {k_ratios.median():.2f}")
    print(f"    Mean: {k_ratios.mean():.2f}")
    print(f"    Range: {k_ratios.min():.2f} - {k_ratios.max():.2f}")

    # Test 2: Fitting method comparison
    print("\n[2] Comparing different fitting methods...")
    methods_df = test_fitting_methods(xl, config)
    methods_df.to_csv(output_dir / "fitting_methods.csv", index=False)

    method_cols = ['curvefit_single', 'global_opt_single', 'numerical_deriv',
                   'linear_early', 'constrained_double']

    print("\n  Method comparison (mean |dA/dt|_0 in AU/s):")
    for col in method_cols:
        mean_val = methods_df[col].abs().mean()
        std_val = methods_df[col].abs().std()
        print(f"    {col:25s}: {mean_val:.6f} ± {std_val:.6f}")

    # Test 3: Trial variability
    print("\n[3] Analyzing individual trial variability...")
    trial_df = analyze_trial_variability(xl, config)
    trial_df.to_csv(output_dir / "trial_variability.csv", index=False)

    # Calculate within-group (same conc, same date) vs between-group variability
    print("\n  Variability decomposition:")
    for sheet in ['12_05', '12_13', '01_14']:
        sheet_data = trial_df[trial_df['sheet'] == sheet]
        print(f"\n  {sheet}:")
        for conc in sheet_data['concentration_mM'].unique():
            conc_data = sheet_data[sheet_data['concentration_mM'] == conc]
            dydt0_values = conc_data['dydt0'].dropna()
            if len(dydt0_values) > 1:
                cv = 100 * dydt0_values.std() / dydt0_values.abs().mean()
                print(f"    {conc:.2f} mM: CV = {cv:.1f}% (n={len(dydt0_values)})")

    # Test 4: Time window sensitivity
    print("\n[4] Testing time window sensitivity...")
    window_results = []
    for sheet in ['12_05', '12_13', '01_14']:
        window_df = test_time_window_sensitivity(xl, sheet, model="double")
        window_results.append(window_df)
    all_windows = pd.concat(window_results, ignore_index=True)
    all_windows.to_csv(output_dir / "time_window_sensitivity.csv", index=False)

    # Summarize time window effects
    print("\n  Time window effects on k_cat:")
    for t_min in [0.01, 0.02, 0.05]:
        for t_max in [30, 40, 50]:
            subset = all_windows[(all_windows['t_min'] == t_min) &
                                 (all_windows['t_max'] == t_max)]
            if len(subset) > 0:
                # Compute k_cat for each sheet
                k_cats = []
                for sheet in ['12_05', '12_13', '01_14']:
                    sheet_sub = subset[subset['sheet'] == sheet]
                    if len(sheet_sub) >= 3:
                        concs = sheet_sub['concentration_mM'].values / 1000  # to M
                        rates = sheet_sub['mean_dydt0'].abs().values
                        if len(concs) >= 2:
                            slope, _, _, _, _ = linregress(concs, rates)
                            k_cats.append(slope)

                if k_cats:
                    k_mean = np.mean(k_cats)
                    k_std = np.std(k_cats, ddof=1) if len(k_cats) > 1 else 0
                    cv = 100 * k_std / abs(k_mean) if abs(k_mean) > 0 else np.nan
                    print(f"    t=[{t_min}, {t_max}]: k_cat = {k_mean:.1f} ± {k_std:.1f} (CV={cv:.1f}%)")

    # Test 5: Identify problematic trials
    print("\n[5] Identifying outlier trials...")

    # Calculate z-scores within each group
    outliers = []
    for sheet in ['12_05', '12_13', '01_14']:
        sheet_data = trial_df[trial_df['sheet'] == sheet]
        for conc in sheet_data['concentration_mM'].unique():
            conc_data = sheet_data[sheet_data['concentration_mM'] == conc].copy()
            dydt0_values = conc_data['dydt0'].dropna()
            if len(dydt0_values) > 2:
                z = zscore(dydt0_values)
                outlier_mask = np.abs(z) > 2
                for idx, (is_outlier, z_val) in enumerate(zip(outlier_mask, z)):
                    if is_outlier:
                        trial_info = conc_data.iloc[idx]
                        outliers.append({
                            'sheet': sheet,
                            'concentration_mM': conc,
                            'trial': trial_info['trial'],
                            'dydt0': trial_info['dydt0'],
                            'z_score': z_val,
                            'r2': trial_info['r2']
                        })

    if outliers:
        print(f"\n  Found {len(outliers)} potential outlier trials (|z| > 2):")
        for o in outliers:
            print(f"    {o['sheet']}, {o['concentration_mM']:.2f} mM, trial {o['trial']}: "
                  f"z={o['z_score']:.2f}, R²={o['r2']:.4f}")
    else:
        print("\n  No outlier trials identified.")

    # Create summary visualization
    print("\n[6] Creating diagnostic plots...")
    create_sensitivity_plots(trial_df, all_windows, output_dir)

    print(f"\nResults saved to: {output_dir}")


def create_sensitivity_plots(trial_df: pd.DataFrame, window_df: pd.DataFrame,
                              output_dir: pathlib.Path) -> None:
    """Create diagnostic visualizations."""

    # Plot 1: k1 vs k2 for double exponential
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for sheet in ['12_05', '12_13', '01_14']:
        subset = trial_df[trial_df['sheet'] == sheet]
        ax.scatter(subset['k1'], subset['k2'], label=sheet, alpha=0.7)
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='k1=k2')
    ax.set_xlabel('k1 (fast rate constant)')
    ax.set_ylabel('k2 (slow rate constant)')
    ax.set_title('Double Exponential Rate Constants')
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)

    # Plot 2: Initial rate by date and concentration
    ax = axes[1]
    width = 0.25
    x = np.arange(3)  # 3 concentrations
    sheets = ['12_05', '12_13', '01_14']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, sheet in enumerate(sheets):
        sheet_data = trial_df[trial_df['sheet'] == sheet]
        means = []
        stds = []
        for conc in [0.25, 0.5, 1.0]:
            conc_data = sheet_data[sheet_data['concentration_mM'] == conc]
            means.append(conc_data['dydt0'].abs().mean())
            stds.append(conc_data['dydt0'].abs().std())
        ax.bar(x + i*width, means, width, yerr=stds, label=sheet,
               color=colors[i], capsize=3)

    ax.set_xlabel('[Catalyst] (mM)')
    ax.set_ylabel('|dA/dt|₀ (AU/s)')
    ax.set_title('Initial Rates by Date')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['0.25', '0.5', '1.0'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_overview.png', dpi=300)
    plt.close()

    # Plot 3: Time window sensitivity heatmap
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, sheet in enumerate(['12_05', '12_13', '01_14']):
        sheet_data = window_df[window_df['sheet'] == sheet]
        # Aggregate across concentrations for each time window
        pivot_data = sheet_data.groupby(['t_min', 't_max'])['mean_dydt0'].mean().unstack()

        if not pivot_data.empty:
            im = axes[idx].imshow(pivot_data.abs().values, aspect='auto', cmap='viridis')
            axes[idx].set_xticks(range(len(pivot_data.columns)))
            axes[idx].set_xticklabels([f'{x:.0f}' for x in pivot_data.columns])
            axes[idx].set_yticks(range(len(pivot_data.index)))
            axes[idx].set_yticklabels([f'{x:.3f}' for x in pivot_data.index])
            axes[idx].set_xlabel('t_max (s)')
            axes[idx].set_ylabel('t_min (s)')
            axes[idx].set_title(f'{sheet}')
            plt.colorbar(im, ax=axes[idx], label='|dA/dt|₀')

    plt.tight_layout()
    plt.savefig(output_dir / 'time_window_heatmap.png', dpi=300)
    plt.close()

    print(f"  Saved: sensitivity_overview.png, time_window_heatmap.png")


if __name__ == "__main__":
    import sys
    file_path = pathlib.Path("Data/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    if len(sys.argv) > 1:
        file_path = pathlib.Path(sys.argv[1])
    run_sensitivity_analysis(file_path)
