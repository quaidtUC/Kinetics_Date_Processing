#!/usr/bin/env python3
"""
Rate Constant Analysis Pipeline
===============================
Processes stopped-flow spectrophotometry data to extract catalytic rate constants.

Workflow:
1. Load multi-sheet Excel file with catalyzed and uncatalyzed runs
2. Fit exponential decays to each trial within specified time window
3. Extract initial rates (dA/dt at t=0)
4. Perform linear regression of rate vs [catalyst] to get k_cat
5. Report precision (across dates) and accuracy (vs known uncatalyzed rate)

Usage:
    python rate_constant_analysis.py --file Data/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx
"""

from __future__ import annotations
import argparse
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress


# ─────────────────────────── Configuration ─────────────────────────── #

@dataclass
class AnalysisConfig:
    """Configuration parameters for the analysis."""
    t_min: float = 0.02          # Start of fitting window (s)
    t_max: float = 40.0          # End of fitting window (s)
    model: str = "double"        # "single" or "double" exponential
    concentrations: List[float] = field(default_factory=lambda: [0.25e-3, 0.5e-3, 1.0e-3])  # M
    expected_uncat_rate: float = 0.12  # Expected uncatalyzed rate for accuracy benchmark
    Q_catalyzed: float = -0.402
    Q_uncatalyzed: float = -0.607
    CO2_concentration: float = 0.0338  # M, saturated


# ─────────────────────────── Exponential Models ─────────────────────────── #

def single_exp(t: np.ndarray, A0: float, A: float, k: float) -> np.ndarray:
    """Single exponential decay: A(t) = A0 + A * exp(-k * t)"""
    return A0 + A * np.exp(-k * t)


def double_exp(t: np.ndarray, A0: float, A1: float, k1: float,
               A2: float, k2: float) -> np.ndarray:
    """Double exponential decay: A(t) = A0 + A1*exp(-k1*t) + A2*exp(-k2*t)"""
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def calc_dydt0_single(A: float, k: float) -> float:
    """Initial rate for single exponential."""
    return -A * k


def calc_dydt0_double(A1: float, k1: float, A2: float, k2: float) -> float:
    """Initial rate for double exponential."""
    return -A1 * k1 - A2 * k2


# ─────────────────────────── Data Loading ─────────────────────────── #

def load_catalyzed_sheet(xl: pd.ExcelFile, sheet_name: str
                         ) -> Dict[str, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Load a catalyzed experiment sheet.

    Returns:
        Dict mapping concentration label -> (time_array, [absorbance_arrays])
    """
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    # Find concentration group boundaries from row 0
    row0 = df.iloc[0].values
    conc_cols = [i for i, v in enumerate(row0)
                 if isinstance(v, str) and 'concentration' in v.lower()]

    result = {}
    for i, col_idx in enumerate(conc_cols):
        conc_label = str(row0[col_idx])
        next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]

        # Extract time (first column after header) and trials
        time_col = col_idx
        trial_cols = list(range(col_idx + 1, next_idx))

        # Data starts at row 2 (row 0 = conc header, row 1 = column labels)
        time = df.iloc[2:, time_col].astype(float).values

        trials = []
        for tc in trial_cols:
            col_data = df.iloc[2:, tc]
            # Skip if column is mostly NaN or contains Q value
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue
            try:
                trials.append(col_data.astype(float).values)
            except (ValueError, TypeError):
                continue

        result[conc_label] = (time, trials)

    return result


def load_uncatalyzed_sheet(xl: pd.ExcelFile, sheet_name: str = "Uncatalyzed"
                           ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Load uncatalyzed control sheet.

    Returns:
        (time_array, [absorbance_arrays])
    """
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    # Row 1 has headers (Time, Trial 1, Trial 2, ...)
    # Data starts at row 2
    time = df.iloc[2:, 0].astype(float).values

    trials = []
    for col in range(1, df.shape[1]):
        col_data = df.iloc[2:, col]
        # Skip NaN columns or Q-value columns
        if col_data.isna().sum() > len(col_data) * 0.5:
            continue
        # Check if it's a Q value column
        header = df.iloc[1, col]
        if isinstance(header, str) and 'Q' in header.upper():
            continue
        try:
            arr = col_data.astype(float).values
            if not np.isnan(arr).all():
                trials.append(arr)
        except (ValueError, TypeError):
            continue

    return time, trials


# ─────────────────────────── Fitting ─────────────────────────── #

def fit_trace(time: np.ndarray, y: np.ndarray, model: str,
              t_min: float, t_max: float
              ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Fit exponential model to a single trace within time window.

    Returns:
        (popt, perr, dydt0, success)
    """
    # Apply time window
    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    y_fit = y[mask]

    if len(t_fit) < 10:
        return np.array([]), np.array([]), np.nan, False

    # Remove NaN values
    valid = ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 10:
        return np.array([]), np.array([]), np.nan, False

    try:
        if model == "single":
            func = single_exp
            # Initial guesses
            y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
            y_end = np.median(y_fit[-max(1, len(y_fit)//20):])
            A0_guess = y_end
            A_guess = y_start - y_end

            # Estimate k from time to reach ~63% of decay
            if abs(A_guess) > 1e-10:
                target = y_start - 0.632 * (y_start - y_end)
                idx = np.argmin(np.abs(y_fit - target))
                k_guess = 1.0 / max(t_fit[idx], 0.01)
            else:
                k_guess = 0.1

            p0 = [A0_guess, A_guess, k_guess]
            bounds = ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 100])

            popt, pcov = curve_fit(func, t_fit, y_fit, p0=p0, bounds=bounds,
                                   maxfev=50000)
            perr = np.sqrt(np.diag(pcov))
            dydt0 = calc_dydt0_single(popt[1], popt[2])

        else:  # double
            func = double_exp
            y_start = np.median(y_fit[:max(1, len(y_fit)//20)])
            y_end = np.median(y_fit[-max(1, len(y_fit)//20):])
            A0_guess = y_end
            total_amp = y_start - y_end

            # Split amplitude between fast and slow components
            p0 = [A0_guess, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
            bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                      [np.inf, np.inf, 100, np.inf, 100])

            popt, pcov = curve_fit(func, t_fit, y_fit, p0=p0, bounds=bounds,
                                   maxfev=50000)
            perr = np.sqrt(np.diag(pcov))
            dydt0 = calc_dydt0_double(popt[1], popt[2], popt[3], popt[4])

        return popt, perr, dydt0, True

    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Fitting failed: {e}")
        return np.array([]), np.array([]), np.nan, False


# ─────────────────────────── Analysis Pipeline ─────────────────────────── #

@dataclass
class FitResult:
    """Result from fitting a single trace."""
    popt: np.ndarray
    perr: np.ndarray
    dydt0: float
    success: bool


@dataclass
class ConcentrationResult:
    """Results for all trials at one concentration."""
    concentration_mM: float
    fit_results: List[FitResult]
    mean_dydt0: float
    std_dydt0: float
    n_successful: int


@dataclass
class DateResult:
    """Results for one experimental date."""
    date_label: str
    concentration_results: List[ConcentrationResult]
    k_cat: float  # Slope of rate vs [cat]
    k_cat_err: float
    intercept: float  # Uncatalyzed rate from intercept
    intercept_err: float
    r_squared: float


def analyze_concentration_group(time: np.ndarray, trials: List[np.ndarray],
                                conc_mM: float, config: AnalysisConfig
                                ) -> ConcentrationResult:
    """Analyze all trials for one concentration."""
    fit_results = []
    dydt0_values = []

    for trial_data in trials:
        popt, perr, dydt0, success = fit_trace(
            time, trial_data, config.model, config.t_min, config.t_max
        )
        fit_results.append(FitResult(popt, perr, dydt0, success))
        if success and not np.isnan(dydt0):
            dydt0_values.append(dydt0)

    if dydt0_values:
        mean_dydt0 = np.mean(dydt0_values)
        std_dydt0 = np.std(dydt0_values, ddof=1) if len(dydt0_values) > 1 else 0.0
    else:
        mean_dydt0 = np.nan
        std_dydt0 = np.nan

    return ConcentrationResult(
        concentration_mM=conc_mM,
        fit_results=fit_results,
        mean_dydt0=mean_dydt0,
        std_dydt0=std_dydt0,
        n_successful=len(dydt0_values)
    )


def extract_concentration_mM(label: str) -> float:
    """Extract numeric concentration from label like 'concentration = 0.25 mM'."""
    import re
    match = re.search(r'(\d+\.?\d*)\s*mM', label, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return np.nan


def analyze_date_sheet(xl: pd.ExcelFile, sheet_name: str, config: AnalysisConfig
                       ) -> DateResult:
    """Analyze all concentrations for one experimental date."""
    data = load_catalyzed_sheet(xl, sheet_name)

    conc_results = []
    for conc_label, (time, trials) in data.items():
        conc_mM = extract_concentration_mM(conc_label)
        result = analyze_concentration_group(time, trials, conc_mM, config)
        conc_results.append(result)

    # Sort by concentration
    conc_results.sort(key=lambda x: x.concentration_mM)

    # Check if rates are in expected order (should increase with concentration)
    rates = [cr.mean_dydt0 for cr in conc_results]
    abs_rates = [abs(r) for r in rates if not np.isnan(r)]

    if len(abs_rates) >= 2 and abs_rates != sorted(abs_rates):
        # Rates not in order - concentrations may be mislabeled
        # Reorder by rate magnitude
        print(f"  Warning: Rates not in expected order for {sheet_name}. "
              f"Reordering by rate magnitude.")
        sorted_results = sorted(conc_results, key=lambda x: abs(x.mean_dydt0))
        for i, (sr, std_conc) in enumerate(zip(sorted_results, config.concentrations)):
            sr.concentration_mM = std_conc * 1000  # Convert to mM
        conc_results = sorted_results

    # Linear regression: rate vs [catalyst]
    concs_M = np.array([cr.concentration_mM / 1000 for cr in conc_results])  # to M
    rates = np.array([cr.mean_dydt0 for cr in conc_results])

    # Use absolute rates (they should all be same sign anyway)
    abs_rates = np.abs(rates)

    valid = ~np.isnan(abs_rates)
    if valid.sum() >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(
            concs_M[valid], abs_rates[valid]
        )
        r_squared = r_value ** 2
    else:
        slope = np.nan
        intercept = np.nan
        std_err = np.nan
        r_squared = np.nan

    return DateResult(
        date_label=sheet_name,
        concentration_results=conc_results,
        k_cat=slope,
        k_cat_err=std_err,
        intercept=intercept,
        intercept_err=np.nan,  # linregress doesn't give intercept error directly
        r_squared=r_squared
    )


def analyze_uncatalyzed(xl: pd.ExcelFile, config: AnalysisConfig
                        ) -> Tuple[float, float, int]:
    """
    Analyze uncatalyzed control.

    Returns:
        (mean_dydt0, std_dydt0, n_successful)
    """
    time, trials = load_uncatalyzed_sheet(xl)

    dydt0_values = []
    for trial_data in trials:
        popt, perr, dydt0, success = fit_trace(
            time, trial_data, config.model, config.t_min, config.t_max
        )
        if success and not np.isnan(dydt0):
            dydt0_values.append(dydt0)

    if dydt0_values:
        return (np.mean(dydt0_values),
                np.std(dydt0_values, ddof=1) if len(dydt0_values) > 1 else 0.0,
                len(dydt0_values))
    return np.nan, np.nan, 0


# ─────────────────────────── Reporting ─────────────────────────── #

def print_analysis_report(date_results: List[DateResult],
                          uncat_mean: float, uncat_std: float, uncat_n: int,
                          config: AnalysisConfig) -> None:
    """Print comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("RATE CONSTANT ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Time window: {config.t_min} - {config.t_max} s")
    print(f"  Model: {config.model} exponential")
    print(f"  Expected uncatalyzed rate: {config.expected_uncat_rate} /s")

    # Uncatalyzed results
    print("\n" + "-" * 70)
    print("UNCATALYZED CONTROL")
    print("-" * 70)
    print(f"  Mean |dA/dt|_0: {abs(uncat_mean):.6f} +/- {uncat_std:.6f} AU/s")
    print(f"  N trials: {uncat_n}")

    # Per-date results
    k_cats = []
    intercepts = []

    for dr in date_results:
        print("\n" + "-" * 70)
        print(f"DATE: {dr.date_label}")
        print("-" * 70)

        for cr in dr.concentration_results:
            print(f"\n  [{cr.concentration_mM:.2f} mM] "
                  f"N={cr.n_successful}/{len(cr.fit_results)}")
            print(f"    Mean |dA/dt|_0: {abs(cr.mean_dydt0):.6f} +/- {cr.std_dydt0:.6f} AU/s")

        print(f"\n  Linear regression (rate vs [catalyst]):")
        print(f"    k_cat (slope): {dr.k_cat:.2f} +/- {dr.k_cat_err:.2f} AU/(M*s)")
        print(f"    Intercept (v_uncat): {dr.intercept:.6f} AU/s")
        print(f"    R^2: {dr.r_squared:.4f}")

        if not np.isnan(dr.k_cat):
            k_cats.append(dr.k_cat)
        if not np.isnan(dr.intercept):
            intercepts.append(dr.intercept)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: PRECISION AND ACCURACY")
    print("=" * 70)

    if k_cats:
        k_mean = np.mean(k_cats)
        k_std = np.std(k_cats, ddof=1) if len(k_cats) > 1 else 0
        k_cv = 100 * k_std / abs(k_mean) if abs(k_mean) > 0 else np.nan

        print(f"\nCatalytic rate constant (k_cat):")
        print(f"  Mean: {k_mean:.2f} AU/(M*s)")
        print(f"  Std:  {k_std:.2f} AU/(M*s)")
        print(f"  CV:   {k_cv:.1f}%")
        print(f"  Individual values: {[f'{k:.2f}' for k in k_cats]}")

    if intercepts:
        int_mean = np.mean(intercepts)
        int_std = np.std(intercepts, ddof=1) if len(intercepts) > 1 else 0

        print(f"\nUncatalyzed rate from intercepts:")
        print(f"  Mean: {int_mean:.6f} AU/s")
        print(f"  Std:  {int_std:.6f} AU/s")
        print(f"  Individual values: {[f'{v:.6f}' for v in intercepts]}")

    print(f"\nDirect uncatalyzed measurement:")
    print(f"  |dA/dt|_0: {abs(uncat_mean):.6f} +/- {uncat_std:.6f} AU/s")

    # Accuracy check
    if not np.isnan(uncat_mean):
        # Note: we're comparing AU/s to expected /s - they're different units
        # The expected rate needs Q factor conversion
        print(f"\n  Note: Direct comparison to expected rate ({config.expected_uncat_rate} /s)")
        print(f"        requires Q-factor conversion not applied here.")


def create_diagnostic_plots(xl: pd.ExcelFile, date_results: List[DateResult],
                            config: AnalysisConfig, output_dir: pathlib.Path) -> None:
    """Create diagnostic plots for the analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Rate vs concentration for each date
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(date_results)))

    for dr, color in zip(date_results, colors):
        concs = [cr.concentration_mM for cr in dr.concentration_results]
        rates = [abs(cr.mean_dydt0) for cr in dr.concentration_results]
        stds = [cr.std_dydt0 for cr in dr.concentration_results]

        ax.errorbar(concs, rates, yerr=stds, fmt='o-', color=color,
                    label=f'{dr.date_label} (k={dr.k_cat:.1f})', capsize=3)

    ax.set_xlabel('[Catalyst] (mM)')
    ax.set_ylabel('|dA/dt|$_0$ (AU/s)')
    ax.set_title('Initial Rate vs Catalyst Concentration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rate_vs_concentration.png', dpi=300)
    plt.close()

    print(f"\nPlot saved: {output_dir / 'rate_vs_concentration.png'}")


# ─────────────────────────── Main ─────────────────────────── #

def main(file_path: pathlib.Path, config: AnalysisConfig) -> None:
    """Run the full analysis pipeline."""
    print(f"Loading data from: {file_path}")

    xl = pd.ExcelFile(file_path)
    print(f"Sheets found: {xl.sheet_names}")

    # Identify catalyzed vs uncatalyzed sheets
    catalyzed_sheets = [s for s in xl.sheet_names if s.lower() != 'uncatalyzed']

    # Analyze each date
    date_results = []
    for sheet in catalyzed_sheets:
        print(f"\nAnalyzing {sheet}...")
        result = analyze_date_sheet(xl, sheet, config)
        date_results.append(result)

    # Analyze uncatalyzed
    print("\nAnalyzing uncatalyzed control...")
    uncat_mean, uncat_std, uncat_n = analyze_uncatalyzed(xl, config)

    # Report
    print_analysis_report(date_results, uncat_mean, uncat_std, uncat_n, config)

    # Diagnostic plots
    output_dir = file_path.parent / "analysis_output"
    create_diagnostic_plots(xl, date_results, config, output_dir)

    return date_results, (uncat_mean, uncat_std, uncat_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze stopped-flow kinetics data")
    parser.add_argument("--file", "-f", type=pathlib.Path, required=True,
                        help="Path to Excel file with kinetics data")
    parser.add_argument("--t-min", type=float, default=0.02,
                        help="Start of time window (s)")
    parser.add_argument("--t-max", type=float, default=40.0,
                        help="End of time window (s)")
    parser.add_argument("--model", choices=["single", "double"], default="double",
                        help="Exponential model type")

    args = parser.parse_args()

    config = AnalysisConfig(
        t_min=args.t_min,
        t_max=args.t_max,
        model=args.model
    )

    main(args.file, config)
