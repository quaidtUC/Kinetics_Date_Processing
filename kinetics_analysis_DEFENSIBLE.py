#!/usr/bin/env python3
"""
DEFENSIBLE KINETICS ANALYSIS
============================
CO2 Hydration Kinetics via Stopped-Flow Spectrophotometry

This script implements a clear, step-by-step analysis that can be explained
and defended in front of reviewers.

PHYSICAL MODEL:
--------------
The indicator absorbance decay follows first-order kinetics:

    A(t) = A_final + A_amplitude * exp(-k_obs * t)

where:
    A_final     = final absorbance (baseline)
    A_amplitude = total absorbance change
    k_obs       = observed rate constant (/s)

The observed rate depends on catalyst concentration:

    k_obs = k_uncat + k_cat * [catalyst]

where:
    k_uncat = uncatalyzed rate (~0.12 /s for CO2 hydration)
    k_cat   = catalytic rate constant (/M/s) - WHAT WE WANT TO MEASURE

FITTING APPROACH:
----------------
Single Exponential: Fits one rate constant to the entire decay
    + Simple, robust, fewer parameters
    + k_uncat recovery is excellent (~0.11-0.13 /s)
    - May underestimate fast initial kinetics

Double Exponential: Fits two rate constants (fast + slow)
    + Better captures fast initial phase
    + Reports initial rate dydt0 = -A1*k1 - A2*k2
    - k_uncat is elevated (~0.15-0.21 /s)
    - More parameters = more variability

RECOMMENDED: Single exponential for k_cat determination because it gives
physically consistent k_uncat values that validate the method.

Author: Thomas Quaid
Date: 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import argparse


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

EXPECTED_K_UNCAT = 0.12  # /s - literature value for uncatalyzed CO2 hydration


# =============================================================================
# MATHEMATICAL MODELS
# =============================================================================

def single_exponential(t, A_final, A_amplitude, k_obs):
    """
    Single exponential decay model.

    A(t) = A_final + A_amplitude * exp(-k_obs * t)

    At t=0:  A(0) = A_final + A_amplitude = A_initial
    At t=inf: A(inf) = A_final

    Parameters:
        t           : time (seconds)
        A_final     : final absorbance (AU)
        A_amplitude : absorbance change (AU), typically positive for decay
        k_obs       : observed rate constant (/s)

    Returns:
        Absorbance at time t
    """
    return A_final + A_amplitude * np.exp(-k_obs * t)


def double_exponential(t, A_final, A1, k1, A2, k2):
    """
    Double exponential decay model.

    A(t) = A_final + A1*exp(-k1*t) + A2*exp(-k2*t)

    This models two concurrent processes:
        - Fast phase (k1): catalyzed reaction
        - Slow phase (k2): uncatalyzed background

    The initial rate is:
        dA/dt|_(t=0) = -A1*k1 - A2*k2
    """
    return A_final + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrialFit:
    """Results from fitting a single kinetic trace."""
    trial_id: int
    k_obs: float          # Observed rate constant (/s)
    k_obs_err: float      # Standard error
    A_final: float        # Baseline absorbance
    A_amplitude: float    # Amplitude
    r_squared: float      # Goodness of fit
    dydt0: float          # Initial rate (AU/s)


@dataclass
class ConcentrationData:
    """All data for one catalyst concentration."""
    concentration_mM: float
    trials: List[TrialFit]

    @property
    def concentration_M(self) -> float:
        return self.concentration_mM / 1000

    @property
    def k_obs_mean(self) -> float:
        return np.mean([t.k_obs for t in self.trials])

    @property
    def k_obs_std(self) -> float:
        return np.std([t.k_obs for t in self.trials], ddof=1) if len(self.trials) > 1 else 0


@dataclass
class AnalysisResult:
    """Final analysis results."""
    k_cat: float           # Catalytic rate constant (/M/s)
    k_cat_err: float       # Standard error
    k_uncat: float         # Uncatalyzed rate (/s) - should be ~0.12
    k_uncat_err: float     # Standard error
    r_squared: float       # R² of linear regression
    method: str            # "single" or "double"


# =============================================================================
# CORE FITTING FUNCTIONS
# =============================================================================

def fit_single_exponential(time: np.ndarray, absorbance: np.ndarray,
                           t_min: float = 0.02, t_max: float = 40.0) -> TrialFit:
    """
    Fit a single exponential to one kinetic trace.

    STEP-BY-STEP PROCEDURE:
    1. Apply time window (exclude mixing artifact at t < 0.02s)
    2. Estimate initial parameters from data
    3. Perform nonlinear least-squares fit
    4. Calculate goodness of fit (R²)
    5. Extract rate constant k_obs

    This is the RECOMMENDED method because k_uncat ~ 0.12 /s (physically correct).
    """
    # Step 1: Apply time window
    mask = (time >= t_min) & (time <= t_max)
    t = time[mask]
    y = absorbance[mask]

    # Remove any NaN values
    valid = ~np.isnan(y) & ~np.isnan(t)
    t = t[valid]
    y = y[valid]

    if len(t) < 50:
        raise ValueError("Insufficient data points")

    # Step 2: Estimate initial parameters
    # Use median of first/last points for robustness
    y_initial = np.median(y[:20])
    y_final = np.median(y[-50:])
    A_amplitude_guess = y_initial - y_final

    # Estimate k from time to 63% decay (1 - 1/e)
    target = y_initial - 0.632 * A_amplitude_guess
    idx = np.argmin(np.abs(y - target))
    k_guess = 1.0 / max(t[idx], 0.1)

    # Step 3: Perform fit
    p0 = [y_final, A_amplitude_guess, k_guess]
    bounds = (
        [-np.inf, -np.inf, 0.01],   # Lower bounds
        [np.inf, np.inf, 10.0]       # Upper bounds (k < 10 /s is physical)
    )

    popt, pcov = curve_fit(single_exponential, t, y, p0=p0,
                           bounds=bounds, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))

    # Step 4: Calculate R²
    y_pred = single_exponential(t, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot

    # Step 5: Extract results
    A_final, A_amplitude, k_obs = popt
    k_obs_err = perr[2]

    # Initial rate: dA/dt|_(t=0) = -A_amplitude * k_obs
    dydt0 = -A_amplitude * k_obs

    return TrialFit(
        trial_id=0,
        k_obs=k_obs,
        k_obs_err=k_obs_err,
        A_final=A_final,
        A_amplitude=A_amplitude,
        r_squared=r_squared,
        dydt0=dydt0
    )


def fit_double_exponential(time: np.ndarray, absorbance: np.ndarray,
                           t_min: float = 0.02, t_max: float = 40.0) -> TrialFit:
    """
    Fit a double exponential to one kinetic trace.

    This captures both fast (catalyzed) and slow (uncatalyzed) phases.

    The INITIAL RATE is: dydt0 = -A1*k1 - A2*k2

    We convert this to an effective k_obs using:
        k_obs = |dydt0| / |A1 + A2|

    This is equivalent to amplitude-weighted averaging of the two rates.

    WARNING: This method gives elevated k_uncat (~0.15-0.21 /s), which is
    higher than the expected 0.12 /s. Use with caution.
    """
    # Step 1: Apply time window
    mask = (time >= t_min) & (time <= t_max)
    t = time[mask]
    y = absorbance[mask]

    valid = ~np.isnan(y) & ~np.isnan(t)
    t = t[valid]
    y = y[valid]

    if len(t) < 100:
        raise ValueError("Insufficient data points for double exponential")

    # Step 2: Estimate initial parameters
    y_initial = np.median(y[:20])
    y_final = np.median(y[-50:])
    total_amplitude = y_initial - y_final

    # Guess: split amplitude 50/50 between fast and slow
    p0 = [y_final, total_amplitude*0.5, 1.0, total_amplitude*0.5, 0.1]
    bounds = (
        [-np.inf, -np.inf, 0.1, -np.inf, 0.01],
        [np.inf, np.inf, 20.0, np.inf, 2.0]
    )

    # Step 3: Perform fit
    popt, pcov = curve_fit(double_exponential, t, y, p0=p0,
                           bounds=bounds, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))

    # Step 4: Calculate R²
    y_pred = double_exponential(t, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot

    # Step 5: Extract results
    A_final, A1, k1, A2, k2 = popt

    # Initial rate
    dydt0 = -A1 * k1 - A2 * k2

    # Effective k_obs (amplitude-weighted)
    total_amp = abs(A1) + abs(A2)
    k_obs = abs(dydt0) / total_amp if total_amp > 0.001 else np.nan

    # Error propagation (simplified)
    k_obs_err = 0.05 * k_obs  # Approximate 5% error

    return TrialFit(
        trial_id=0,
        k_obs=k_obs,
        k_obs_err=k_obs_err,
        A_final=A_final,
        A_amplitude=A1 + A2,
        r_squared=r_squared,
        dydt0=dydt0
    )


# =============================================================================
# LINEAR REGRESSION
# =============================================================================

def linear_regression_kobs_vs_concentration(
    concentrations_M: np.ndarray,
    k_obs_values: np.ndarray
) -> AnalysisResult:
    """
    Perform linear regression: k_obs = k_uncat + k_cat * [catalyst]

    This is the KEY STEP that extracts the catalytic rate constant.

    VALIDATION: k_uncat (intercept) should be ~0.12 /s
                If it's much higher, the fitting method may be biased.
    """
    slope, intercept, r_value, p_value, std_err = linregress(
        concentrations_M, k_obs_values
    )

    # Calculate intercept error
    n = len(concentrations_M)
    x_mean = np.mean(concentrations_M)
    ss_x = np.sum((concentrations_M - x_mean)**2)
    y_pred = slope * concentrations_M + intercept
    mse = np.sum((k_obs_values - y_pred)**2) / (n - 2) if n > 2 else 0
    intercept_err = np.sqrt(mse * (1/n + x_mean**2 / ss_x)) if n > 2 else 0

    return AnalysisResult(
        k_cat=slope,
        k_cat_err=std_err,
        k_uncat=intercept,
        k_uncat_err=intercept_err,
        r_squared=r_value**2,
        method=""
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_excel_data(file_path: Path) -> Dict[float, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Load kinetics data from Excel file.

    Supports two formats:
    1. Multi-sheet: Each sheet is a concentration (e.g., "0p25mm", "0p5mm", "1mm")
    2. Single-sheet: Columns grouped by concentration headers

    Returns:
        Dict mapping concentration_mM -> (time_array, [trial_arrays])
    """
    import re

    def extract_conc(s):
        s = str(s).lower().replace(' ', '')
        patterns = [r'(\d+)p(\d+)mm', r'(\d+\.?\d*)mm', r'concentration\s*=?\s*(\d+\.?\d*)']
        for pattern in patterns:
            match = re.search(pattern, s)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return float(f"{groups[0]}.{groups[1]}")
                return float(groups[0])
        return None

    xl = pd.ExcelFile(file_path)
    data = {}

    # Try multi-sheet format first
    for sheet in xl.sheet_names:
        conc = extract_conc(sheet)
        if conc is not None:
            df = pd.read_excel(xl, sheet_name=sheet, header=None)
            time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values

            trials = []
            for col in range(1, df.shape[1]):
                trial = pd.to_numeric(df.iloc[1:, col], errors='coerce').values
                # Skip if mostly NaN or monotonically increasing (blank column)
                valid = ~np.isnan(trial)
                if valid.sum() > 100:
                    diffs = np.diff(trial[valid])
                    if not ((diffs > 0).sum() / len(diffs) > 0.95):
                        trials.append(trial)

            if trials:
                data[conc] = (time, trials)

    if not data:
        # Try single-sheet format
        df = pd.read_excel(xl, sheet_name=0, header=None)
        row0 = df.iloc[0].values
        conc_cols = [(i, extract_conc(v)) for i, v in enumerate(row0)
                     if extract_conc(v) is not None]

        for idx, (col_start, conc) in enumerate(conc_cols):
            next_col = conc_cols[idx+1][0] if idx+1 < len(conc_cols) else df.shape[1]
            time = pd.to_numeric(df.iloc[2:, col_start], errors='coerce').values

            trials = []
            for col in range(col_start+1, next_col):
                trial = pd.to_numeric(df.iloc[2:, col], errors='coerce').values
                if not np.isnan(trial).all():
                    trials.append(trial)

            if trials:
                data[conc] = (time, trials)

    return data


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_dataset(file_path: Path, method: str = "single",
                    t_min: float = 0.02, t_max: float = 40.0,
                    verbose: bool = True) -> Tuple[AnalysisResult, List[ConcentrationData]]:
    """
    Full analysis pipeline.

    PROCEDURE:
    1. Load data from Excel file
    2. For each concentration:
       a. Fit each trial with chosen method
       b. Calculate mean k_obs
    3. Linear regression: k_obs vs [catalyst]
    4. Extract k_cat and k_uncat
    5. Validate: k_uncat should be ~0.12 /s
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"KINETICS ANALYSIS: {file_path.name}")
        print(f"{'='*70}")
        print(f"Method: {method} exponential")
        print(f"Time window: {t_min} - {t_max} s")

    # Step 1: Load data
    data = load_excel_data(file_path)

    if not data:
        raise ValueError("No data loaded from file")

    if verbose:
        print(f"\nLoaded concentrations: {sorted(data.keys())} mM")

    # Step 2: Fit each concentration
    fit_func = fit_single_exponential if method == "single" else fit_double_exponential

    conc_data_list = []

    for conc_mM in sorted(data.keys()):
        time, trials = data[conc_mM]
        trial_fits = []

        for i, trial in enumerate(trials):
            try:
                result = fit_func(time, trial, t_min, t_max)
                result.trial_id = i + 1
                trial_fits.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to fit trial {i+1} at {conc_mM} mM: {e}")

        if trial_fits:
            conc_data = ConcentrationData(conc_mM, trial_fits)
            conc_data_list.append(conc_data)

            if verbose:
                print(f"\n[{conc_mM:.2f} mM] {len(trial_fits)} trials fitted")
                print(f"  k_obs = {conc_data.k_obs_mean:.4f} +/- {conc_data.k_obs_std:.4f} /s")

    # Step 3: Linear regression
    concentrations_M = np.array([cd.concentration_M for cd in conc_data_list])
    k_obs_means = np.array([cd.k_obs_mean for cd in conc_data_list])

    result = linear_regression_kobs_vs_concentration(concentrations_M, k_obs_means)
    result.method = method

    if verbose:
        print(f"\n{'-'*70}")
        print("LINEAR REGRESSION: k_obs = k_uncat + k_cat * [catalyst]")
        print(f"{'-'*70}")
        print(f"  k_cat   = {result.k_cat:.1f} +/- {result.k_cat_err:.1f} /M/s")
        print(f"  k_uncat = {result.k_uncat:.4f} +/- {result.k_uncat_err:.4f} /s")
        print(f"  R^2     = {result.r_squared:.4f}")

        print(f"\n{'-'*70}")
        print("VALIDATION")
        print(f"{'-'*70}")
        deviation = abs(result.k_uncat - EXPECTED_K_UNCAT) / EXPECTED_K_UNCAT * 100
        if deviation < 20:
            print(f"  k_uncat is within 20% of expected value (0.12 /s) - GOOD")
        else:
            print(f"  WARNING: k_uncat deviates {deviation:.0f}% from expected 0.12 /s")
            print(f"  This may indicate systematic bias in the fitting method.")

    return result, conc_data_list


def create_summary_plot(conc_data_list: List[ConcentrationData],
                        result: AnalysisResult,
                        output_path: Path) -> None:
    """Create publication-quality plot of k_obs vs [catalyst]."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Data points
    concs_mM = [cd.concentration_mM for cd in conc_data_list]
    k_obs_means = [cd.k_obs_mean for cd in conc_data_list]
    k_obs_stds = [cd.k_obs_std for cd in conc_data_list]

    ax.errorbar(concs_mM, k_obs_means, yerr=k_obs_stds,
                fmt='o', markersize=10, capsize=5, capthick=2,
                color='#2E86AB', ecolor='#2E86AB', label='Data')

    # Trendline
    x_line = np.linspace(0, max(concs_mM) * 1.1, 100)
    y_line = result.k_uncat + result.k_cat * (x_line / 1000)

    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'k_obs = {result.k_uncat:.3f} + {result.k_cat:.0f}[cat]\n'
                  f'R² = {result.r_squared:.4f}')

    ax.set_xlabel('[Catalyst] (mM)', fontsize=12)
    ax.set_ylabel('k_obs (/s)', fontsize=12)
    ax.set_title(f'Rate vs Concentration ({result.method.title()} Exp)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(concs_mM) * 1.15)
    ax.set_ylim(0, max(k_obs_means) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved plot: {output_path}")


# =============================================================================
# EXCEL EXPORT FOR VERIFICATION
# =============================================================================

def export_to_excel(conc_data_list: List[ConcentrationData],
                    result: AnalysisResult,
                    output_path: Path) -> None:
    """
    Export all data to Excel for manual verification.

    Creates a spreadsheet showing:
    1. All fitted parameters for each trial
    2. Summary statistics per concentration
    3. Linear regression calculation

    This allows reviewers to verify the analysis step-by-step.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Sheet 1: Raw fit results
        rows = []
        for cd in conc_data_list:
            for t in cd.trials:
                rows.append({
                    'Concentration (mM)': cd.concentration_mM,
                    'Concentration (M)': cd.concentration_M,
                    'Trial': t.trial_id,
                    'k_obs (/s)': t.k_obs,
                    'k_obs_err': t.k_obs_err,
                    'A_final (AU)': t.A_final,
                    'A_amplitude (AU)': t.A_amplitude,
                    'dydt0 (AU/s)': t.dydt0,
                    'R²': t.r_squared
                })

        df_raw = pd.DataFrame(rows)
        df_raw.to_excel(writer, sheet_name='All Trials', index=False)

        # Sheet 2: Summary per concentration
        summary_rows = []
        for cd in conc_data_list:
            summary_rows.append({
                'Concentration (mM)': cd.concentration_mM,
                'Concentration (M)': cd.concentration_M,
                'N trials': len(cd.trials),
                'k_obs mean (/s)': cd.k_obs_mean,
                'k_obs std (/s)': cd.k_obs_std
            })

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 3: Regression results
        reg_data = {
            'Parameter': ['k_cat', 'k_cat_err', 'k_uncat', 'k_uncat_err', 'R²', 'Method'],
            'Value': [result.k_cat, result.k_cat_err, result.k_uncat,
                     result.k_uncat_err, result.r_squared, result.method],
            'Units': ['/M/s', '/M/s', '/s', '/s', '', '']
        }
        df_reg = pd.DataFrame(reg_data)
        df_reg.to_excel(writer, sheet_name='Regression', index=False)

    print(f"Saved Excel summary: {output_path}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze stopped-flow kinetics data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE USAGE:
    python kinetics_analysis_DEFENSIBLE.py data.xlsx --method single
    python kinetics_analysis_DEFENSIBLE.py data.xlsx --method double
    python kinetics_analysis_DEFENSIBLE.py data.xlsx --t-min 0.02 --t-max 40

VALIDATION:
    The intercept (k_uncat) should be approximately 0.12 /s.
    If it deviates significantly, the fitting method may be biased.

    Single exponential: k_uncat ~ 0.11-0.13 /s (GOOD)
    Double exponential: k_uncat ~ 0.15-0.21 /s (ELEVATED)
        """
    )

    parser.add_argument("file", type=Path, help="Excel file with kinetics data")
    parser.add_argument("--method", choices=["single", "double"], default="single",
                        help="Fitting method (default: single)")
    parser.add_argument("--t-min", type=float, default=0.02,
                        help="Start of fitting window in seconds (default: 0.02)")
    parser.add_argument("--t-max", type=float, default=40.0,
                        help="End of fitting window in seconds (default: 40.0)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: same as input)")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    # Run analysis
    result, conc_data = analyze_dataset(
        args.file,
        method=args.method,
        t_min=args.t_min,
        t_max=args.t_max
    )

    # Output
    output_dir = args.output_dir or args.file.parent
    stem = args.file.stem

    create_summary_plot(
        conc_data, result,
        output_dir / f"{stem}_{args.method}_plot.png"
    )

    export_to_excel(
        conc_data, result,
        output_dir / f"{stem}_{args.method}_results.xlsx"
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    exit(main())
