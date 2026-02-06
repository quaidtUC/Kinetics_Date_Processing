#!/usr/bin/env python3
"""
Stopped-Flow Kinetics Data Processor
====================================
Processes Excel files containing stopped-flow spectrophotometry data for CO2 hydration.

Features:
- Automatic detection of Excel file format (multi-sheet or single-sheet layouts)
- Single exponential fitting for robust rate constant extraction
- Weighted and unweighted linear regression
- Publication-quality plots

Usage:
    python process_kinetics.py /path/to/data.xlsx
    python process_kinetics.py /path/to/data.xlsx --Q -0.452
    python process_kinetics.py /path/to/data.xlsx --t-min 0.02 --t-max 40

Output:
    - Console: Initial rates, k values, intercept, standard deviations
    - CSV: Summary of all fitted parameters
    - PNG: Rate vs concentration plot with trendline
    - PNG: Individual fits with data overlay
"""

from __future__ import annotations
import argparse
import pathlib
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress


# ─────────────────────────── Configuration ─────────────────────────── #

@dataclass
class Config:
    Q: float = -0.452           # Extinction coefficient factor
    t_min: float = 0.02         # Start of fitting window (s)
    t_max: float = 40.0         # End of fitting window (s)
    concentrations: List[float] = None  # Will be auto-detected


# ─────────────────────────── Model ─────────────────────────── #

def single_exp(t: np.ndarray, A0: float, A: float, k: float) -> np.ndarray:
    """Single exponential decay: A(t) = A0 + A * exp(-k * t)"""
    return A0 + A * np.exp(-k * t)


# ─────────────────────────── Data Loading ─────────────────────────── #

def extract_concentration_from_string(s: str) -> Optional[float]:
    """Extract concentration in mM from various string formats."""
    s = str(s).lower().replace(' ', '')

    # Try patterns like "0p25mm", "0.25mm", "concentration = 0.25 mM"
    patterns = [
        r'(\d+)p(\d+)mm',           # 0p25mm -> 0.25
        r'(\d+\.?\d*)mm',           # 0.25mm or 1mm
        r'concentration\s*=?\s*(\d+\.?\d*)',  # concentration = 0.25
    ]

    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            groups = match.groups()
            if len(groups) == 2:  # 0p25 format
                return float(f"{groups[0]}.{groups[1]}")
            else:
                return float(groups[0])
    return None


def detect_file_format(xl: pd.ExcelFile) -> str:
    """
    Detect the format of the Excel file.

    Returns:
        'multi_sheet': One sheet per concentration (e.g., '0p25mm', '0p5mm', '1mm')
        'single_sheet': All concentrations in one sheet with column groups
    """
    sheet_names = xl.sheet_names

    # Check if sheet names contain concentration info
    conc_found = 0
    for name in sheet_names:
        if extract_concentration_from_string(name) is not None:
            conc_found += 1

    if conc_found >= 2:
        return 'multi_sheet'

    # Check first sheet for concentration column headers
    df = pd.read_excel(xl, sheet_name=0, header=None)
    row0 = df.iloc[0].values
    for val in row0:
        if isinstance(val, str) and 'concentration' in val.lower():
            return 'single_sheet'

    # Default to multi_sheet if sheets look like concentrations
    return 'multi_sheet'


def is_time_column(col_data: np.ndarray) -> bool:
    """
    Check if a column appears to be a time column (monotonically increasing).
    """
    valid = ~np.isnan(col_data)
    if valid.sum() < 10:
        return False
    col_valid = col_data[valid]
    # Check if mostly monotonically increasing (allow some noise)
    diffs = np.diff(col_valid)
    return (diffs > 0).sum() / len(diffs) > 0.95


def is_valid_absorbance_trial(col_data: np.ndarray, time: np.ndarray, t_min: float = 0.02, t_max: float = 40.0) -> bool:
    """
    Check if a column appears to be valid absorbance data (not a time column, not all NaN).
    """
    # Must have sufficient non-NaN values
    valid = ~np.isnan(col_data)
    if valid.sum() < 100:
        return False

    # Should not be monotonically increasing (that would be a time column)
    if is_time_column(col_data):
        return False

    # Values should be in reasonable absorbance range (typically -0.5 to 2.0)
    col_valid = col_data[valid]
    if col_valid.min() < -1.0 or col_valid.max() > 3.0:
        return False

    return True


def load_multi_sheet_format(xl: pd.ExcelFile) -> Dict[float, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Load data from multi-sheet format (one sheet per concentration).

    Returns:
        Dict mapping concentration (mM) -> (time_array, [trial_arrays])
    """
    data = {}

    for sheet_name in xl.sheet_names:
        conc_mM = extract_concentration_from_string(sheet_name)
        if conc_mM is None:
            print(f"  Warning: Could not extract concentration from sheet '{sheet_name}', skipping")
            continue

        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

        # Row 0 contains column headers, data starts at row 1
        # First column is time (x), rest are trials
        time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values

        trials = []
        for col in range(1, df.shape[1]):
            trial_data = pd.to_numeric(df.iloc[1:, col], errors='coerce').values

            # Skip columns that are empty, time columns, or invalid
            if is_valid_absorbance_trial(trial_data, time):
                trials.append(trial_data)

        data[conc_mM] = (time, trials)
        print(f"  Loaded {sheet_name}: {len(trials)} trials")

    return data


def load_single_sheet_format(xl: pd.ExcelFile, sheet_name: str = None
                              ) -> Dict[float, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Load data from single-sheet format (concentration groups in columns).

    Returns:
        Dict mapping concentration (mM) -> (time_array, [trial_arrays])
    """
    if sheet_name is None:
        sheet_name = xl.sheet_names[0]

    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    # Find concentration group boundaries from row 0
    row0 = df.iloc[0].values
    conc_cols = []
    for i, val in enumerate(row0):
        if isinstance(val, str) and 'concentration' in val.lower():
            conc_mM = extract_concentration_from_string(val)
            conc_cols.append((i, conc_mM))

    data = {}

    for idx, (col_idx, conc_mM) in enumerate(conc_cols):
        next_idx = conc_cols[idx + 1][0] if idx + 1 < len(conc_cols) else df.shape[1]

        # Time is first column of group, trials are rest
        time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

        trials = []
        for col in range(col_idx + 1, next_idx):
            trial_data = pd.to_numeric(df.iloc[2:, col], errors='coerce').values
            # Skip columns that are mostly NaN or contain metadata
            if np.isnan(trial_data).sum() < len(trial_data) * 0.5:
                trials.append(trial_data)

        data[conc_mM] = (time, trials)
        print(f"  Loaded {conc_mM} mM: {len(trials)} trials")

    return data


def load_data(file_path: pathlib.Path) -> Dict[float, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Load kinetics data from Excel file, auto-detecting format.

    Returns:
        Dict mapping concentration (mM) -> (time_array, [trial_arrays])
    """
    print(f"\nLoading: {file_path.name}")
    xl = pd.ExcelFile(file_path)

    file_format = detect_file_format(xl)
    print(f"  Detected format: {file_format}")

    if file_format == 'multi_sheet':
        return load_multi_sheet_format(xl)
    else:
        return load_single_sheet_format(xl)


# ─────────────────────────── Fitting ─────────────────────────── #

@dataclass
class FitResult:
    """Results from fitting a single trial."""
    k: float                    # Rate constant (/s)
    k_err: float               # Standard error of k
    A0: float                  # Baseline
    A: float                   # Amplitude
    dydt0: float               # Initial rate (AU/s)
    r_squared: float           # Goodness of fit
    success: bool              # Whether fit converged


def fit_single_trial(time: np.ndarray, absorbance: np.ndarray,
                     t_min: float, t_max: float) -> FitResult:
    """
    Fit single exponential to one trial.
    """
    # Apply time window
    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    # Remove NaN values
    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 20:
        return FitResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)

    # Robust initial parameter estimation
    n_window = max(1, len(y_fit) // 20)
    y_start = np.median(y_fit[:n_window])
    y_end = np.median(y_fit[-n_window:])

    A0_guess = y_end
    A_guess = y_start - y_end

    # Estimate k from half-life
    if abs(A_guess) > 1e-10:
        half_amp = y_start - 0.5 * (y_start - y_end)
        idx_half = np.argmin(np.abs(y_fit - half_amp))
        t_half = t_fit[idx_half] if idx_half > 0 else 1.0
        k_guess = 0.693 / max(t_half, 0.01)
    else:
        k_guess = 0.1

    try:
        p0 = [A0_guess, A_guess, k_guess]
        bounds = ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 50])

        popt, pcov = curve_fit(single_exp, t_fit, y_fit, p0=p0,
                               bounds=bounds, maxfev=50000)
        perr = np.sqrt(np.diag(pcov))

        # Calculate R²
        y_pred = single_exp(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - ss_res / ss_tot

        # Initial rate
        dydt0 = -popt[1] * popt[2]

        return FitResult(
            k=popt[2],
            k_err=perr[2],
            A0=popt[0],
            A=popt[1],
            dydt0=dydt0,
            r_squared=r_squared,
            success=True
        )

    except (RuntimeError, ValueError) as e:
        return FitResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)


@dataclass
class ConcentrationResults:
    """Results for all trials at one concentration."""
    concentration_mM: float
    fit_results: List[FitResult]

    @property
    def successful_fits(self) -> List[FitResult]:
        return [f for f in self.fit_results if f.success]

    @property
    def n_successful(self) -> int:
        return len(self.successful_fits)

    @property
    def k_values(self) -> np.ndarray:
        return np.array([f.k for f in self.successful_fits])

    @property
    def k_errors(self) -> np.ndarray:
        return np.array([f.k_err for f in self.successful_fits])

    @property
    def mean_k(self) -> float:
        return np.mean(self.k_values) if self.n_successful > 0 else np.nan

    @property
    def std_k(self) -> float:
        return np.std(self.k_values, ddof=1) if self.n_successful > 1 else 0.0

    @property
    def dydt0_values(self) -> np.ndarray:
        return np.array([f.dydt0 for f in self.successful_fits])

    @property
    def mean_dydt0(self) -> float:
        return np.mean(self.dydt0_values) if self.n_successful > 0 else np.nan

    @property
    def std_dydt0(self) -> float:
        return np.std(self.dydt0_values, ddof=1) if self.n_successful > 1 else 0.0


def process_concentration(conc_mM: float, time: np.ndarray, trials: List[np.ndarray],
                          config: Config) -> ConcentrationResults:
    """Process all trials for one concentration."""
    fit_results = []

    for trial in trials:
        result = fit_single_trial(time, trial, config.t_min, config.t_max)
        fit_results.append(result)

    return ConcentrationResults(conc_mM, fit_results)


# ─────────────────────────── Linear Regression ─────────────────────────── #

@dataclass
class RegressionResults:
    """Results from linear regression of k vs [catalyst]."""
    k_cat: float               # Slope (catalytic rate constant, /M/s)
    k_cat_err: float           # Standard error of slope
    k_uncat: float             # Intercept (uncatalyzed rate, /s)
    k_uncat_err: float         # Standard error of intercept
    r_squared: float           # R² value
    weighted: bool             # Whether weighted regression was used


def weighted_linear_regression(x: np.ndarray, y: np.ndarray,
                                y_err: np.ndarray) -> RegressionResults:
    """
    Perform weighted least squares linear regression.

    Weights are 1/variance (1/y_err²).
    """
    # Remove any points with zero or nan errors
    valid = (y_err > 0) & ~np.isnan(y_err) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]
    y_err = y_err[valid]

    if len(x) < 2:
        return RegressionResults(np.nan, np.nan, np.nan, np.nan, np.nan, True)

    weights = 1.0 / (y_err ** 2)

    # Weighted means
    sum_w = np.sum(weights)
    x_mean = np.sum(weights * x) / sum_w
    y_mean = np.sum(weights * y) / sum_w

    # Weighted slope and intercept
    numerator = np.sum(weights * (x - x_mean) * (y - y_mean))
    denominator = np.sum(weights * (x - x_mean) ** 2)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Standard errors
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # Weighted residual sum of squares
    chi_sq = np.sum(weights * residuals ** 2)
    n = len(x)

    # Standard errors (using weighted formulas)
    slope_err = np.sqrt(1.0 / denominator)
    intercept_err = np.sqrt(1.0 / sum_w + x_mean ** 2 / denominator)

    # Weighted R²
    ss_tot = np.sum(weights * (y - y_mean) ** 2)
    ss_res = np.sum(weights * residuals ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return RegressionResults(slope, slope_err, intercept, intercept_err, r_squared, True)


def unweighted_linear_regression(x: np.ndarray, y: np.ndarray) -> RegressionResults:
    """Perform ordinary least squares linear regression."""
    valid = ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return RegressionResults(np.nan, np.nan, np.nan, np.nan, np.nan, False)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Calculate intercept standard error
    n = len(x)
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)
    y_pred = slope * x + intercept
    mse = np.sum((y - y_pred) ** 2) / (n - 2)
    intercept_err = np.sqrt(mse * (1/n + x_mean**2 / ss_x))

    return RegressionResults(slope, std_err, intercept, intercept_err, r_value**2, False)


# ─────────────────────────── Plotting ─────────────────────────── #

def plot_rate_vs_concentration(conc_results: List[ConcentrationResults],
                                reg_unweighted: RegressionResults,
                                reg_weighted: RegressionResults,
                                output_path: pathlib.Path) -> None:
    """Create rate vs concentration plot with trendline."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    concs_mM = np.array([cr.concentration_mM for cr in conc_results])
    concs_M = concs_mM / 1000  # Convert to M for regression
    mean_ks = np.array([cr.mean_k for cr in conc_results])
    std_ks = np.array([cr.std_k for cr in conc_results])

    # Plot data points with error bars
    ax.errorbar(concs_mM, mean_ks, yerr=std_ks, fmt='o', markersize=10,
                capsize=5, capthick=2, color='#2E86AB', ecolor='#2E86AB',
                label='Data', zorder=3)

    # Plot unweighted trendline
    x_line = np.linspace(0, concs_mM.max() * 1.1, 100)
    x_line_M = x_line / 1000
    y_unweighted = reg_unweighted.k_uncat + reg_unweighted.k_cat * x_line_M
    ax.plot(x_line, y_unweighted, 'r-', linewidth=2,
            label=f'Unweighted: k = {reg_unweighted.k_uncat:.4f} + {reg_unweighted.k_cat:.1f}[cat]\n'
                  f'R² = {reg_unweighted.r_squared:.4f}', zorder=2)

    # Plot weighted trendline
    y_weighted = reg_weighted.k_uncat + reg_weighted.k_cat * x_line_M
    ax.plot(x_line, y_weighted, 'g--', linewidth=2,
            label=f'Weighted: k = {reg_weighted.k_uncat:.4f} + {reg_weighted.k_cat:.1f}[cat]\n'
                  f'R² = {reg_weighted.r_squared:.4f}', zorder=2)

    ax.set_xlabel('[Catalyst] (mM)', fontsize=12)
    ax.set_ylabel('k_obs (/s)', fontsize=12)
    ax.set_title('Observed Rate Constant vs Catalyst Concentration', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, concs_mM.max() * 1.15)
    ax.set_ylim(0, mean_ks.max() * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved: {output_path}")


def plot_fits(data: Dict[float, Tuple[np.ndarray, List[np.ndarray]]],
              conc_results: List[ConcentrationResults],
              config: Config,
              output_path: pathlib.Path) -> None:
    """Create plot showing data with exponential fit overlay for each concentration."""
    n_concs = len(conc_results)
    fig, axes = plt.subplots(1, n_concs, figsize=(5 * n_concs, 4))

    if n_concs == 1:
        axes = [axes]

    # Sort by concentration
    sorted_results = sorted(conc_results, key=lambda x: x.concentration_mM)

    for ax, cr in zip(axes, sorted_results):
        time, trials = data[cr.concentration_mM]

        # Apply time window for plotting
        mask = (time >= config.t_min) & (time <= config.t_max)
        t_plot = time[mask]

        # Plot all trials (faded)
        for i, trial in enumerate(trials):
            y_plot = trial[mask]
            valid = ~np.isnan(y_plot)
            ax.plot(t_plot[valid], y_plot[valid], '-', alpha=0.3,
                    color='#7EC8E3', linewidth=0.5)

        # Plot fit using mean parameters
        if cr.n_successful > 0:
            # Use parameters from first successful fit for overlay
            first_success = cr.successful_fits[0]
            t_smooth = np.linspace(config.t_min, config.t_max, 500)
            y_fit = single_exp(t_smooth, first_success.A0, first_success.A, first_success.k)
            ax.plot(t_smooth, y_fit, 'r-', linewidth=2,
                    label=f'k = {cr.mean_k:.4f} ± {cr.std_k:.4f} /s')

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Absorbance (AU)', fontsize=10)
        ax.set_title(f'{cr.concentration_mM} mM\n(n = {cr.n_successful} trials)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Single Exponential Fits', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────── Main Processing ─────────────────────────── #

def process_file(file_path: pathlib.Path, config: Config) -> None:
    """Main processing function."""

    # Load data
    data = load_data(file_path)

    if not data:
        print("Error: No data loaded!")
        return

    # Process each concentration
    conc_results = []
    for conc_mM in sorted(data.keys()):
        time, trials = data[conc_mM]
        result = process_concentration(conc_mM, time, trials, config)
        conc_results.append(result)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTime window: {config.t_min} - {config.t_max} s")
    print(f"Q factor: {config.Q}")

    print("\n" + "-" * 70)
    print("Per-Concentration Results:")
    print("-" * 70)

    for cr in conc_results:
        print(f"\n[{cr.concentration_mM:.2f} mM] - {cr.n_successful}/{len(cr.fit_results)} successful fits")
        print(f"  k_obs:    {cr.mean_k:.4f} ± {cr.std_k:.4f} /s")
        print(f"  dA/dt|₀:  {cr.mean_dydt0:.6f} ± {cr.std_dydt0:.6f} AU/s")

        # Print individual trial k values
        if cr.n_successful > 0:
            k_strs = [f"{f.k:.4f}" for f in cr.successful_fits]
            print(f"  Individual k values: [{', '.join(k_strs)}]")

    # Linear regression
    concs_M = np.array([cr.concentration_mM / 1000 for cr in conc_results])
    mean_ks = np.array([cr.mean_k for cr in conc_results])
    std_ks = np.array([cr.std_k for cr in conc_results])

    reg_unweighted = unweighted_linear_regression(concs_M, mean_ks)
    reg_weighted = weighted_linear_regression(concs_M, mean_ks, std_ks)

    print("\n" + "-" * 70)
    print("Linear Regression: k_obs = k_uncat + k_cat × [catalyst]")
    print("-" * 70)

    print("\nUnweighted:")
    print(f"  k_cat (slope):      {reg_unweighted.k_cat:.1f} ± {reg_unweighted.k_cat_err:.1f} /M/s")
    print(f"  k_uncat (intercept): {reg_unweighted.k_uncat:.4f} ± {reg_unweighted.k_uncat_err:.4f} /s")
    print(f"  R²:                  {reg_unweighted.r_squared:.4f}")

    print("\nWeighted:")
    print(f"  k_cat (slope):      {reg_weighted.k_cat:.1f} ± {reg_weighted.k_cat_err:.1f} /M/s")
    print(f"  k_uncat (intercept): {reg_weighted.k_uncat:.4f} ± {reg_weighted.k_uncat_err:.4f} /s")
    print(f"  R²:                  {reg_weighted.r_squared:.4f}")

    # Output directory
    output_dir = file_path.parent / f"{file_path.stem}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV summary
    summary_data = []
    for cr in conc_results:
        for i, fr in enumerate(cr.fit_results):
            summary_data.append({
                'concentration_mM': cr.concentration_mM,
                'trial': i + 1,
                'k': fr.k,
                'k_err': fr.k_err,
                'A0': fr.A0,
                'A': fr.A,
                'dydt0': fr.dydt0,
                'r_squared': fr.r_squared,
                'success': fr.success
            })

    summary_df = pd.DataFrame(summary_data)
    csv_path = output_dir / f"{file_path.stem}_fit_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save regression summary
    reg_summary = {
        'metric': ['k_cat (/M/s)', 'k_cat_err', 'k_uncat (/s)', 'k_uncat_err', 'R_squared'],
        'unweighted': [reg_unweighted.k_cat, reg_unweighted.k_cat_err,
                       reg_unweighted.k_uncat, reg_unweighted.k_uncat_err,
                       reg_unweighted.r_squared],
        'weighted': [reg_weighted.k_cat, reg_weighted.k_cat_err,
                     reg_weighted.k_uncat, reg_weighted.k_uncat_err,
                     reg_weighted.r_squared]
    }
    reg_df = pd.DataFrame(reg_summary)
    reg_csv_path = output_dir / f"{file_path.stem}_regression.csv"
    reg_df.to_csv(reg_csv_path, index=False)
    print(f"Saved: {reg_csv_path}")

    # Create plots
    plot_rate_vs_concentration(
        conc_results, reg_unweighted, reg_weighted,
        output_dir / f"{file_path.stem}_rate_vs_conc.png"
    )

    plot_fits(
        data, conc_results, config,
        output_dir / f"{file_path.stem}_fits.png"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# ─────────────────────────── CLI ─────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Process stopped-flow kinetics data from Excel files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python process_kinetics.py data.xlsx
    python process_kinetics.py data.xlsx --Q -0.452
    python process_kinetics.py data.xlsx --t-min 0.02 --t-max 40
        """
    )

    parser.add_argument("file", type=pathlib.Path,
                        help="Path to Excel file with kinetics data")
    parser.add_argument("--Q", type=float, default=-0.452,
                        help="Q factor (extinction coefficient ratio). Default: -0.452")
    parser.add_argument("--t-min", type=float, default=0.02,
                        help="Start of fitting window in seconds. Default: 0.02")
    parser.add_argument("--t-max", type=float, default=40.0,
                        help="End of fitting window in seconds. Default: 40.0")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    config = Config(
        Q=args.Q,
        t_min=args.t_min,
        t_max=args.t_max
    )

    process_file(args.file, config)
    return 0


if __name__ == "__main__":
    exit(main())
