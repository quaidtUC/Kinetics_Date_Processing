#!/usr/bin/env python3
"""
DEFENSIBLE Kinetics Analysis: Double Exponential Method
========================================================

This script implements the user's original method for extracting k_cat
from stopped-flow CO2 hydration kinetics data.

METHOD:
1. Fit double exponential: A(t) = A0 + A1*exp(-k1*t) + A2*exp(-k2*t)
2. Calculate initial rate: dydt0 = -A1*k1 - A2*k2 (AU/s)
3. Convert to observed rate: rate = |dydt0| / [CO2] * |Q|
4. Linear regression: rate = k_uncat + k_cat × [catalyst]
   - Slope = k_cat (/M/s)
   - Intercept = k_uncat (/s)

VALIDATION:
- k_uncat should be ~0.12-0.15 /s (known uncatalyzed CO2 hydration rate)
- R² should be > 0.99 for good fits

Author: Auto-generated for defense presentation
Date: 2026-02-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (from instrument/literature)
# ═══════════════════════════════════════════════════════════════════════════

CO2_CONCENTRATION = 0.0338  # M (saturated CO2 at 25°C)
Q_FACTOR = 0.402            # Dimensionless (from instrument calibration)

# ═══════════════════════════════════════════════════════════════════════════
# FITTING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

T_MIN = 0.02   # Start of fitting window (s) - excludes mixing artifacts
T_MAX = 40.0   # End of fitting window (s) - captures full decay

# ═══════════════════════════════════════════════════════════════════════════
# MATHEMATICAL MODEL
# ═══════════════════════════════════════════════════════════════════════════

def double_exponential(t, A0, A1, k1, A2, k2):
    """
    Double exponential decay model.

    A(t) = A0 + A1*exp(-k1*t) + A2*exp(-k2*t)

    Parameters:
        t: time (s)
        A0: baseline absorbance (AU)
        A1: amplitude of fast component (AU)
        k1: rate constant of fast component (/s)
        A2: amplitude of slow component (AU)
        k2: rate constant of slow component (/s)

    Returns:
        Predicted absorbance at time t
    """
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def calculate_initial_rate(A1, k1, A2, k2):
    """
    Calculate the initial rate of absorbance change.

    dydt|_{t=0} = d/dt[A0 + A1*exp(-k1*t) + A2*exp(-k2*t)]|_{t=0}
                = -A1*k1 - A2*k2

    Units: AU/s
    """
    return -A1 * k1 - A2 * k2


def convert_to_observed_rate(dydt0, CO2_conc=CO2_CONCENTRATION, Q=Q_FACTOR):
    """
    Convert initial rate (AU/s) to observed rate constant.

    rate = |dydt0| / [CO2] × |Q|

    This conversion accounts for:
    - [CO2]: substrate concentration
    - Q: instrument calibration factor

    Parameters:
        dydt0: initial rate (AU/s)
        CO2_conc: CO2 concentration (M)
        Q: Q factor (dimensionless)

    Returns:
        Observed rate (/s)
    """
    return abs(dydt0) / CO2_conc * abs(Q)


# ═══════════════════════════════════════════════════════════════════════════
# DATA FITTING
# ═══════════════════════════════════════════════════════════════════════════

def fit_single_trace(time, absorbance, t_min=T_MIN, t_max=T_MAX):
    """
    Fit double exponential to a single kinetic trace.

    Parameters:
        time: time array (s)
        absorbance: absorbance array (AU)
        t_min, t_max: fitting window bounds (s)

    Returns:
        dict with fit results, or None if fit failed
    """
    # Apply time window
    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    # Remove NaN values
    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 50:
        return None

    # Estimate initial parameters from data
    n_window = max(1, len(y_fit) // 20)
    y_start = np.median(y_fit[:n_window])
    y_end = np.median(y_fit[-n_window:])

    A0_guess = y_end
    total_amplitude = y_start - y_end

    # Initial guesses: split amplitude between fast and slow components
    p0 = [
        A0_guess,           # A0: baseline
        total_amplitude * 0.5,  # A1: fast amplitude
        1.0,                # k1: fast rate (guess ~1 /s)
        total_amplitude * 0.5,  # A2: slow amplitude
        0.1                 # k2: slow rate (guess ~0.1 /s)
    ]

    # Bounds to ensure physical meaning
    bounds = (
        [-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],  # lower bounds
        [np.inf, np.inf, 100, np.inf, 100]         # upper bounds
    )

    try:
        popt, pcov = curve_fit(
            double_exponential, t_fit, y_fit,
            p0=p0, bounds=bounds, maxfev=50000
        )

        A0, A1, k1, A2, k2 = popt

        # Calculate fit quality
        y_pred = double_exponential(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - ss_res / ss_tot

        # Calculate initial rate
        dydt0 = calculate_initial_rate(A1, k1, A2, k2)

        # Convert to observed rate
        rate = convert_to_observed_rate(dydt0)

        return {
            'A0': A0,
            'A1': A1,
            'k1': k1,
            'A2': A2,
            'k2': k2,
            'dydt0': dydt0,
            'rate': rate,
            'r_squared': r_squared,
            'parameters': popt
        }

    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_excel_file(file_path):
    """
    Analyze all sheets in an Excel file.

    Returns:
        DataFrame with results for each date
    """
    xl = pd.ExcelFile(file_path)
    date_sheets = [s for s in xl.sheet_names if s.lower() != 'uncatalyzed']

    all_results = []

    for sheet in date_sheets:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {sheet}")
        print(f"{'='*60}")

        df = pd.read_excel(xl, sheet_name=sheet, header=None)

        # Find concentration columns
        row0 = df.iloc[0].values
        row1 = df.iloc[1].values

        conc_cols = [i for i, v in enumerate(row0)
                     if isinstance(v, str) and 'concentration' in v.lower()]

        conc_data = []

        for i, col_idx in enumerate(conc_cols):
            # Extract concentration from header
            import re
            header = str(row0[col_idx])
            match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
            conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

            print(f"\n  [{conc_mM} mM]")

            # Find trial columns
            next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
            time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

            trial_results = []
            for tc in range(col_idx + 1, next_idx):
                trial_name = str(row1[tc]) if tc < len(row1) else f"Trial {tc - col_idx}"
                col_data = df.iloc[2:, tc]

                if col_data.isna().sum() > len(col_data) * 0.5:
                    continue

                trial_data = pd.to_numeric(col_data, errors='coerce').values
                result = fit_single_trace(time, trial_data)

                if result:
                    trial_results.append(result)
                    print(f"    {trial_name}: dydt0 = {result['dydt0']:.6f} AU/s, "
                          f"rate = {result['rate']:.4f}, R² = {result['r_squared']:.4f}")

            if trial_results:
                mean_rate = np.mean([r['rate'] for r in trial_results])
                std_rate = np.std([r['rate'] for r in trial_results], ddof=1) if len(trial_results) > 1 else 0

                conc_data.append({
                    'conc_mM': conc_mM,
                    'conc_M': conc_mM / 1000,
                    'mean_rate': mean_rate,
                    'std_rate': std_rate,
                    'n_trials': len(trial_results),
                    'trial_results': trial_results
                })

                print(f"    MEAN: rate = {mean_rate:.4f} ± {std_rate:.4f} (n={len(trial_results)})")

        # Linear regression: rate vs [catalyst]
        if len(conc_data) >= 2:
            concs_M = np.array([c['conc_M'] for c in conc_data])
            rates = np.array([c['mean_rate'] for c in conc_data])

            slope, intercept, r_value, p_value, std_err = linregress(concs_M, rates)

            print(f"\n  LINEAR REGRESSION:")
            print(f"    k_obs = k_uncat + k_cat × [catalyst]")
            print(f"    ")
            print(f"    k_cat (slope):      {slope:.1f} ± {std_err:.1f} /M/s")
            print(f"    k_uncat (intercept): {intercept:.4f} /s")
            print(f"    R²:                  {r_value**2:.4f}")

            all_results.append({
                'date': sheet,
                'k_cat': slope,
                'k_cat_err': std_err,
                'k_uncat': intercept,
                'r_squared': r_value**2,
                'concentration_data': conc_data
            })

    return all_results


def create_summary_plot(results, output_path):
    """Create summary plot of rate vs concentration for all dates."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']

    for i, result in enumerate(results):
        conc_data = result['concentration_data']
        concs = [c['conc_mM'] for c in conc_data]
        rates = [c['mean_rate'] for c in conc_data]
        stds = [c['std_rate'] for c in conc_data]

        # Plot data points
        ax.errorbar(concs, rates, yerr=stds, fmt=markers[i], markersize=10,
                    capsize=5, capthick=2, color=colors[i],
                    label=f"{result['date']}: k_cat = {result['k_cat']:.0f} /M/s")

        # Plot regression line
        x_line = np.linspace(0, max(concs) * 1.1, 100)
        y_line = result['k_uncat'] + result['k_cat'] * (x_line / 1000)
        ax.plot(x_line, y_line, '-', color=colors[i], alpha=0.7, linewidth=2)

    ax.set_xlabel('[Catalyst] (mM)', fontsize=12)
    ax.set_ylabel('Observed Rate (/s)', fontsize=12)
    ax.set_title('Rate vs Catalyst Concentration\n(Double Exponential Method)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    # Add annotation with method details
    method_text = (
        f"Method: Double Exponential\n"
        f"dydt₀ = -A₁k₁ - A₂k₂\n"
        f"rate = |dydt₀| / [CO₂] × |Q|\n"
        f"[CO₂] = {CO2_CONCENTRATION} M\n"
        f"|Q| = {Q_FACTOR}"
    )
    ax.text(0.98, 0.02, method_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")


def main():
    """Main entry point."""
    print("="*70)
    print("DOUBLE EXPONENTIAL KINETICS ANALYSIS")
    print("="*70)
    print(f"\nParameters:")
    print(f"  [CO2] = {CO2_CONCENTRATION} M (saturated)")
    print(f"  |Q| = {Q_FACTOR} (instrument factor)")
    print(f"  Time window: {T_MIN} - {T_MAX} s")

    # Find the data file
    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    # Run analysis
    results = analyze_excel_file(file_path)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Date':<10} {'k_cat (/M/s)':<15} {'k_uncat (/s)':<15} {'R²':<10}")
    print("-"*50)

    k_cats = []
    k_uncats = []
    for r in results:
        print(f"{r['date']:<10} {r['k_cat']:<15.1f} {r['k_uncat']:<15.4f} {r['r_squared']:<10.4f}")
        k_cats.append(r['k_cat'])
        k_uncats.append(r['k_uncat'])

    print("-"*50)
    print(f"{'Mean':<10} {np.mean(k_cats):<15.1f} {np.mean(k_uncats):<15.4f}")
    print(f"{'Std':<10} {np.std(k_cats, ddof=1):<15.1f} {np.std(k_uncats, ddof=1):<15.4f}")
    print(f"{'CV (%)':<10} {100*np.std(k_cats, ddof=1)/np.mean(k_cats):<15.1f}")

    print(f"\nExpected k_uncat (literature): ~0.12-0.15 /s")
    print(f"Measured k_uncat (mean): {np.mean(k_uncats):.4f} /s")

    # Create plot
    output_dir = file_path.parent
    create_summary_plot(results, output_dir / "double_exp_summary_plot.png")

    # Save to Excel
    summary_df = pd.DataFrame([{
        'Date': r['date'],
        'k_cat (/M/s)': r['k_cat'],
        'k_cat_err': r['k_cat_err'],
        'k_uncat (/s)': r['k_uncat'],
        'R_squared': r['r_squared']
    } for r in results])

    excel_path = output_dir / "double_exp_summary_results.xlsx"
    summary_df.to_excel(excel_path, index=False)
    print(f"\nResults saved: {excel_path}")


if __name__ == "__main__":
    main()
