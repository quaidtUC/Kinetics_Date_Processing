#!/usr/bin/env python3
"""
Reproduce Original User Method
==============================
Implements the user's original calculation:
1. Fit double exponential to get dydt0 = -A1*k1 - A2*k2 (AU/s)
2. Convert: rate = |dydt0| / [CO2] * |Q|
3. Regress rate vs [catalyst]
4. Slope = k_cat, intercept = k_uncat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib

# Constants
CO2_CONCENTRATION = 0.0338  # M (saturated CO2)
Q_CATALYZED = -0.402  # From the instrument
Q_UNCATALYZED = -0.607

# Time window
T_MIN = 0.02
T_MAX = 40.0


def double_exp(t, A0, A1, k1, A2, k2):
    """Double exponential decay."""
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_double_exp(time, absorbance, t_min=T_MIN, t_max=T_MAX):
    """Fit double exponential and return dydt0."""
    mask = (time >= t_min) & (time <= t_max)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit = t_fit[valid]
    y_fit = y_fit[valid]

    if len(t_fit) < 20:
        return None, None, None

    # Initial guesses
    n_window = max(1, len(y_fit) // 20)
    y_start = np.median(y_fit[:n_window])
    y_end = np.median(y_fit[-n_window:])
    A0_guess = y_end
    total_amp = y_start - y_end

    try:
        p0 = [A0_guess, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
        bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                  [np.inf, np.inf, 100, np.inf, 100])

        popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)

        A0, A1, k1, A2, k2 = popt
        dydt0 = -A1 * k1 - A2 * k2  # Initial rate in AU/s

        # R² calculation
        y_pred = double_exp(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - ss_res / ss_tot

        return popt, dydt0, r_squared
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None, None, None


def load_and_process_date_sheet(xl, sheet_name, concentrations_mM=[0.25, 0.5, 1.0]):
    """Load one date's data and extract dydt0 for each concentration."""
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    # Find concentration columns
    row0 = df.iloc[0].values
    conc_cols = [i for i, v in enumerate(row0)
                 if isinstance(v, str) and 'concentration' in v.lower()]

    results = []

    for i, col_idx in enumerate(conc_cols):
        # Get concentration from header
        import re
        header = str(row0[col_idx])
        match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
        if match:
            conc_mM = float(match.group(1))
        else:
            conc_mM = concentrations_mM[i] if i < len(concentrations_mM) else None

        next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]

        # Extract time and trials
        time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

        dydt0_values = []
        for tc in range(col_idx + 1, next_idx):
            col_data = df.iloc[2:, tc]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue
            try:
                trial_data = pd.to_numeric(col_data, errors='coerce').values
                popt, dydt0, r2 = fit_double_exp(time, trial_data)
                if dydt0 is not None:
                    dydt0_values.append(dydt0)
            except:
                continue

        if dydt0_values:
            mean_dydt0 = np.mean(dydt0_values)
            results.append({
                'conc_mM': conc_mM,
                'mean_dydt0': mean_dydt0,
                'n_trials': len(dydt0_values),
                'dydt0_values': dydt0_values
            })

    return results


def convert_dydt0_to_rate(dydt0, CO2_conc=CO2_CONCENTRATION, Q=Q_CATALYZED):
    """
    Convert initial rate (AU/s) to observed rate constant (/s).

    User's method: rate = |dydt0| / [CO2] * |Q|

    But wait - that gives units of AU/(M*s) * dimensionless = AU/(M*s)

    Actually, let me think about this more carefully...

    dydt0 has units AU/s
    [CO2] has units M
    Q is dimensionless (ratio of extinction coefficients)

    dydt0 / [CO2] = AU/(M*s)

    To get /s, we need something else...

    Actually, I think the conversion should be:
    rate (1/s) = |dydt0| / (amplitude in AU)

    OR the user's method might be:
    rate = |dydt0| / [CO2] / Q
    which would give AU/(M*s) / (AU/(M*cm)) = cm/s... still not /s

    Let me just try the literal interpretation first.
    """
    return abs(dydt0) / CO2_conc * abs(Q)


def main():
    file_path = pathlib.Path("/Users/thomasquaid/Projects/Kinetics_Data_Processing/Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")

    xl = pd.ExcelFile(file_path)
    print(f"Sheets: {xl.sheet_names}")

    date_sheets = [s for s in xl.sheet_names if s.lower() != 'uncatalyzed']

    # Try different [CO2] interpretations
    co2_interpretations = [
        ("0.0338 M (saturated)", 0.0338),
        ("0.0338 mM (user said)", 0.0338e-3),
        ("33.8 mM", 0.0338),
    ]

    for co2_label, CO2_conc in co2_interpretations[:1]:  # Just use the standard one
        print(f"\n{'#'*60}")
        print(f"Using [CO2] = {co2_label}")
        print(f"{'#'*60}")

        all_results = []

        for sheet in date_sheets:
            print(f"\n{'='*60}")
            print(f"Processing: {sheet}")
            print(f"{'='*60}")

            results = load_and_process_date_sheet(xl, sheet)

            print("\nDouble Exponential Fit Results (dydt0 in AU/s):")
            for r in results:
                print(f"  {r['conc_mM']} mM: dydt0 = {r['mean_dydt0']:.6f} AU/s (n={r['n_trials']})")

            # Convert using user's method
            print("\nConverted rates using |dydt0| / [CO2] * |Q|:")
            converted_rates = []
            for r in results:
                rate = abs(r['mean_dydt0']) / CO2_conc * abs(Q_CATALYZED)
                converted_rates.append(rate)
                print(f"  {r['conc_mM']} mM: rate = {rate:.4f}")

            # Linear regression: rate vs [catalyst]
            concs_M = np.array([r['conc_mM'] / 1000 for r in results])
            rates = np.array(converted_rates)

            slope, intercept, r_value, p_value, std_err = linregress(concs_M, rates)

            print(f"\nLinear regression: rate = intercept + k_cat × [catalyst]")
            print(f"  k_cat (slope) = {slope:.1f}")
            print(f"  k_uncat (intercept) = {intercept:.4f}")
            print(f"  R² = {r_value**2:.4f}")

            all_results.append({
                'date': sheet,
                'k_cat': slope,
                'k_uncat': intercept,
                'r_squared': r_value**2,
                'concs': concs_M,
                'rates': rates
            })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: k_cat values across dates")
    print("="*60)
    k_cats = [r['k_cat'] for r in all_results]
    k_uncats = [r['k_uncat'] for r in all_results]

    for r in all_results:
        print(f"  {r['date']}: k_cat = {r['k_cat']:.1f}, k_uncat = {r['k_uncat']:.4f}")

    print(f"\n  Mean k_cat: {np.mean(k_cats):.1f}")
    print(f"  Std k_cat: {np.std(k_cats, ddof=1):.1f}")
    print(f"  CV: {100*np.std(k_cats, ddof=1)/np.mean(k_cats):.1f}%")

    print(f"\n  Mean k_uncat: {np.mean(k_uncats):.4f}")
    print(f"  Expected k_uncat: 0.12 /s")

    # Now let's try different Q interpretations
    print("\n" + "="*60)
    print("Trying different interpretations...")
    print("="*60)

    # What Q would make k_uncat = 0.12?
    # Let's work backwards from user's values
    print("\nUser's reported values were approximately:")
    print("  k_cat = 164, 210, 251 /M/s")
    print("  k_uncat ≈ 0.14 /s")

    # Let's see what raw dydt0 values look like
    print("\nRaw dydt0 summary (from first concentration group, 0.25 mM):")
    for sheet in date_sheets:
        results = load_and_process_date_sheet(xl, sheet)
        if results:
            print(f"  {sheet}: dydt0 = {results[0]['mean_dydt0']:.6f} AU/s")


if __name__ == "__main__":
    main()
