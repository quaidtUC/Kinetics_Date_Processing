#!/usr/bin/env python3
"""
Assess Monodentate study for publishable quantitative accuracy.

Structure: Each date folder has ONE concentration, tested on multiple days.
We need to combine all concentrations to get k_cat.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib
from collections import defaultdict

T_MIN, T_MAX = 0.02, 40.0
CO2_CONC = 0.0338
Q = 0.402


def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_and_get_rate(time, absorbance):
    """Fit double exp and return rate using Working Method."""
    mask = (time >= T_MIN) & (time <= T_MAX)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit, y_fit = t_fit[valid], y_fit[valid]

    if len(t_fit) < 50:
        return None, None, None

    n = max(1, len(y_fit) // 20)
    y_start, y_end = np.median(y_fit[:n]), np.median(y_fit[-n:])
    total_amp = y_start - y_end

    try:
        p0 = [y_end, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
        bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                  [np.inf, np.inf, 100, np.inf, 100])
        popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)

        A0, A1, k1, A2, k2 = popt
        dydt0 = -A1 * k1 - A2 * k2
        rate = abs(dydt0) / CO2_CONC * Q

        # Calculate R²
        y_pred = double_exp(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot

        return rate, r2, total_amp
    except:
        return None, None, None


def main():
    base_path = pathlib.Path("Data/Monodentate_Study")

    print("=" * 80)
    print("MONODENTATE STUDY: PUBLISHABILITY ASSESSMENT (v2)")
    print("=" * 80)

    # Collect all data organized by concentration and sheet (day within concentration)
    import re

    # Structure: {conc_x: {sheet_name: [rates]}}
    data_by_conc_sheet = defaultdict(lambda: defaultdict(list))
    amp_by_conc_sheet = defaultdict(lambda: defaultdict(list))

    files = list(base_path.glob("**/*.xlsx"))

    for f in sorted(files):
        match = re.search(r'(\d+)x', f.stem, re.IGNORECASE)
        if not match:
            continue
        conc_x = int(match.group(1))

        try:
            xl = pd.ExcelFile(f)
            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet, header=None)
                time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values

                for col in range(1, df.shape[1]):
                    col_data = df.iloc[1:, col]
                    if col_data.isna().sum() > len(col_data) * 0.5:
                        continue

                    header = df.iloc[0, col] if col < len(df.iloc[0]) else None
                    if isinstance(header, str) and 'Q' in str(header).upper():
                        continue

                    trial_data = pd.to_numeric(col_data, errors='coerce').values
                    rate, r2, amp = fit_and_get_rate(time, trial_data)

                    if rate and r2 and r2 > 0.99:
                        data_by_conc_sheet[conc_x][sheet].append(rate)
                        amp_by_conc_sheet[conc_x][sheet].append(amp)
        except:
            pass

    # Print structure
    print("\n" + "=" * 60)
    print("DATA STRUCTURE")
    print("=" * 60)

    concs = sorted(data_by_conc_sheet.keys())
    print(f"\nConcentrations (x-fold): {concs}")

    for conc in concs:
        sheets = sorted(data_by_conc_sheet[conc].keys())
        n_total = sum(len(rates) for rates in data_by_conc_sheet[conc].values())
        print(f"  {conc}x: {len(sheets)} sheets, {n_total} trials total")

    # Key insight: Each concentration has 3 sheets (days)
    # We can assess day-to-day consistency WITHIN each concentration
    print("\n" + "=" * 60)
    print("DAY-TO-DAY CONSISTENCY WITHIN EACH CONCENTRATION")
    print("=" * 60)

    conc_cv_data = []

    for conc in concs:
        sheet_means = []
        for sheet, rates in sorted(data_by_conc_sheet[conc].items()):
            mean_rate = np.mean(rates)
            sheet_means.append(mean_rate)

        if len(sheet_means) >= 2:
            cv = 100 * np.std(sheet_means, ddof=1) / np.mean(sheet_means)
            conc_cv_data.append({'conc': conc, 'cv': cv, 'n_days': len(sheet_means)})
            print(f"  {conc}x: CV = {cv:.1f}% across {len(sheet_means)} days")
            print(f"       Day means: {[f'{m:.3f}' for m in sheet_means]}")

    mean_cv = np.mean([d['cv'] for d in conc_cv_data])
    print(f"\n  Average within-concentration CV: {mean_cv:.1f}%")

    # Now do linear regression to get k_cat
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION: rate vs [catalyst]")
    print("=" * 60)

    # Method 1: Use all trials (pooled)
    print("\n  Method 1: Pooled (all trials)")
    all_concs = []
    all_rates = []

    for conc in concs:
        for sheet, rates in data_by_conc_sheet[conc].items():
            for rate in rates:
                all_concs.append(conc)
                all_rates.append(rate)

    slope, intercept, r_value, p_value, std_err = linregress(all_concs, all_rates)
    print(f"    Slope (∝ k_cat): {slope:.6f} per x-fold")
    print(f"    Intercept: {intercept:.4f}")
    print(f"    R²: {r_value**2:.4f}")
    print(f"    n = {len(all_rates)} trials")

    # Method 2: Use concentration means
    print("\n  Method 2: Concentration means")
    conc_means = []
    conc_stds = []

    for conc in concs:
        all_rates_conc = []
        for sheet, rates in data_by_conc_sheet[conc].items():
            all_rates_conc.extend(rates)
        conc_means.append((conc, np.mean(all_rates_conc), np.std(all_rates_conc, ddof=1)))

    concs_arr = np.array([c for c, _, _ in conc_means])
    means_arr = np.array([m for _, m, _ in conc_means])

    slope2, intercept2, r_value2, _, std_err2 = linregress(concs_arr, means_arr)
    print(f"    Slope (∝ k_cat): {slope2:.6f} per x-fold")
    print(f"    Intercept: {intercept2:.4f}")
    print(f"    R²: {r_value2**2:.4f}")

    print("\n  Rate by concentration:")
    for conc, mean, std in conc_means:
        print(f"    {conc}x: {mean:.4f} ± {std:.4f}")

    # Method 3: Bootstrap - do regression on each "day" combination
    print("\n  Method 3: Day-by-day regression (treating each sheet as independent)")

    # Each concentration has multiple sheets; pick one sheet from each
    from itertools import product

    sheets_by_conc = {conc: list(data_by_conc_sheet[conc].keys()) for conc in concs}

    # Get all combinations of picking one sheet per concentration
    sheet_combinations = list(product(*[sheets_by_conc[c] for c in concs]))

    slopes_bootstrap = []
    intercepts_bootstrap = []

    for combo in sheet_combinations:
        combo_concs = []
        combo_rates = []

        for i, conc in enumerate(concs):
            sheet = combo[i]
            rates = data_by_conc_sheet[conc][sheet]
            mean_rate = np.mean(rates)
            combo_concs.append(conc)
            combo_rates.append(mean_rate)

        if len(combo_concs) >= 2:
            s, i, r, _, _ = linregress(combo_concs, combo_rates)
            slopes_bootstrap.append(s)
            intercepts_bootstrap.append(i)

    if slopes_bootstrap:
        mean_slope = np.mean(slopes_bootstrap)
        std_slope = np.std(slopes_bootstrap, ddof=1)
        cv_slope = 100 * std_slope / mean_slope

        print(f"    {len(slopes_bootstrap)} day-combinations tested")
        print(f"    Slope: {mean_slope:.6f} ± {std_slope:.6f}")
        print(f"    CV of slope: {cv_slope:.1f}%")
        print(f"    Intercept: {np.mean(intercepts_bootstrap):.4f} ± {np.std(intercepts_bootstrap, ddof=1):.4f}")

    # Amplitude analysis
    print("\n" + "=" * 60)
    print("AMPLITUDE CONSISTENCY")
    print("=" * 60)

    all_amps = []
    amp_by_conc = {}

    for conc in concs:
        conc_amps = []
        for sheet, amps in amp_by_conc_sheet[conc].items():
            conc_amps.extend(amps)
            all_amps.extend(amps)
        amp_by_conc[conc] = conc_amps
        print(f"  {conc}x: {np.mean(conc_amps):.4f} ± {np.std(conc_amps):.4f}")

    print(f"\n  Overall: {np.mean(all_amps):.4f} ± {np.std(all_amps):.4f}")
    print(f"  CV: {100 * np.std(all_amps) / np.mean(all_amps):.1f}%")

    conc_amp_means = [np.mean(amps) for amps in amp_by_conc.values()]
    print(f"  CV of concentration means: {100 * np.std(conc_amp_means) / np.mean(conc_amp_means):.1f}%")

    # Final assessment
    print("\n" + "=" * 80)
    print("PUBLISHABILITY VERDICT")
    print("=" * 80)

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MONODENTATE STUDY METRICS                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Within-concentration day-to-day CV:  {mean_cv:.1f}%
  Amplitude CV:                         {100 * np.std(all_amps) / np.mean(all_amps):.1f}%
  Bootstrap slope CV:                   {cv_slope:.1f}%
  Linear regression R²:                 {r_value2**2:.4f}

╔══════════════════════════════════════════════════════════════════════════════╗
║  COMPARISON TO BENZ_CYCLEN                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Metric                    Monodentate     Benz_Cyclen
  ─────────────────────────────────────────────────────
  Day-to-day CV             {mean_cv:<8.1f}%       27.3%
  Amplitude CV              {100 * np.std(all_amps) / np.mean(all_amps):<8.1f}%       11.5%
  k_cat CV (bootstrap)      {cv_slope:<8.1f}%       27.3%
""")

    if cv_slope < 15 and mean_cv < 15:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✓ VERDICT: PUBLISHABLE AS QUANTITATIVE                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

  The Monodentate study shows good day-to-day reproducibility.
  You can report quantitative k_cat values with confidence.
""")
    elif cv_slope < 25:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ~ VERDICT: PUBLISHABLE WITH HONEST UNCERTAINTY                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

  The Monodentate study shows moderate reproducibility.
  Report with proper error bars; emphasize relative comparisons.
""")
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✗ VERDICT: SIMILAR ISSUES TO BENZ_CYCLEN                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

  The Monodentate study has comparable variability to Benz_Cyclen.
  This suggests the ~20-30% CV is your experimental precision limit.
""")


if __name__ == "__main__":
    main()
