#!/usr/bin/env python3
"""
Investigate why Monodentate shows no concentration dependence.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib
from collections import defaultdict
import re

T_MIN, T_MAX = 0.02, 40.0
CO2_CONC = 0.0338
Q = 0.402


def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_and_get_rate(time, absorbance):
    mask = (time >= T_MIN) & (time <= T_MAX)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit, y_fit = t_fit[valid], y_fit[valid]

    if len(t_fit) < 50:
        return None

    n = max(1, len(y_fit) // 20)
    y_start, y_end = np.median(y_fit[:n]), np.median(y_fit[-n:])
    total_amp = y_start - y_end

    try:
        p0 = [y_end, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
        bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                  [np.inf, np.inf, 100, np.inf, 100])
        popt, _ = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)

        A0, A1, k1, A2, k2 = popt
        dydt0 = -A1 * k1 - A2 * k2
        rate = abs(dydt0) / CO2_CONC * Q

        return rate
    except:
        return None


def main():
    base_path = pathlib.Path("Data/Monodentate_Study")

    print("=" * 80)
    print("MONODENTATE: INVESTIGATING CONCENTRATION DEPENDENCE")
    print("=" * 80)

    # Collect data
    data = defaultdict(list)
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
                    rate = fit_and_get_rate(time, trial_data)

                    if rate:
                        data[conc_x].append(rate)
        except:
            pass

    # Print summary
    print("\nRate by concentration:")
    print(f"{'Conc (x)':<10} {'Mean Rate':<12} {'Std':<10} {'N':<6}")
    print("-" * 40)

    concs = []
    means = []

    for conc in sorted(data.keys()):
        rates = data[conc]
        mean_rate = np.mean(rates)
        std_rate = np.std(rates, ddof=1)
        concs.append(conc)
        means.append(mean_rate)
        print(f"{conc:<10} {mean_rate:<12.4f} {std_rate:<10.4f} {len(rates):<6}")

    # Linear regression
    slope, intercept, r_value, p_value, _ = linregress(concs, means)

    print(f"\nLinear regression:")
    print(f"  Slope: {slope:.6f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4f}")

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  INTERPRETATION                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    if abs(slope) < 0.001 and r_value**2 < 0.5:
        print("""
  The rate shows NO significant dependence on catalyst concentration!

  Possible explanations:
  1. The monodentate catalyst has very low activity (k_cat ≈ 0)
  2. The concentration range is too narrow to see the effect
  3. The reaction is already saturated even at 10x

  This is actually a SCIENTIFIC FINDING, not a data quality issue.

  If the monodentate (N-methylimidazole) shows no catalytic enhancement
  compared to uncatalyzed, this is meaningful - it may indicate that
  the ligand coordination geometry matters (cyclen vs monodentate).
""")

        # Compare to uncatalyzed rate
        print("\n  Checking if rates match uncatalyzed...")
        all_rates = [r for rates in data.values() for r in rates]
        mean_rate = np.mean(all_rates)
        print(f"    Mean observed rate: {mean_rate:.4f} /s")
        print(f"    Expected k_uncat:   ~0.15 /s")
        print(f"    Ratio: {mean_rate / 0.15:.1f}x")

        if 0.5 < mean_rate / 0.15 < 2:
            print("\n  ⚠ The rates are ~SAME as uncatalyzed!")
            print("    This suggests N-methylimidazole is NOT an effective catalyst.")
        else:
            print("\n  The rates ARE elevated above uncatalyzed.")
            print("  But there's no concentration dependence...")

    else:
        print(f"  There IS concentration dependence (slope = {slope:.6f})")


if __name__ == "__main__":
    main()
