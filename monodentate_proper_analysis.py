#!/usr/bin/env python3
"""
Proper analysis of Monodentate study.

Structure: Each file (e.g., MIm_25x.xlsx) has sheets for different Zn concentrations
(0.25mM, 0.5mM, 1mM), just like Benz_Cyclen.

The "25x" means 25 equivalents of N-methylimidazole per Zn.
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
        return None, None

    n = max(1, len(y_fit) // 20)
    y_start, y_end = np.median(y_fit[:n]), np.median(y_fit[-n:])
    total_amp = y_start - y_end

    try:
        p0 = [y_end, total_amp * 0.5, 1.0, total_amp * 0.5, 0.1]
        bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                  [np.inf, np.inf, 100, np.inf, 100])
        popt, _ = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)

        A0, A1, k1, A2, k2 = popt

        # Calculate R²
        y_pred = double_exp(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot

        dydt0 = -A1 * k1 - A2 * k2
        rate = abs(dydt0) / CO2_CONC * Q

        return rate, r2
    except:
        return None, None


def analyze_file(filepath):
    """Analyze one Monodentate file, return k_cat and k_uncat."""
    xl = pd.ExcelFile(filepath)

    conc_rates = []

    for sheet in xl.sheet_names:
        if sheet.lower() == 'results':
            continue

        # Parse concentration from sheet name
        match = re.search(r'(\d+\.?\d*)p?(\d*)mm', sheet.lower())
        if match:
            if match.group(2):  # e.g., "0p25mm"
                conc_mM = float(f"{match.group(1)}.{match.group(2)}")
            else:  # e.g., "1mm"
                conc_mM = float(match.group(1))
        else:
            continue

        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values

        rates = []
        for col in range(1, df.shape[1]):
            col_data = df.iloc[1:, col]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue

            header = df.iloc[0, col] if col < len(df.iloc[0]) else None
            if isinstance(header, str) and 'Q' in str(header).upper():
                continue

            trial_data = pd.to_numeric(col_data, errors='coerce').values
            rate, r2 = fit_and_get_rate(time, trial_data)

            if rate and r2 and r2 > 0.99:
                rates.append(rate)

        if rates:
            conc_rates.append((conc_mM, np.mean(rates), np.std(rates, ddof=1), len(rates)))

    if len(conc_rates) >= 2:
        concs_M = np.array([c / 1000 for c, _, _, _ in conc_rates])
        means = np.array([m for _, m, _, _ in conc_rates])
        slope, intercept, r_value, _, _ = linregress(concs_M, means)

        return {
            'k_cat': slope,
            'k_uncat': intercept,
            'r2': r_value**2,
            'conc_rates': conc_rates
        }

    return None


def main():
    base_path = pathlib.Path("Data/Monodentate_Study")

    print("=" * 80)
    print("MONODENTATE STUDY: PROPER ANALYSIS")
    print("=" * 80)
    print("\nEach file = different ligand equivalents")
    print("Each sheet = different Zn concentration (0.25, 0.5, 1.0 mM)")
    print("Linear regression gives k_cat for each ligand equivalent setting\n")

    files = list(base_path.glob("**/*.xlsx"))

    results = []

    for f in sorted(files):
        # Get ligand equivalents from filename
        match = re.search(r'(\d+)x', f.stem, re.IGNORECASE)
        if not match:
            continue
        ligand_eq = int(match.group(1))

        # Get date from path
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', str(f))
        date = date_match.group(1) if date_match else 'unknown'

        try:
            result = analyze_file(f)
            if result:
                result['ligand_eq'] = ligand_eq
                result['date'] = date
                result['file'] = f.stem
                results.append(result)

                print(f"{ligand_eq}x NMeIm ({date}):")
                print(f"  k_cat = {result['k_cat']:.1f} /M/s")
                print(f"  k_uncat = {result['k_uncat']:.4f} /s")
                print(f"  R² = {result['r2']:.4f}")
                for conc, mean, std, n in result['conc_rates']:
                    print(f"    {conc}mM: rate = {mean:.4f} ± {std:.4f} (n={n})")
                print()
        except Exception as e:
            print(f"Error with {f}: {e}")

    # Summary
    print("=" * 80)
    print("SUMMARY BY LIGAND EQUIVALENTS")
    print("=" * 80)

    # Group by ligand equivalents
    by_ligand = defaultdict(list)
    for r in results:
        by_ligand[r['ligand_eq']].append(r)

    print(f"\n{'Ligand Eq':<12} {'k_cat (/M/s)':<20} {'k_uncat (/s)':<15} {'N days':<8} {'CV(%)':<8}")
    print("-" * 70)

    all_kcats = []
    ligand_summary = []

    for ligand_eq in sorted(by_ligand.keys()):
        runs = by_ligand[ligand_eq]
        k_cats = [r['k_cat'] for r in runs]
        k_uncats = [r['k_uncat'] for r in runs]

        mean_kcat = np.mean(k_cats)
        std_kcat = np.std(k_cats, ddof=1) if len(k_cats) > 1 else 0
        cv_kcat = 100 * std_kcat / mean_kcat if mean_kcat != 0 and len(k_cats) > 1 else 0

        all_kcats.extend(k_cats)

        ligand_summary.append({
            'ligand_eq': ligand_eq,
            'mean_kcat': mean_kcat,
            'std_kcat': std_kcat,
            'cv': cv_kcat,
            'n': len(runs)
        })

        kcat_str = f"{mean_kcat:.1f} ± {std_kcat:.1f}" if len(k_cats) > 1 else f"{mean_kcat:.1f}"
        print(f"{ligand_eq}x{'':<10} {kcat_str:<20} {np.mean(k_uncats):<15.4f} {len(runs):<8} {cv_kcat:<8.1f}")

    # Overall CV comparison
    print("\n" + "=" * 80)
    print("COMPARISON TO BENZ_CYCLEN")
    print("=" * 80)

    # For ligand equivalents with multiple measurements
    multi_day = [s for s in ligand_summary if s['n'] > 1]
    if multi_day:
        avg_cv = np.mean([s['cv'] for s in multi_day])
        print(f"\n  Monodentate average within-condition CV: {avg_cv:.1f}%")
        print(f"  Benz_Cyclen day-to-day CV: 27.3%")

        if avg_cv < 20:
            print(f"\n  ✓ Monodentate ({avg_cv:.1f}%) is MORE CONSISTENT than Benz_Cyclen (27.3%)")
        else:
            print(f"\n  ~ Similar variability")

    # Look at k_uncat validation
    print("\n" + "=" * 80)
    print("k_uncat VALIDATION")
    print("=" * 80)

    all_kuncats = [r['k_uncat'] for r in results]
    print(f"\n  Mean k_uncat: {np.mean(all_kuncats):.4f} ± {np.std(all_kuncats):.4f} /s")
    print(f"  Expected: ~0.12-0.15 /s")

    if 0.10 <= np.mean(all_kuncats) <= 0.20:
        print("  ✓ VALID - k_uncat matches expected range")
    else:
        print("  ⚠ k_uncat outside expected range")

    # Final verdict
    print("\n" + "=" * 80)
    print("PUBLISHABILITY VERDICT")
    print("=" * 80)

    if multi_day:
        avg_cv = np.mean([s['cv'] for s in multi_day])
        valid_kuncat = 0.10 <= np.mean(all_kuncats) <= 0.20

        if avg_cv < 20 and valid_kuncat:
            print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✓ PUBLISHABLE AS QUANTITATIVE                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

  - Day-to-day CV is acceptable (<20%)
  - k_uncat validates the method
  - Multiple ligand equivalents provide trend information
""")
        else:
            print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ~ PUBLISHABLE WITH CAVEATS                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

  - Day-to-day CV: {avg_cv:.1f}%
  - k_uncat valid: {valid_kuncat}
  - Report uncertainties honestly
""")


if __name__ == "__main__":
    main()
