#!/usr/bin/env python3
"""
Assess Monodentate study for publishable quantitative accuracy.

Key questions:
1. What is the day-to-day CV?
2. Are measurements from different days statistically consistent?
3. Can we report a single k_cat value with confidence?
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress, f_oneway, ttest_ind
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
        return None, None

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

        return rate, r2
    except:
        return None, None


def main():
    base_path = pathlib.Path("Data/Monodentate_Study")

    print("=" * 80)
    print("MONODENTATE STUDY: PUBLISHABILITY ASSESSMENT")
    print("=" * 80)

    # Organize by date and concentration
    data_by_date = defaultdict(lambda: defaultdict(list))

    files = list(base_path.glob("**/*.xlsx"))

    for f in sorted(files):
        # Extract date from path (parent folder)
        date_folder = f.parent.name

        # Extract concentration from filename
        import re
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
                    rate, r2 = fit_and_get_rate(time, trial_data)

                    if rate and r2 and r2 > 0.99:
                        data_by_date[date_folder][conc_x].append(rate)
        except Exception as e:
            pass

    # Print structure
    print("\n" + "=" * 60)
    print("DATA STRUCTURE")
    print("=" * 60)

    all_dates = sorted(data_by_date.keys())
    print(f"\nDates found: {len(all_dates)}")
    for date in all_dates:
        concs = sorted(data_by_date[date].keys())
        print(f"  {date}: concentrations {concs}")

    # For each date, calculate k_cat via linear regression
    print("\n" + "=" * 60)
    print("k_cat BY DATE (Linear Regression: rate vs [catalyst])")
    print("=" * 60)

    date_results = []

    for date in all_dates:
        conc_data = data_by_date[date]

        # Get concentration-rate pairs
        conc_rate_pairs = []
        for conc_x, rates in sorted(conc_data.items()):
            # Convert x-fold to actual concentration (assuming base concentration)
            # Need to know what 1x means - let's work with relative values
            mean_rate = np.mean(rates)
            std_rate = np.std(rates, ddof=1) if len(rates) > 1 else 0
            conc_rate_pairs.append((conc_x, mean_rate, std_rate, len(rates)))

        if len(conc_rate_pairs) >= 2:
            concs = np.array([c for c, _, _, _ in conc_rate_pairs])
            rates = np.array([r for _, r, _, _ in conc_rate_pairs])

            slope, intercept, r_value, p_value, std_err = linregress(concs, rates)

            date_results.append({
                'date': date,
                'slope': slope,  # This is d(rate)/d(conc_x), proportional to k_cat
                'intercept': intercept,
                'r2': r_value**2,
                'n_conc': len(conc_rate_pairs),
                'conc_rate_pairs': conc_rate_pairs
            })

            print(f"\n  {date}:")
            print(f"    Concentrations tested: {[c for c, _, _, _ in conc_rate_pairs]}")
            print(f"    Slope (∝ k_cat): {slope:.4f} per x-fold")
            print(f"    Intercept (∝ k_uncat): {intercept:.4f}")
            print(f"    R²: {r_value**2:.4f}")

    # Compare slopes across dates
    print("\n" + "=" * 60)
    print("DAY-TO-DAY CONSISTENCY OF k_cat")
    print("=" * 60)

    if len(date_results) >= 2:
        slopes = [r['slope'] for r in date_results]
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes, ddof=1)
        cv_slope = 100 * std_slope / mean_slope

        print(f"\n  Slopes by date: {[f'{s:.4f}' for s in slopes]}")
        print(f"  Mean: {mean_slope:.4f}")
        print(f"  Std:  {std_slope:.4f}")
        print(f"  CV:   {cv_slope:.1f}%")

        # Compare to Benz_Cyclen
        print(f"\n  Compare to Benz_Cyclen: CV = 27.3%")

        if cv_slope < 15:
            print(f"\n  ✓ Monodentate CV ({cv_slope:.1f}%) is BETTER than Benz_Cyclen (27.3%)")
        elif cv_slope < 25:
            print(f"\n  ~ Monodentate CV ({cv_slope:.1f}%) is SIMILAR to Benz_Cyclen (27.3%)")
        else:
            print(f"\n  ✗ Monodentate CV ({cv_slope:.1f}%) is WORSE than Benz_Cyclen (27.3%)")

    # Check amplitude consistency
    print("\n" + "=" * 60)
    print("AMPLITUDE CONSISTENCY BY DATE")
    print("=" * 60)

    amp_by_date = defaultdict(list)

    for f in sorted(files):
        date_folder = f.parent.name

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

                    mask = (time >= T_MIN) & (time <= T_MAX)
                    t_fit = time[mask]
                    y_fit = trial_data[mask]
                    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
                    y_fit = y_fit[valid]

                    if len(y_fit) >= 50:
                        n = max(1, len(y_fit) // 20)
                        amp = np.median(y_fit[:n]) - np.median(y_fit[-n:])
                        if amp > 0.01:
                            amp_by_date[date_folder].append(amp)
        except:
            pass

    for date in sorted(amp_by_date.keys()):
        amps = amp_by_date[date]
        print(f"  {date}: {np.mean(amps):.4f} ± {np.std(amps):.4f} (n={len(amps)})")

    all_amps = [a for amps in amp_by_date.values() for a in amps]
    date_means = [np.mean(amps) for amps in amp_by_date.values()]

    print(f"\n  Overall amplitude: {np.mean(all_amps):.4f} ± {np.std(all_amps):.4f}")
    print(f"  CV of date means: {100 * np.std(date_means) / np.mean(date_means):.1f}%")

    # Final assessment
    print("\n" + "=" * 80)
    print("PUBLISHABILITY ASSESSMENT")
    print("=" * 80)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CRITERIA FOR QUANTITATIVE PUBLICATION                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

  1. Day-to-day reproducibility (CV < 15-20%)
  2. Good linear correlation (R² > 0.99) for rate vs [catalyst]
  3. k_uncat matches expected value (~0.15 /s)
  4. Amplitude consistency within days
  5. Sufficient replicates (n ≥ 3 independent days)
""")

    if len(date_results) >= 2:
        slopes = [r['slope'] for r in date_results]
        cv_slope = 100 * np.std(slopes, ddof=1) / np.mean(slopes)
        mean_r2 = np.mean([r['r2'] for r in date_results])

        print(f"  Your data:")
        print(f"    Day-to-day CV: {cv_slope:.1f}%", end="")
        print(f" {'✓' if cv_slope < 20 else '✗'}")

        print(f"    Mean R²: {mean_r2:.4f}", end="")
        print(f" {'✓' if mean_r2 > 0.99 else '~' if mean_r2 > 0.95 else '✗'}")

        print(f"    Independent days: {len(date_results)}", end="")
        print(f" {'✓' if len(date_results) >= 3 else '~'}")

        amp_cv = 100 * np.std(date_means) / np.mean(date_means)
        print(f"    Amplitude CV (between days): {amp_cv:.1f}%", end="")
        print(f" {'✓' if amp_cv < 15 else '~' if amp_cv < 25 else '✗'}")

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  VERDICT                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        if cv_slope < 15 and mean_r2 > 0.99 and len(date_results) >= 3:
            print("  ✓ PUBLISHABLE as quantitative data")
            print(f"    Report: slope = {np.mean(slopes):.4f} ± {np.std(slopes, ddof=1):.4f} (n={len(date_results)} days)")
        elif cv_slope < 25 and mean_r2 > 0.95:
            print("  ~ PUBLISHABLE with caveats")
            print("    Report uncertainty honestly; consider 'semi-quantitative'")
        else:
            print("  ✗ NOT recommended for quantitative claims")
            print("    Consider as qualitative/comparative data only")


if __name__ == "__main__":
    main()
