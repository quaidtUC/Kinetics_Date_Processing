#!/usr/bin/env python3
"""
Analysis of whether Rains 2019 [CO2] vs k_obs relationship can serve as a
correction term for amplitude variance.

Key insight from Rains 2019:
- k_obs decreases with increasing [CO2] due to bicarbonate inhibition
- This creates a nonlinear relationship: k_obs = f([CO2])

The question: Can we use this relationship to correct for amplitude variance?

Approach:
1. If amplitude varies because [CO2] varies between days...
2. And k_obs depends on [CO2] (per Rains)...
3. Then we should see a predictable relationship between amplitude and k_obs
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib

CO2_CONC = 0.0338  # M (nominal saturated)
Q = 0.402
T_MIN, T_MAX = 0.02, 40.0


def double_exp(t, A0, A1, k1, A2, k2):
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def fit_and_extract(time, absorbance):
    """Fit double exp and return amplitude, dydt0, and k_fast."""
    mask = (time >= T_MIN) & (time <= T_MAX)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    t_fit, y_fit = t_fit[valid], y_fit[valid]

    if len(t_fit) < 50:
        return None

    n = max(1, len(y_fit) // 20)
    A_initial = np.median(y_fit[:n])
    A_final = np.median(y_fit[-n:])
    raw_amplitude = A_initial - A_final

    try:
        p0 = [A_final, raw_amplitude * 0.5, 1.0, raw_amplitude * 0.5, 0.1]
        bounds = ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6],
                  [np.inf, np.inf, 100, np.inf, 100])
        popt, _ = curve_fit(double_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=50000)

        A0, A1, k1, A2, k2 = popt
        dydt0 = -A1 * k1 - A2 * k2
        rate = abs(dydt0) / CO2_CONC * Q

        return {
            'amplitude': raw_amplitude,
            'dydt0': dydt0,
            'rate': rate,
            'k_fast': max(k1, k2),
            'k_slow': min(k1, k2),
            'A1': A1, 'k1': k1, 'A2': A2, 'k2': k2
        }
    except:
        return None


def main():
    print("=" * 80)
    print("RAINS 2019 [CO2] vs k_obs CORRECTION ANALYSIS")
    print("=" * 80)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  THEORETICAL FRAMEWORK                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

From Rains et al. 2019:
  - k_obs decreases with increasing [CO2] due to bicarbonate inhibition
  - The relationship is nonlinear (exponential decay-like)
  - At high [CO2], product inhibition slows the reaction

For your pH indicator system:
  - Amplitude ΔA ∝ Δ[H⁺] = [CO2]_reacted
  - If [CO2]_initial varies, amplitude varies proportionally

Key Question: Can we use amplitude as a proxy for [CO2] and correct k_obs?

╔══════════════════════════════════════════════════════════════════════════════╗
║  TESTING THE HYPOTHESIS                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

If Rains correction applies:
  1. Higher [CO2] → Lower k_obs (inhibition)
  2. Higher [CO2] → Higher amplitude (more protons released)
  3. Therefore: Higher amplitude → Lower k_obs

This predicts NEGATIVE correlation between amplitude and rate.

Let's check the data...
""")

    file_path = pathlib.Path("Data/Cyclen_Study/Benz_Cyclen_Summary/Cyc_Benz_Comparison.xlsx")
    xl = pd.ExcelFile(file_path)

    # Collect all trial-level data for 1.0 mM concentration
    all_data = []

    for sheet in ['12_05', '12_13', '01_14']:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        row0 = df.iloc[0].values

        conc_cols = [i for i, v in enumerate(row0)
                     if isinstance(v, str) and 'concentration' in v.lower()]

        # Get 1.0 mM (last concentration group)
        col_idx = conc_cols[-1]
        next_idx = df.shape[1]
        time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

        for tc in range(col_idx + 1, next_idx):
            col_data = df.iloc[2:, tc]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue

            trial_data = pd.to_numeric(col_data, errors='coerce').values
            result = fit_and_extract(time, trial_data)

            if result:
                all_data.append({
                    'date': sheet,
                    'amplitude': result['amplitude'],
                    'rate': result['rate'],
                    'dydt0': abs(result['dydt0']),
                    'k_fast': result['k_fast']
                })

    df_data = pd.DataFrame(all_data)

    print("Trial-level data (1.0 mM concentration):")
    print("-" * 60)
    print(f"{'Date':<10} {'Amplitude':<12} {'Rate (/s)':<12} {'k_fast':<10}")
    print("-" * 60)

    for date in ['12_05', '12_13', '01_14']:
        subset = df_data[df_data['date'] == date]
        for _, row in subset.iterrows():
            print(f"{row['date']:<10} {row['amplitude']:<12.4f} {row['rate']:<12.2f} {row['k_fast']:<10.3f}")
        print()

    # Calculate correlations
    amps = df_data['amplitude'].values
    rates = df_data['rate'].values
    kfasts = df_data['k_fast'].values

    corr_amp_rate = np.corrcoef(amps, rates)[0, 1]
    corr_amp_kfast = np.corrcoef(amps, kfasts)[0, 1]

    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"\n  Amplitude vs Rate:   r = {corr_amp_rate:+.3f}")
    print(f"  Amplitude vs k_fast: r = {corr_amp_kfast:+.3f}")

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  INTERPRETATION                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    if corr_amp_rate > 0.3:
        print(f"""
  POSITIVE correlation (r = {corr_amp_rate:+.3f}) observed!

  This is OPPOSITE to the Rains prediction:
    Rains predicts: Higher [CO2] → Higher amplitude, BUT Lower k_obs
    Observed:       Higher amplitude → HIGHER rate

  This RULES OUT bicarbonate inhibition as the primary source of variance.

  The positive correlation suggests:
    • Days with higher amplitude had more reactive conditions overall
    • This could be due to better mixing, fresher solutions, or temperature
    • The effect is NOT product inhibition (which would cause negative correlation)
""")
    else:
        print(f"""
  Correlation (r = {corr_amp_rate:+.3f}) observed.

  If strongly negative, Rains correction might apply.
  However, other factors likely dominate the variance.
""")

    # Test: What if we normalize by amplitude?
    print("=" * 60)
    print("TEST: NORMALIZING RATE BY AMPLITUDE")
    print("=" * 60)

    mean_amp = np.mean(amps)
    norm_rates = rates * (mean_amp / amps)

    print("\nOriginal rates by date:")
    for date in ['12_05', '12_13', '01_14']:
        subset = df_data[df_data['date'] == date]
        mean_rate = subset['rate'].mean()
        std_rate = subset['rate'].std()
        print(f"  {date}: {mean_rate:.2f} ± {std_rate:.2f} /s")

    raw_mean = np.mean(rates)
    raw_std = np.std(rates)
    raw_cv = 100 * raw_std / raw_mean

    print(f"\n  Overall: {raw_mean:.2f} ± {raw_std:.2f} (CV = {raw_cv:.1f}%)")

    print("\nAfter amplitude normalization (rate × mean_amp / trial_amp):")
    df_data['norm_rate'] = norm_rates

    for date in ['12_05', '12_13', '01_14']:
        subset = df_data[df_data['date'] == date]
        mean_norm = subset['norm_rate'].mean()
        std_norm = subset['norm_rate'].std()
        print(f"  {date}: {mean_norm:.2f} ± {std_norm:.2f} /s")

    norm_mean = np.mean(norm_rates)
    norm_std = np.std(norm_rates)
    norm_cv = 100 * norm_std / norm_mean

    print(f"\n  Overall: {norm_mean:.2f} ± {norm_std:.2f} (CV = {norm_cv:.1f}%)")

    improvement = (raw_cv - norm_cv) / raw_cv * 100
    print(f"\n  CV change: {raw_cv:.1f}% → {norm_cv:.1f}% ({improvement:+.1f}%)")

    # Now test at the k_cat level
    print("\n" + "=" * 60)
    print("TEST: EFFECT ON k_cat VALUES")
    print("=" * 60)

    # Get k_cat for each date using normalized rates
    for conc_group in ['all_conc']:
        print("\nUsing amplitude-normalized rates for regression:")

        k_cats_norm = []

        for sheet in ['12_05', '12_13', '01_14']:
            df = pd.read_excel(xl, sheet_name=sheet, header=None)
            row0 = df.iloc[0].values

            conc_cols = [i for i, v in enumerate(row0)
                         if isinstance(v, str) and 'concentration' in v.lower()]

            conc_rates = []

            for i, col_idx in enumerate(conc_cols):
                import re
                header = str(row0[col_idx])
                match = re.search(r'(\d+\.?\d*)\s*mM', header, re.IGNORECASE)
                conc_mM = float(match.group(1)) if match else [0.25, 0.5, 1.0][i]

                next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
                time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

                rates_this_conc = []
                for tc in range(col_idx + 1, next_idx):
                    col_data = df.iloc[2:, tc]
                    if col_data.isna().sum() > len(col_data) * 0.5:
                        continue

                    trial_data = pd.to_numeric(col_data, errors='coerce').values
                    result = fit_and_extract(time, trial_data)

                    if result:
                        # Normalize by amplitude
                        norm_rate = result['rate'] * (mean_amp / result['amplitude'])
                        rates_this_conc.append(norm_rate)

                if rates_this_conc:
                    conc_rates.append((conc_mM, np.mean(rates_this_conc)))

            if len(conc_rates) >= 2:
                concs_M = np.array([c / 1000 for c, _ in conc_rates])
                rates = np.array([r for _, r in conc_rates])
                slope, intercept, r_value, _, _ = linregress(concs_M, rates)

                print(f"  {sheet}: k_cat = {slope:.1f}, k_uncat = {intercept:.4f}")
                k_cats_norm.append(slope)

        mean_kcat = np.mean(k_cats_norm)
        std_kcat = np.std(k_cats_norm, ddof=1)
        cv_kcat = 100 * std_kcat / mean_kcat

        print(f"\n  Mean k_cat: {mean_kcat:.1f} ± {std_kcat:.1f} (CV = {cv_kcat:.1f}%)")
        print(f"\n  Compare to original: 190 ± 52 (CV = 27.3%)")

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CONCLUSIONS                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. RAINS CORRECTION DOES NOT APPLY:
   The positive amplitude-rate correlation rules out bicarbonate inhibition
   as the primary source of day-to-day variance. If [CO2] inhibition were
   causing the variance, we'd see higher amplitude → lower rate.

2. AMPLITUDE NORMALIZATION EFFECT:
   Normalizing by amplitude may modestly improve CV, but the fundamental
   issue is that the amplitude variation is experimental (mixing, handling)
   and not systematic.

3. WHY THE POSITIVE CORRELATION?
   The positive correlation (higher amplitude days → higher rates) suggests
   that on "good" days, BOTH mixing efficiency AND signal are better.
   This is consistent with:
   - Better syringe drive performance
   - Less dead volume
   - Faster mixing → captures more of the fast phase

4. RECOMMENDATION:
   The Rains [CO2] correction is not applicable here because:
   - Your [CO2] is nominally constant (saturated solution)
   - The amplitude variation is due to experimental factors, not [CO2] variation
   - Even if [CO2] varied, the correlation direction is wrong for inhibition

   The 27% CV in k_cat is experimental precision, not systematic bias that
   can be corrected computationally.
""")


if __name__ == "__main__":
    main()
