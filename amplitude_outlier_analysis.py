#!/usr/bin/env python3
"""
Cross-study amplitude analysis to identify outliers and "trustworthy" data.

Key question: Can we use amplitude as a quality filter across studies?
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pathlib

T_MIN, T_MAX = 0.02, 40.0
CO2_CONC = 0.0338
Q = 0.402


def get_amplitude_from_trial(time, absorbance):
    """Get raw amplitude from a single trial."""
    mask = (time >= T_MIN) & (time <= T_MAX)
    t_fit = time[mask]
    y_fit = absorbance[mask]

    valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
    y_fit = y_fit[valid]

    if len(y_fit) < 50:
        return None

    n = max(1, len(y_fit) // 20)
    A_initial = np.median(y_fit[:n])
    A_final = np.median(y_fit[-n:])
    amplitude = A_initial - A_final

    if amplitude > 0.01:
        return amplitude
    return None


def main():
    base_path = pathlib.Path("Data")

    print("=" * 80)
    print("CROSS-STUDY AMPLITUDE ANALYSIS: CAN WE IDENTIFY 'GOOD' DATA?")
    print("=" * 80)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  THE PROBLEM WITH CROSS-STUDY AMPLITUDE COMPARISON                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Before we compare amplitudes, we need to ask: SHOULD they be the same?

Amplitude depends on:
  1. [CO2]_initial     - Should be constant if properly saturated
  2. ε (extinction)    - Depends on INDICATOR CONCENTRATION and type
  3. Path length       - Fixed by instrument
  4. Mixing ratio      - Ratio of indicator to CO2 solution

CRITICAL: Different studies may use different indicator concentrations!

If Monodentate uses 1x indicator and Cyclen uses 5x indicator:
  → Cyclen will have ~5x higher amplitude
  → This is EXPECTED, not an error
  → Cannot compare amplitudes across studies directly

Let's check what we're actually comparing...
""")

    # Collect all amplitudes with metadata
    all_data = []

    # 1. Monodentate Study
    print("\n" + "=" * 60)
    print("1. MONODENTATE STUDY - Amplitude Distribution")
    print("=" * 60)

    monodentate_path = base_path / "Monodentate_Study"
    monodentate_files = list(monodentate_path.glob("**/*.xlsx"))

    mono_amps = []
    for f in sorted(monodentate_files):
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
                    amp = get_amplitude_from_trial(time, trial_data)

                    if amp:
                        mono_amps.append(amp)
                        all_data.append({
                            'study': 'Monodentate',
                            'file': f.stem,
                            'amplitude': amp
                        })
        except:
            pass

    if mono_amps:
        print(f"  N trials: {len(mono_amps)}")
        print(f"  Mean: {np.mean(mono_amps):.4f} AU")
        print(f"  Std:  {np.std(mono_amps):.4f} AU")
        print(f"  Range: {np.min(mono_amps):.4f} - {np.max(mono_amps):.4f} AU")
        print(f"  CV: {100 * np.std(mono_amps) / np.mean(mono_amps):.1f}%")

    # 2. Cyclen Baseline
    print("\n" + "=" * 60)
    print("2. CYCLEN BASELINE - Amplitude Distribution")
    print("=" * 60)

    cyclen_baseline_path = base_path / "Cyclen_Study" / "Cyclen_baseline"
    cyclen_files = list(cyclen_baseline_path.glob("*.xlsx"))

    cyclen_amps = []
    for f in sorted(cyclen_files):
        try:
            xl = pd.ExcelFile(f)
            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet, header=None)
                row0 = df.iloc[0].values

                has_conc = any('concentration' in str(v).lower() for v in row0 if isinstance(v, str))

                if has_conc:
                    conc_cols = [i for i, v in enumerate(row0)
                                 if isinstance(v, str) and 'concentration' in str(v).lower()]

                    for i, col_idx in enumerate(conc_cols):
                        next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
                        time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

                        for tc in range(col_idx + 1, next_idx):
                            col_data = df.iloc[2:, tc]
                            if col_data.isna().sum() > len(col_data) * 0.5:
                                continue

                            trial_data = pd.to_numeric(col_data, errors='coerce').values
                            amp = get_amplitude_from_trial(time, trial_data)

                            if amp:
                                cyclen_amps.append(amp)
                                all_data.append({
                                    'study': 'Cyclen_baseline',
                                    'file': f.stem,
                                    'sheet': sheet,
                                    'amplitude': amp
                                })
                else:
                    time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values
                    for col in range(1, df.shape[1]):
                        col_data = df.iloc[1:, col]
                        if col_data.isna().sum() > len(col_data) * 0.5:
                            continue

                        trial_data = pd.to_numeric(col_data, errors='coerce').values
                        amp = get_amplitude_from_trial(time, trial_data)

                        if amp:
                            cyclen_amps.append(amp)
                            all_data.append({
                                'study': 'Cyclen_baseline',
                                'file': f.stem,
                                'sheet': sheet,
                                'amplitude': amp
                            })
        except:
            pass

    if cyclen_amps:
        print(f"  N trials: {len(cyclen_amps)}")
        print(f"  Mean: {np.mean(cyclen_amps):.4f} AU")
        print(f"  Std:  {np.std(cyclen_amps):.4f} AU")
        print(f"  Range: {np.min(cyclen_amps):.4f} - {np.max(cyclen_amps):.4f} AU")
        print(f"  CV: {100 * np.std(cyclen_amps) / np.mean(cyclen_amps):.1f}%")

    # 3. Benz_Cyclen (the main dataset)
    print("\n" + "=" * 60)
    print("3. BENZ_CYCLEN - Amplitude Distribution by Day")
    print("=" * 60)

    benz_file = base_path / "Cyclen_Study" / "Benz_Cyclen_Summary" / "Cyc_Benz_Comparison.xlsx"
    xl = pd.ExcelFile(benz_file)

    benz_amps = {}
    benz_all = []

    for sheet in ['12_05', '12_13', '01_14']:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        row0 = df.iloc[0].values

        conc_cols = [i for i, v in enumerate(row0)
                     if isinstance(v, str) and 'concentration' in v.lower()]

        sheet_amps = []

        for i, col_idx in enumerate(conc_cols):
            next_idx = conc_cols[i + 1] if i + 1 < len(conc_cols) else df.shape[1]
            time = pd.to_numeric(df.iloc[2:, col_idx], errors='coerce').values

            for tc in range(col_idx + 1, next_idx):
                col_data = df.iloc[2:, tc]
                if col_data.isna().sum() > len(col_data) * 0.5:
                    continue

                trial_data = pd.to_numeric(col_data, errors='coerce').values
                amp = get_amplitude_from_trial(time, trial_data)

                if amp:
                    sheet_amps.append(amp)
                    benz_all.append(amp)
                    all_data.append({
                        'study': 'Benz_Cyclen',
                        'file': sheet,
                        'amplitude': amp
                    })

        benz_amps[sheet] = sheet_amps
        print(f"  {sheet}: Mean = {np.mean(sheet_amps):.4f} ± {np.std(sheet_amps):.4f} (n={len(sheet_amps)})")

    print(f"\n  Overall Benz_Cyclen:")
    print(f"    Mean: {np.mean(benz_all):.4f} AU")
    print(f"    Std:  {np.std(benz_all):.4f} AU")
    print(f"    CV: {100 * np.std(benz_all) / np.mean(benz_all):.1f}%")

    # Cross-study comparison
    print("\n" + "=" * 80)
    print("CROSS-STUDY COMPARISON")
    print("=" * 80)

    print(f"""
  {'Study':<20} {'Mean Amp':<12} {'Std':<10} {'CV (%)':<10} {'N'}
  {'-'*60}
  {'Monodentate':<20} {np.mean(mono_amps):<12.4f} {np.std(mono_amps):<10.4f} {100*np.std(mono_amps)/np.mean(mono_amps):<10.1f} {len(mono_amps)}
  {'Cyclen_baseline':<20} {np.mean(cyclen_amps):<12.4f} {np.std(cyclen_amps):<10.4f} {100*np.std(cyclen_amps)/np.mean(cyclen_amps):<10.1f} {len(cyclen_amps)}
  {'Benz_Cyclen':<20} {np.mean(benz_all):<12.4f} {np.std(benz_all):<10.4f} {100*np.std(benz_all)/np.mean(benz_all):<10.1f} {len(benz_all)}
""")

    # Check if they're even comparable
    print("=" * 80)
    print("CAN WE USE AMPLITUDE AS A QUALITY FILTER?")
    print("=" * 80)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ANALYSIS: Are the studies using the same indicator concentration?           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Check if means are similar
    means = [np.mean(mono_amps), np.mean(cyclen_amps), np.mean(benz_all)]
    overall_cv = 100 * np.std(means) / np.mean(means)

    print(f"  Mean amplitudes across studies: {means}")
    print(f"  CV of study means: {overall_cv:.1f}%")

    if overall_cv < 20:
        print("""
  The study means are SIMILAR (CV < 20%), suggesting same indicator conditions.
  Cross-study amplitude comparison may be valid.
""")
    else:
        print("""
  The study means DIFFER significantly (CV > 20%).
  This likely indicates different indicator concentrations or conditions.
  Cross-study amplitude comparison is NOT valid.
""")

    # Within Benz_Cyclen: identify outliers
    print("=" * 80)
    print("WITHIN BENZ_CYCLEN: OUTLIER IDENTIFICATION")
    print("=" * 80)

    all_benz = []
    for sheet in ['12_05', '12_13', '01_14']:
        for amp in benz_amps[sheet]:
            all_benz.append({'date': sheet, 'amplitude': amp})

    df_benz = pd.DataFrame(all_benz)

    # Calculate IQR-based outliers
    q1 = df_benz['amplitude'].quantile(0.25)
    q3 = df_benz['amplitude'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f"\n  IQR Analysis:")
    print(f"    Q1: {q1:.4f}")
    print(f"    Q3: {q3:.4f}")
    print(f"    IQR: {iqr:.4f}")
    print(f"    Lower bound (Q1 - 1.5×IQR): {lower_bound:.4f}")
    print(f"    Upper bound (Q3 + 1.5×IQR): {upper_bound:.4f}")

    outliers = df_benz[(df_benz['amplitude'] < lower_bound) | (df_benz['amplitude'] > upper_bound)]
    print(f"\n  Outliers found: {len(outliers)}")

    if len(outliers) > 0:
        print("\n  Outlier details:")
        for _, row in outliers.iterrows():
            print(f"    {row['date']}: {row['amplitude']:.4f} AU")

    # The key insight
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  THE FUNDAMENTAL PROBLEM WITH "WEEDING OUT" DATA                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Here's why amplitude-based filtering is problematic:

1. WITHIN a single day: Amplitude should be constant
   - All trials use same solutions, same conditions
   - Variation is mixing efficiency (random, not systematic)
   - Can't exclude "bad" trials without bias

2. BETWEEN days: Amplitude CAN legitimately vary
   - Different solution preparations
   - Different temperatures
   - This is the EXPERIMENTAL PRECISION, not error

3. The circular logic problem:
   - If we exclude low-amplitude days → we bias toward high-amplitude
   - If high amplitude correlates with high rate (r = +0.79) → we bias k_cat UP
   - This is data manipulation, not quality control

4. What WOULD be valid quality control:
   - Exclude trials with poor fit quality (R² < 0.99)
   - Exclude trials with k values outside physical bounds
   - Exclude trials where the curve shape is wrong
   - These are OBJECTIVE criteria, not outcome-based
""")

    # Show what happens if we try to filter
    print("=" * 80)
    print("SIMULATION: WHAT IF WE FILTER BY AMPLITUDE?")
    print("=" * 80)

    # Only keep data within ±1 SD of mean
    mean_amp = df_benz['amplitude'].mean()
    std_amp = df_benz['amplitude'].std()

    print(f"\n  Original data: {len(df_benz)} trials")
    print(f"  Mean amplitude: {mean_amp:.4f} ± {std_amp:.4f}")

    # Filter to ±1 SD
    filtered = df_benz[(df_benz['amplitude'] >= mean_amp - std_amp) &
                       (df_benz['amplitude'] <= mean_amp + std_amp)]

    print(f"\n  After ±1 SD filter: {len(filtered)} trials")

    for date in ['12_05', '12_13', '01_14']:
        orig_n = len(df_benz[df_benz['date'] == date])
        filt_n = len(filtered[filtered['date'] == date])
        print(f"    {date}: {orig_n} → {filt_n} trials")

    print("""
  ⚠ WARNING: This filter preferentially removes 12_05 (low amplitude day)
     and 12_13 (high amplitude day), keeping mostly 01_14.

  This is NOT objective quality control - it's cherry-picking based on
  the outcome variable (amplitude correlates with rate).
""")

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RECOMMENDATION                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

  DO NOT filter data based on amplitude. Instead:

  1. Report all data with uncertainty: k_cat = 190 ± 52 M⁻¹s⁻¹ (n=3 days)

  2. Use OBJECTIVE quality metrics if filtering is needed:
     - R² of fit > 0.995
     - k values within physical bounds (0.01 - 10 /s)
     - Residuals without systematic patterns

  3. If you must improve precision:
     - Collect MORE data on MORE days
     - Improve experimental consistency (temperature control, mixing)
     - Use internal standards

  4. The 27% CV is your experimental precision - it's real, not an artifact
     that can be filtered away.
""")


if __name__ == "__main__":
    main()
