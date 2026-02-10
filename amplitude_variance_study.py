#!/usr/bin/env python3
"""
Amplitude variance study across all datasets:
1. Monodentate_Study - multiple concentrations, different days
2. Cyclen_baseline - baseline measurements
3. Benz_Cyclen (for comparison)
"""

import numpy as np
import pandas as pd
import pathlib
from datetime import datetime

T_MIN, T_MAX = 0.02, 40.0


def get_amplitude_from_sheet(df, sheet_type='multi_conc'):
    """Extract raw amplitude (A_initial - A_final) from data."""
    amplitudes = []

    if sheet_type == 'multi_conc':
        # Format with concentration groups
        row0 = df.iloc[0].values
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

                # Apply time window
                mask = (time >= T_MIN) & (time <= T_MAX)
                t_fit = time[mask]
                y_fit = trial_data[mask]

                valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
                y_fit = y_fit[valid]

                if len(y_fit) < 50:
                    continue

                n = max(1, len(y_fit) // 20)
                A_initial = np.median(y_fit[:n])
                A_final = np.median(y_fit[-n:])
                amplitude = A_initial - A_final

                if amplitude > 0.01:  # Filter out noise
                    amplitudes.append(amplitude)

    elif sheet_type == 'single':
        # Simple format: time in col 0, trials in subsequent columns
        time = pd.to_numeric(df.iloc[1:, 0], errors='coerce').values

        for col in range(1, df.shape[1]):
            col_data = df.iloc[1:, col]
            if col_data.isna().sum() > len(col_data) * 0.5:
                continue

            # Check if it's a Q value column
            header = df.iloc[0, col] if col < len(df.iloc[0]) else None
            if isinstance(header, str) and 'Q' in str(header).upper():
                continue

            trial_data = pd.to_numeric(col_data, errors='coerce').values

            mask = (time >= T_MIN) & (time <= T_MAX)
            t_fit = time[mask]
            y_fit = trial_data[mask]

            valid = ~np.isnan(t_fit) & ~np.isnan(y_fit)
            y_fit = y_fit[valid]

            if len(y_fit) < 50:
                continue

            n = max(1, len(y_fit) // 20)
            A_initial = np.median(y_fit[:n])
            A_final = np.median(y_fit[-n:])
            amplitude = A_initial - A_final

            if amplitude > 0.01:
                amplitudes.append(amplitude)

    return amplitudes


def main():
    base_path = pathlib.Path("Data")

    print("=" * 80)
    print("AMPLITUDE VARIANCE STUDY ACROSS ALL DATASETS")
    print("=" * 80)

    all_results = []

    # 1. Monodentate Study
    print("\n" + "=" * 60)
    print("1. MONODENTATE STUDY")
    print("=" * 60)

    monodentate_path = base_path / "Monodentate_Study"
    monodentate_files = list(monodentate_path.glob("**/*.xlsx"))

    for f in sorted(monodentate_files):
        try:
            xl = pd.ExcelFile(f)
            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet, header=None)
                amplitudes = get_amplitude_from_sheet(df, 'single')

                if amplitudes:
                    mean_amp = np.mean(amplitudes)
                    std_amp = np.std(amplitudes, ddof=1) if len(amplitudes) > 1 else 0

                    # Extract date from path
                    date_str = f.parent.name if '202' in f.parent.name else f.stem

                    print(f"  {f.stem}: ΔA = {mean_amp:.4f} ± {std_amp:.4f} (n={len(amplitudes)})")

                    all_results.append({
                        'dataset': 'Monodentate',
                        'file': f.stem,
                        'date': date_str,
                        'mean_amplitude': mean_amp,
                        'std_amplitude': std_amp,
                        'n_trials': len(amplitudes)
                    })
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    # 2. Cyclen Baseline
    print("\n" + "=" * 60)
    print("2. CYCLEN BASELINE")
    print("=" * 60)

    cyclen_baseline_path = base_path / "Cyclen_Study" / "Cyclen_baseline"
    cyclen_files = list(cyclen_baseline_path.glob("*.xlsx"))

    for f in sorted(cyclen_files):
        try:
            xl = pd.ExcelFile(f)
            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet, header=None)

                # Try multi_conc format first
                row0 = df.iloc[0].values
                has_conc = any('concentration' in str(v).lower() for v in row0 if isinstance(v, str))

                if has_conc:
                    amplitudes = get_amplitude_from_sheet(df, 'multi_conc')
                else:
                    amplitudes = get_amplitude_from_sheet(df, 'single')

                if amplitudes:
                    mean_amp = np.mean(amplitudes)
                    std_amp = np.std(amplitudes, ddof=1) if len(amplitudes) > 1 else 0

                    print(f"  {f.stem} ({sheet}): ΔA = {mean_amp:.4f} ± {std_amp:.4f} (n={len(amplitudes)})")

                    all_results.append({
                        'dataset': 'Cyclen_baseline',
                        'file': f.stem,
                        'date': sheet,
                        'mean_amplitude': mean_amp,
                        'std_amplitude': std_amp,
                        'n_trials': len(amplitudes)
                    })
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    # 3. Benz_Cyclen (for comparison)
    print("\n" + "=" * 60)
    print("3. BENZ_CYCLEN (Reference)")
    print("=" * 60)

    benz_cyclen_file = base_path / "Cyclen_Study" / "Benz_Cyclen_Summary" / "Cyc_Benz_Comparison.xlsx"

    try:
        xl = pd.ExcelFile(benz_cyclen_file)
        for sheet in xl.sheet_names:
            if sheet.lower() == 'uncatalyzed':
                continue
            df = pd.read_excel(xl, sheet_name=sheet, header=None)
            amplitudes = get_amplitude_from_sheet(df, 'multi_conc')

            if amplitudes:
                mean_amp = np.mean(amplitudes)
                std_amp = np.std(amplitudes, ddof=1) if len(amplitudes) > 1 else 0

                print(f"  Benz_Cyclen ({sheet}): ΔA = {mean_amp:.4f} ± {std_amp:.4f} (n={len(amplitudes)})")

                all_results.append({
                    'dataset': 'Benz_Cyclen',
                    'file': 'Cyc_Benz_Comparison',
                    'date': sheet,
                    'mean_amplitude': mean_amp,
                    'std_amplitude': std_amp,
                    'n_trials': len(amplitudes)
                })
    except Exception as e:
        print(f"  Error: {e}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY BY DATASET")
    print("=" * 80)

    df_results = pd.DataFrame(all_results)

    for dataset in df_results['dataset'].unique():
        subset = df_results[df_results['dataset'] == dataset]
        amps = subset['mean_amplitude'].values

        if len(amps) >= 2:
            mean_amp = np.mean(amps)
            std_amp = np.std(amps, ddof=1)
            cv = 100 * std_amp / mean_amp
            min_amp = np.min(amps)
            max_amp = np.max(amps)
            range_pct = 100 * (max_amp - min_amp) / mean_amp

            print(f"\n{dataset}:")
            print(f"  N measurements: {len(amps)}")
            print(f"  Mean amplitude: {mean_amp:.4f} AU")
            print(f"  Std amplitude:  {std_amp:.4f} AU")
            print(f"  CV:             {cv:.1f}%")
            print(f"  Range:          {min_amp:.4f} - {max_amp:.4f} ({range_pct:.1f}%)")

    # Cross-dataset comparison
    print("\n" + "=" * 80)
    print("CROSS-DATASET AMPLITUDE COMPARISON")
    print("=" * 80)

    for dataset in df_results['dataset'].unique():
        subset = df_results[df_results['dataset'] == dataset]
        print(f"\n{dataset}:")
        for _, row in subset.iterrows():
            print(f"  {row['date']}: {row['mean_amplitude']:.4f} AU")


if __name__ == "__main__":
    main()
