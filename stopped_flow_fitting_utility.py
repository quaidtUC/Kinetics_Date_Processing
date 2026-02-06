#!/usr/bin/env python3
"""
Stopped-flow spectrometer fitting utility
=========================================
Fits single or double exponential decays to each absorbance trace in a CSV or
XLSX file, plots the result, and stores the fitted parameters in a summary CSV.

Install requirements once:
    pip install pandas numpy scipy matplotlib openpyxl
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# File-chooser (optional GUI)
try:
    from tkinter import Tk, filedialog      # type: ignore
except ImportError:
    filedialog = None                       # headless environment


# ────────────────────────── Exponential models ───────────────────────── #

def single_exp(t: np.ndarray, A0: float, A: float, k: float) -> np.ndarray:
    return A0 + A * np.exp(-k * t)


def single_dydt0(A: float, k: float) -> float:
    return -A * k


def double_exp(t: np.ndarray, A0: float, A1: float, k1: float,
               A2: float, k2: float) -> np.ndarray:
    return A0 + A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)


def double_dydt0(A1: float, k1: float, A2: float, k2: float) -> float:
    return -A1 * k1 - A2 * k2


# ────────────────────────── File helpers ─────────────────────────────── #

def ask_for_file() -> pathlib.Path:
    if filedialog is not None:
        root = Tk()
        root.withdraw()
        fname = filedialog.askopenfilename(
            title="Select data file (CSV or XLSX)",
            filetypes=[("Data files", "*.csv *.xlsx"),
                       ("CSV files", "*.csv"),
                       ("Excel files", "*.xlsx"),
                       ("All files", "*.*")]
        )
        root.destroy()
        if not fname:
            raise SystemExit("No file selected. Exiting.")
        return pathlib.Path(fname)

    return pathlib.Path(input("Enter path to CSV/XLSX file: ").strip())


def load_dataframe(path: pathlib.Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".xlsx":
        try:
            return pd.read_excel(path, sheet_name=0)
        except ImportError as exc:
            raise RuntimeError("Install openpyxl to read .xlsx files "
                               "(pip install openpyxl)") from exc
    raise ValueError(f"Unsupported file type: {ext}")


# ────────────────────────── Fitting routines ─────────────────────────── #

def fit_trace(time: np.ndarray, y: np.ndarray, model: str
              ) -> tuple[np.ndarray, np.ndarray, float, Callable]:
    if model == "single":
        func = single_exp
        p0 = [y.min(), y.max() - y.min(),
              1.0 / max(time[np.argmax(np.diff(y))], 1e-6)]
    else:
        func = double_exp
        half_range = (y.max() - y.min()) / 2
        p0 = [y.min(), half_range, 1.0, half_range, 0.1]

    popt, pcov = curve_fit(func, time, y, p0=p0, maxfev=50_000)
    perr = np.sqrt(np.diag(pcov))

    dydt0 = (single_dydt0(popt[1], popt[2]) if model == "single"
             else double_dydt0(popt[1], popt[2], popt[3], popt[4]))
    return popt, perr, dydt0, func


def process_file(data_path: pathlib.Path, model: str,
                 selected_cols: List[int] | None = None) -> None:
    df = load_dataframe(data_path)
    if df.empty or df.shape[1] < 2:
        raise ValueError("File must contain at least two columns "
                         "(time + ≥1 absorbance).")

    time = df.iloc[:, 0].to_numpy(float)
    cols = selected_cols if selected_cols else list(range(1, df.shape[1]))

    summary_records: List[dict] = []
    all_dydt0: List[float] = []           # ← NEW

    for idx in cols:
        col_name = str(df.columns[idx])
        y = df.iloc[:, idx].to_numpy(float)

        popt, perr, dydt0, func = fit_trace(time, y, model)

        all_dydt0.append(dydt0)           # ← NEW

        # ── console output ──────────────────────────────────────────────
        print(f"\nTrace '{col_name}' (model: {model})")
        if model == "single":
            labels = ["A0", "A", "k"]
        else:
            labels = ["A0", "A1", "k1", "A2", "k2"]
        for lbl, p, e in zip(labels, popt, perr):
            print(f"  {lbl:<4}= {p:.6g} ± {e:.2g}")
        print(f"  dA/dt @ t=0 = {dydt0:.6g} AU/s")

        # record for summary CSV
        rec = {"trace": col_name, "model": model, "dA_dt_t0": dydt0}
        for lbl, p, e in zip(labels, popt, perr):
            rec[f"{lbl}"] = p
            rec[f"{lbl}_err"] = e
        summary_records.append(rec)

        # ── plotting ────────────────────────────────────────────────────
        t_fit = np.linspace(time.min(), time.max(), 1_000)
        y_fit = func(t_fit, *popt)

        plt.figure(figsize=(5, 3))
        plt.scatter(time, y, s=10, color="#7EC8E3", label="data", zorder=3)
        plt.plot(t_fit, y_fit, color="black", linewidth=1.5,
                 label="fit", zorder=4)
        plt.xlabel("Time (s)")
        plt.ylabel("Absorbance (AU)")
        plt.title(f"{col_name} — {model} exponential fit")
        plt.legend()
        plt.tight_layout()

        safe_col = "".join(c if c.isalnum() else "_" for c in col_name)
        out_png = data_path.with_name(
            f"{data_path.stem}_{safe_col}_{model}_fit.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"  ⤷ plot saved to {out_png}")

    # ── NEW: compute average derivative ──────────────────────────────────
    if all_dydt0:
        mean_dydt0 = float(np.mean(all_dydt0))
        std_dydt0  = float(np.std(all_dydt0, ddof=1))  # sample σ
        print(f"\nAverage dA/dt @ t=0 = {mean_dydt0:.6g} ± {std_dydt0:.2g} AU/s")

    # ── write summary CSV ───────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_records)

    # Append the average as an extra row
    if all_dydt0:
        avg_row = {
            "trace": "AVG",
            "model": model,
            "dA_dt_t0": mean_dydt0,
            "dA_dt_t0_err": std_dydt0
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])],
                               ignore_index=True)

    out_csv = data_path.with_name(f"{data_path.stem}_fit_results.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSummary written to {out_csv}")


# ────────────────────────── CLI boilerplate ────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit stopped-flow data (CSV or XLSX) with single or "
                    "double exponential models."
    )
    p.add_argument("data", nargs="?", type=pathlib.Path,
                   help="Path to CSV or XLSX file.")
    p.add_argument("--model",
                choices=["single","double"], default="double",
                help="Model to fit (default: double).")
    p.add_argument("--cols", type=int, nargs="*",
                   help="1-based indices of absorbance columns to fit "
                        "(default: all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data if args.data is not None else ask_for_file()
    col_idx = [c - 1 for c in args.cols] if args.cols else None
    process_file(data_path, args.model, col_idx)


if __name__ == "__main__":
    main()