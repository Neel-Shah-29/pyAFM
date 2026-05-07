#!/usr/bin/env python3
"""Plot geometry learning curves comparing predicate thresholds for each framework."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geometry_learning_curves_three_frameworks import (
    FRAMEWORK_COLORS,
    FRAMEWORK_LABELS,
    FRAMEWORK_ORDER,
    aggregate_curves,
    kc_curve_from_counts,
    summarize_curve_collection,
)


def load_transactions(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".txt" else ","
    df = pd.read_csv(path, sep=sep)

    required = ["Training Framework", "Anon Student Id", "Outcome", "Predicate Threshold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if "KC (field)" in df.columns:
        kc_series = df["KC (field)"].fillna("")
    elif "Step Name" in df.columns:
        kc_series = df["Step Name"].fillna("")
    else:
        raise ValueError("Expected either 'KC (field)' or 'Step Name' in aggregated data.")

    df = df.copy()
    df["KC"] = kc_series.astype(str).str.strip()
    df = df[df["KC"] != ""]
    df["Outcome"] = df["Outcome"].fillna("").astype(str).str.upper()
    df["IsCorrect"] = (df["Outcome"] == "CORRECT").astype(int)
    df["Student"] = df["Anon Student Id"].fillna("unknown_student").astype(str)
    df["Training Framework"] = df["Training Framework"].astype(str)
    df["Predicate Threshold"] = pd.to_numeric(df["Predicate Threshold"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Predicate Threshold"])
    df["Predicate Threshold"] = df["Predicate Threshold"].astype(int)
    df["OppIndex"] = df.groupby(
        ["Predicate Threshold", "Training Framework", "Student", "KC"],
        sort=False,
    ).cumcount()
    return df


def build_threshold_framework_curves(
    df: pd.DataFrame,
    opp_count_cutoff: int,
    opp_cutoff: int,
    monotonic_envelope: bool = False,
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    counts = (
        df.groupby(["Predicate Threshold", "Training Framework", "Student", "KC", "OppIndex"], sort=False)
        .agg(correct=("IsCorrect", "sum"), total=("IsCorrect", "size"))
        .reset_index()
    )

    curves: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for framework in FRAMEWORK_ORDER:
        fw_counts = counts[counts["Training Framework"] == framework]
        threshold_curves: Dict[int, Dict[str, np.ndarray]] = {}
        for threshold, threshold_df in fw_counts.groupby("Predicate Threshold", sort=True):
            student_curves = []
            for _, student_df in threshold_df.groupby("Student", sort=False):
                kc_curves = {}
                for kc, kc_df in student_df.groupby("KC", sort=False):
                    curve = kc_curve_from_counts(kc_df, opp_count_cutoff, opp_cutoff)
                    if len(curve) > 0:
                        kc_curves[kc] = curve
                agg_curve = aggregate_curves(kc_curves, monotonic_envelope=monotonic_envelope)
                if len(agg_curve) > 0:
                    student_curves.append(agg_curve)
            threshold_curves[int(threshold)] = summarize_curve_collection(
                student_curves,
                monotonic_envelope=monotonic_envelope,
            )
        curves[framework] = threshold_curves
    return curves


def plot_threshold_curves(
    curves: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for framework in FRAMEWORK_ORDER:
        plt.figure(figsize=(8, 5))
        plotted = 0
        for threshold, summary in sorted(curves.get(framework, {}).items()):
            mean = summary.get("mean", np.array([], dtype=float))
            std = summary.get("std", np.array([], dtype=float))
            count = summary.get("count", np.array([], dtype=float))
            if len(mean) == 0:
                continue
            xs = np.arange(1, len(mean) + 1)
            plt.plot(
                xs,
                mean,
                linewidth=2,
                marker="o",
                label=f"Threshold {threshold}",
            )
            if len(std) > 0 and np.nanmax(count) > 1:
                stderr = np.divide(std, np.sqrt(np.maximum(count, 1)), where=count > 0)
                lower = np.clip(mean - stderr, 0.0, 1.0)
                upper = np.clip(mean + stderr, 0.0, 1.0)
                plt.fill_between(xs, lower, upper, alpha=0.12)
            plotted += 1

        if plotted == 0:
            plt.close()
            continue

        plt.grid(alpha=0.3)
        plt.title(f"{FRAMEWORK_LABELS.get(framework, framework)}: Predicate Threshold Sweep", size=14)
        plt.xlabel("Number of Learning Opportunities", size=12)
        plt.ylabel("Average Error Rate", size=12)
        plt.ylim(-0.02, 1.02)
        plt.legend()
        out_path = out_dir / f"{framework}_threshold_sweep.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot geometry learning curves comparing predicate thresholds for each framework."
    )
    parser.add_argument(
        "--aggregated-input",
        required=True,
        help="Combined threshold-sweep TSV/CSV with a Predicate Threshold column.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "geometry_threshold_sweep_curves"),
        help="Output directory for per-framework threshold comparison plots.",
    )
    parser.add_argument("--opp-count-cutoff", type=int, default=0)
    parser.add_argument("--opp-cutoff", type=int, default=15)
    parser.add_argument(
        "--monotonic-envelope",
        action="store_true",
        help="Plot a cumulative-min envelope so averaged error curves never increase.",
    )
    args = parser.parse_args()

    aggregated_input = Path(args.aggregated_input).resolve()
    output_dir = Path(args.output_dir).resolve()

    df = load_transactions(aggregated_input)
    curves = build_threshold_framework_curves(
        df,
        args.opp_count_cutoff,
        args.opp_cutoff,
        monotonic_envelope=args.monotonic_envelope,
    )
    plot_threshold_curves(curves, output_dir)
    print(f"Threshold sweep plots: {output_dir}")


if __name__ == "__main__":
    main()
