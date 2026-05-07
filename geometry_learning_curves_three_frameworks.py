#!/usr/bin/env python3
"""Generate geometry learning curves for three frameworks, including per-hint plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRAMEWORK_ORDER = ["feedback_only", "nl_hint_only", "feedback_and_nl_hint"]
FRAMEWORK_LABELS = {
    "feedback_only": "Feedback Only",
    "nl_hint_only": "NL Hint Only",
    "feedback_and_nl_hint": "Feedback + NL Hint",
}
FRAMEWORK_COLORS = {
    "feedback_only": "#1f77b4",
    "nl_hint_only": "#ff7f0e",
    "feedback_and_nl_hint": "#2ca02c",
}


def sanitize_filename(text: str, max_len: int = 96) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "hint"
    return cleaned[:max_len]


def load_transactions(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".txt" else ","
    df = pd.read_csv(path, sep=sep)

    required = ["Training Framework", "Anon Student Id", "Outcome"]
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

    df["OppIndex"] = df.groupby(["Training Framework", "Student", "KC"], sort=False).cumcount()
    return df


def kc_curve_from_counts(kc_df: pd.DataFrame, opp_count_cutoff: int, opp_cutoff: int) -> np.ndarray:
    kc_df = kc_df.sort_values("OppIndex")
    correct = kc_df["correct"].to_numpy(dtype=float)
    total = kc_df["total"].to_numpy(dtype=float)

    max_opp = len(total)
    if opp_count_cutoff > 0:
        bad = np.where(total < opp_count_cutoff)[0]
        if len(bad) > 0:
            max_opp = int(bad[0])

    if opp_cutoff != -1:
        max_opp = min(max_opp, int(opp_cutoff))

    if max_opp <= 0:
        return np.array([], dtype=float)

    return 1.0 - (correct[:max_opp] / total[:max_opp])


def _curve_iter(curves: Dict[str, np.ndarray] | Iterable[np.ndarray]) -> List[np.ndarray]:
    if isinstance(curves, dict):
        items = list(curves.values())
    else:
        items = list(curves)
    return [np.asarray(curve, dtype=float) for curve in items if len(curve) > 0]


def _pad_curves(curves: Dict[str, np.ndarray] | Iterable[np.ndarray]) -> np.ndarray:
    curve_list = _curve_iter(curves)
    if not curve_list:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(curve) for curve in curve_list)
    padded = np.full((len(curve_list), max_len), np.nan, dtype=float)
    for idx, curve in enumerate(curve_list):
        padded[idx, : len(curve)] = curve
    return padded


def summarize_curve_collection(
    curves: Dict[str, np.ndarray] | Iterable[np.ndarray],
    monotonic_envelope: bool = False,
) -> Dict[str, np.ndarray]:
    padded = _pad_curves(curves)
    if padded.size == 0:
        empty = np.array([], dtype=float)
        return {"mean": empty, "std": empty, "count": empty.astype(int)}

    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    count = np.sum(~np.isnan(padded), axis=0)

    if monotonic_envelope and len(mean) > 0:
        mean = np.minimum.accumulate(mean)

    return {"mean": mean, "std": std, "count": count}


def aggregate_curves(
    kc_curves: Dict[str, np.ndarray] | Iterable[np.ndarray],
    monotonic_envelope: bool = False,
) -> np.ndarray:
    return summarize_curve_collection(kc_curves, monotonic_envelope=monotonic_envelope)["mean"]


def build_student_framework_kc_curves(
    df: pd.DataFrame,
    opp_count_cutoff: int,
    opp_cutoff: int,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    counts = (
        df.groupby(["Training Framework", "Student", "KC", "OppIndex"], sort=False)
        .agg(correct=("IsCorrect", "sum"), total=("IsCorrect", "size"))
        .reset_index()
    )

    framework_student_kc_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        framework: {} for framework in FRAMEWORK_ORDER
    }
    for (framework, student), student_df in counts.groupby(["Training Framework", "Student"], sort=False):
        kc_curves: Dict[str, np.ndarray] = {}
        for kc, kc_df in student_df.groupby("KC", sort=False):
            curve = kc_curve_from_counts(kc_df, opp_count_cutoff, opp_cutoff)
            if len(curve) > 0:
                kc_curves[kc] = curve
        if kc_curves:
            framework_student_kc_curves.setdefault(framework, {})[student] = kc_curves
    return framework_student_kc_curves


def summarize_framework_curves(
    framework_student_kc_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    monotonic_envelope: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    summaries: Dict[str, Dict[str, np.ndarray]] = {}
    for framework in FRAMEWORK_ORDER:
        student_curves = []
        for kc_curves in framework_student_kc_curves.get(framework, {}).values():
            curve = aggregate_curves(kc_curves, monotonic_envelope=monotonic_envelope)
            if len(curve) > 0:
                student_curves.append(curve)
        summaries[framework] = summarize_curve_collection(
            student_curves,
            monotonic_envelope=monotonic_envelope,
        )
    return summaries


def plot_overall_curves(
    framework_student_kc_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: Path,
    monotonic_envelope: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    summaries = summarize_framework_curves(
        framework_student_kc_curves,
        monotonic_envelope=monotonic_envelope,
    )
    for framework in FRAMEWORK_ORDER:
        summary = summaries.get(framework, {})
        mean = summary.get("mean", np.array([], dtype=float))
        std = summary.get("std", np.array([], dtype=float))
        count = summary.get("count", np.array([], dtype=float))
        if len(mean) == 0:
            continue

        xs = np.arange(1, len(mean) + 1)
        color = FRAMEWORK_COLORS.get(framework)
        plt.plot(
            xs,
            mean,
            linewidth=2,
            marker="o",
            color=color,
            label=FRAMEWORK_LABELS.get(framework, framework),
        )
        if len(std) > 0 and np.nanmax(count) > 1:
            stderr = np.divide(std, np.sqrt(np.maximum(count, 1)), where=count > 0)
            lower = np.clip(mean - stderr, 0.0, 1.0)
            upper = np.clip(mean + stderr, 0.0, 1.0)
            plt.fill_between(xs, lower, upper, color=color, alpha=0.12)

    plt.grid(alpha=0.3)
    plt.title("Geometry Learning Curve (All Frameworks)", size=14)
    plt.xlabel("Number of Learning Opportunities", size=12)
    plt.ylabel("Average Error Rate", size=12)
    plt.ylim(-0.02, 1.02)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_per_hint_curves(
    framework_student_kc_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_dir: Path,
    require_all_frameworks: bool,
    monotonic_envelope: bool = False,
) -> List[str]:
    all_kc_sets = []
    for framework in FRAMEWORK_ORDER:
        kc_names = set()
        for kc_curves in framework_student_kc_curves.get(framework, {}).values():
            kc_names.update(kc_curves.keys())
        all_kc_sets.append(kc_names)

    if not all_kc_sets:
        return []

    if require_all_frameworks:
        hint_names = sorted(set.intersection(*all_kc_sets)) if all_kc_sets else []
    else:
        hint_names = sorted(set.union(*all_kc_sets))

    out_dir.mkdir(parents=True, exist_ok=True)
    written_hints: List[str] = []

    for kc in hint_names:
        plt.figure(figsize=(8, 5))
        plotted = 0
        for framework in FRAMEWORK_ORDER:
            student_curves = []
            for kc_curves in framework_student_kc_curves.get(framework, {}).values():
                curve = kc_curves.get(kc)
                if curve is not None and len(curve) > 0:
                    student_curves.append(curve)
            if not student_curves:
                continue

            summary = summarize_curve_collection(student_curves, monotonic_envelope=monotonic_envelope)
            mean = summary["mean"]
            std = summary["std"]
            count = summary["count"]
            xs = np.arange(1, len(mean) + 1)
            color = FRAMEWORK_COLORS.get(framework)
            plt.plot(
                xs,
                mean,
                linewidth=2,
                marker="o",
                color=color,
                label=FRAMEWORK_LABELS.get(framework, framework),
            )
            if len(std) > 0 and np.nanmax(count) > 1:
                stderr = np.divide(std, np.sqrt(np.maximum(count, 1)), where=count > 0)
                lower = np.clip(mean - stderr, 0.0, 1.0)
                upper = np.clip(mean + stderr, 0.0, 1.0)
                plt.fill_between(xs, lower, upper, color=color, alpha=0.12)
            plotted += 1

        if plotted == 0:
            plt.close()
            continue

        plt.grid(alpha=0.3)
        plt.title(f"Hint Learning Curve: {kc}", size=14)
        plt.xlabel("Number of Learning Opportunities", size=12)
        plt.ylabel("Average Error Rate", size=12)
        plt.ylim(-0.02, 1.02)
        plt.legend()
        out_path = out_dir / f"{sanitize_filename(kc)}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        written_hints.append(kc)

    return written_hints


def write_hint_curve_rows(
    framework_student_kc_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_csv: Path,
    monotonic_envelope: bool = False,
) -> None:
    rows = []
    for framework in FRAMEWORK_ORDER:
        kc_names = sorted(
            {
                kc
                for kc_curves in framework_student_kc_curves.get(framework, {}).values()
                for kc in kc_curves.keys()
            }
        )
        for kc in kc_names:
            student_curves = []
            for kc_curves in framework_student_kc_curves.get(framework, {}).values():
                curve = kc_curves.get(kc)
                if curve is not None and len(curve) > 0:
                    student_curves.append(curve)
            summary = summarize_curve_collection(student_curves, monotonic_envelope=monotonic_envelope)
            for opp_idx, err in enumerate(summary["mean"], start=1):
                rows.append(
                    {
                        "framework": framework,
                        "kc_field": kc,
                        "opportunity": opp_idx,
                        "error_rate": float(err),
                        "stderr": float(summary["std"][opp_idx - 1] / np.sqrt(max(summary["count"][opp_idx - 1], 1))),
                        "n_students": int(summary["count"][opp_idx - 1]),
                    }
                )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Plot overall and per-hint geometry learning curves for three frameworks."
    )
    parser.add_argument(
        "--aggregated-input",
        default=str(workspace_root / "tutor_gym" / "sandbox" / "geometry" / "geometry_log_al_3_frameworks_aggregated.txt"),
        help="Aggregated TSV/CSV from three frameworks.",
    )
    parser.add_argument(
        "--overall-plot",
        default=str(script_dir / "geometry_learning_curve_3_frameworks.png"),
        help="Output path for the combined overall learning-curve plot.",
    )
    parser.add_argument(
        "--per-hint-dir",
        default=str(script_dir / "geometry_learning_curves_by_hint_3_frameworks"),
        help="Output directory for per-hint plots.",
    )
    parser.add_argument(
        "--hint-curves-csv",
        default=str(script_dir / "geometry_learning_curves_by_hint_3_frameworks.csv"),
        help="Output CSV containing per-hint curve points.",
    )
    parser.add_argument("--opp-count-cutoff", type=int, default=0)
    parser.add_argument("--opp-cutoff", type=int, default=15)
    parser.add_argument(
        "--allow-missing-framework-hints",
        action="store_true",
        help="If set, include hints that are present in any framework (not only common hints).",
    )
    parser.add_argument(
        "--monotonic-envelope",
        action="store_true",
        help="Plot a cumulative-min envelope so averaged error curves never increase.",
    )
    args = parser.parse_args()

    aggregated_input = Path(args.aggregated_input).resolve()
    overall_plot = Path(args.overall_plot).resolve()
    per_hint_dir = Path(args.per_hint_dir).resolve()
    hint_curves_csv = Path(args.hint_curves_csv).resolve()

    df = load_transactions(aggregated_input)
    framework_student_kc_curves = build_student_framework_kc_curves(
        df,
        args.opp_count_cutoff,
        args.opp_cutoff,
    )

    plot_overall_curves(
        framework_student_kc_curves,
        overall_plot,
        monotonic_envelope=args.monotonic_envelope,
    )
    hints_written = plot_per_hint_curves(
        framework_student_kc_curves,
        per_hint_dir,
        require_all_frameworks=not args.allow_missing_framework_hints,
        monotonic_envelope=args.monotonic_envelope,
    )
    write_hint_curve_rows(
        framework_student_kc_curves,
        hint_curves_csv,
        monotonic_envelope=args.monotonic_envelope,
    )

    print(f"Overall plot: {overall_plot}")
    print(f"Per-hint plots directory: {per_hint_dir}")
    print(f"Per-hint curve CSV: {hint_curves_csv}")
    print(f"Hints plotted: {len(hints_written)}")


if __name__ == "__main__":
    main()
