#!/usr/bin/env python3
"""Generate geometry learning curves for three frameworks, including per-hint plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRAMEWORK_ORDER = ["feedback_only", "nl_hint_only", "feedback_and_nl_hint"]
FRAMEWORK_LABELS = {
    "feedback_only": "Feedback Only",
    "nl_hint_only": "NL Hint Only",
    "feedback_and_nl_hint": "Feedback + NL Hint",
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

    # Opportunity index within each (framework, student, KC) sequence.
    df["OppIndex"] = df.groupby(["Training Framework", "Student", "KC"], sort=False).cumcount()
    return df


def build_kc_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["Training Framework", "KC", "OppIndex"], sort=False)
        .agg(correct=("IsCorrect", "sum"), total=("IsCorrect", "size"))
        .reset_index()
    )
    return counts


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

    err_curve = 1.0 - (correct[:max_opp] / total[:max_opp])
    return err_curve


def aggregate_curves(kc_curves: Dict[str, np.ndarray]) -> np.ndarray:
    if not kc_curves:
        return np.array([], dtype=float)
    max_len = max(len(arr) for arr in kc_curves.values())
    padded_curves = [np.pad(arr, (0, max_len - len(arr))) for arr in kc_curves.values()]
    return np.sum(padded_curves, axis=0) / len(padded_curves)


def build_framework_curves(
    counts: pd.DataFrame, opp_count_cutoff: int, opp_cutoff: int
) -> Dict[str, Dict[str, np.ndarray]]:
    framework_kc_curves: Dict[str, Dict[str, np.ndarray]] = {}
    for framework in FRAMEWORK_ORDER:
        fw_counts = counts[counts["Training Framework"] == framework]
        kc_curves: Dict[str, np.ndarray] = {}
        for kc, kc_df in fw_counts.groupby("KC", sort=False):
            curve = kc_curve_from_counts(kc_df, opp_count_cutoff, opp_cutoff)
            if len(curve) > 0:
                kc_curves[kc] = curve
        framework_kc_curves[framework] = kc_curves
    return framework_kc_curves


def plot_overall_curves(framework_kc_curves: Dict[str, Dict[str, np.ndarray]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for framework in FRAMEWORK_ORDER:
        agg_curve = aggregate_curves(framework_kc_curves.get(framework, {}))
        if len(agg_curve) == 0:
            continue
        plt.plot(
            np.arange(1, len(agg_curve) + 1),
            agg_curve,
            linewidth=2,
            marker="o",
            label=FRAMEWORK_LABELS.get(framework, framework),
        )

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
    framework_kc_curves: Dict[str, Dict[str, np.ndarray]], out_dir: Path, require_all_frameworks: bool
) -> List[str]:
    all_kc_sets = [set(framework_kc_curves.get(framework, {}).keys()) for framework in FRAMEWORK_ORDER]
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
            curve = framework_kc_curves.get(framework, {}).get(kc)
            if curve is None or len(curve) == 0:
                continue
            plt.plot(
                np.arange(1, len(curve) + 1),
                curve,
                linewidth=2,
                marker="o",
                label=FRAMEWORK_LABELS.get(framework, framework),
            )
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


def write_hint_curve_rows(framework_kc_curves: Dict[str, Dict[str, np.ndarray]], out_csv: Path) -> None:
    rows = []
    for framework in FRAMEWORK_ORDER:
        kc_curves = framework_kc_curves.get(framework, {})
        for kc, curve in kc_curves.items():
            for opp_idx, err in enumerate(curve, start=1):
                rows.append(
                    {
                        "framework": framework,
                        "kc_field": kc,
                        "opportunity": opp_idx,
                        "error_rate": float(err),
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
    args = parser.parse_args()

    aggregated_input = Path(args.aggregated_input).resolve()
    overall_plot = Path(args.overall_plot).resolve()
    per_hint_dir = Path(args.per_hint_dir).resolve()
    hint_curves_csv = Path(args.hint_curves_csv).resolve()

    df = load_transactions(aggregated_input)
    counts = build_kc_counts(df)
    framework_kc_curves = build_framework_curves(counts, args.opp_count_cutoff, args.opp_cutoff)

    plot_overall_curves(framework_kc_curves, overall_plot)
    hints_written = plot_per_hint_curves(
        framework_kc_curves,
        per_hint_dir,
        require_all_frameworks=not args.allow_missing_framework_hints,
    )
    write_hint_curve_rows(framework_kc_curves, hint_curves_csv)

    print(f"Overall plot: {overall_plot}")
    print(f"Per-hint plots directory: {per_hint_dir}")
    print(f"Per-hint curve CSV: {hint_curves_csv}")
    print(f"Hints plotted: {len(hints_written)}")


if __name__ == "__main__":
    main()
