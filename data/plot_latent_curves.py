#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "motivation"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    w = rank - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def _load_jsonl(pattern: str) -> tuple[list[dict], list[str]]:
    files = sorted(glob.glob(pattern))
    rows: list[dict] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows, files


def _aggregate_by_step(rows: list[dict], metric: str) -> list[dict]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        step = row.get("step")
        value = row.get(metric)
        if step is None or value is None:
            continue
        by_step[int(step)].append(float(value))

    out = []
    for step in sorted(by_step):
        values = by_step[step]
        out.append(
            {
                "step": step,
                "mean": float(sum(values) / len(values)),
                "p10": _percentile(values, 0.1),
                "p90": _percentile(values, 0.9),
                "count": len(values),
            }
        )
    return out


def _plot_metric(model_curves: dict[str, list[dict]], metric: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for model_name, curve in model_curves.items():
        if not curve:
            continue
        xs = [row["step"] for row in curve]
        ys = [row["mean"] for row in curve]
        lo = [row["p10"] for row in curve]
        hi = [row["p90"] for row in curve]
        plt.plot(xs, ys, linewidth=2.0, label=model_name)
        if all(v is not None for v in lo) and all(v is not None for v in hi):
            plt.fill_between(xs, lo, hi, alpha=0.2)

    plt.xlabel("Denoising step")
    plt.ylabel(metric)
    plt.title(f"Baseline adjacent-step {metric}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot baseline latent cosine/rel_l2 curves for FLUX, Hi-Dream, and Wan2.1."
    )
    parser.add_argument(
        "--flux-glob",
        type=str,
        default=str(DATA_DIR / "flux_latent_similarity.jsonl"),
    )
    parser.add_argument(
        "--hidream-glob",
        type=str,
        default=str(DATA_DIR / "hi-dream-latent_similarity.jsonl"),
    )
    parser.add_argument(
        "--wan-glob",
        type=str,
        default=str(DATA_DIR / "wan_latent_similarity.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_inputs = {
        "FLUX.1-dev": args.flux_glob,
        "Hi-Dream": args.hidream_glob,
        "Wan2.1": args.wan_glob,
    }

    model_rows: dict[str, list[dict]] = {}
    source_files: dict[str, list[str]] = {}
    for model_name, pattern in model_inputs.items():
        rows, files = _load_jsonl(pattern)
        model_rows[model_name] = rows
        source_files[model_name] = files

    cosine_curves = {
        model: _aggregate_by_step(rows, "cosine_mean")
        for model, rows in model_rows.items()
    }
    rel_l2_curves = {
        model: _aggregate_by_step(rows, "rel_l2_mean")
        for model, rows in model_rows.items()
    }

    _plot_metric(cosine_curves, "cosine_mean", output_dir / "cosine_vs_step.png")
    _plot_metric(rel_l2_curves, "rel_l2_mean", output_dir / "rel_l2_vs_step.png")

    csv_path = output_dir / "aggregated_latent_curves.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "metric", "step", "mean", "p10", "p90", "count"])
        for model_name, curve in cosine_curves.items():
            for row in curve:
                writer.writerow([
                    model_name,
                    "cosine_mean",
                    row["step"],
                    row["mean"],
                    row["p10"],
                    row["p90"],
                    row["count"],
                ])
        for model_name, curve in rel_l2_curves.items():
            for row in curve:
                writer.writerow([
                    model_name,
                    "rel_l2_mean",
                    row["step"],
                    row["mean"],
                    row["p10"],
                    row["p90"],
                    row["count"],
                ])

    summary = {
        "inputs": source_files,
        "output_files": {
            "cosine_plot": str(output_dir / "cosine_vs_step.png"),
            "rel_l2_plot": str(output_dir / "rel_l2_vs_step.png"),
            "aggregate_csv": str(csv_path),
        },
        "num_rows": {model: len(rows) for model, rows in model_rows.items()},
    }
    with (output_dir / "plot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved plots and summary to: {output_dir}")


if __name__ == "__main__":
    main()
