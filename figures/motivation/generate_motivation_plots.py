#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _read_step_stats_csv(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k is None:
                    continue
                cols.setdefault(k, [])
                cols[k].append(float(v) if v is not None and v != "" else float("nan"))
    return cols


def _plot_quantile_bands(
    *,
    stats_csv: Path,
    out_png: Path,
    y_label: str,
    y_ref: float | None,
    figsize: tuple[float, float] = (9.0, 4.0),
    dpi: int = 300,
) -> None:
    cols = _read_step_stats_csv(stats_csv)

    x = cols.get("step")
    if not x:
        raise ValueError(f"Missing 'step' column in {stats_csv}")

    p10 = cols["p10"]
    p90 = cols["p90"]
    mean = cols["mean"]

    # Downscaling in the two-column PDF can wash out subtle bands. Use higher DPI and
    # a stronger hatch overlay so p10–p90 remains legible after inclusion.
    with plt.rc_context({"hatch.linewidth": 2.2}):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        outer_face_color = "#BDBDBD"
        outer_edge_color = "#5A5A5A"
        mean_color = "#0B4F8A"

        ax.fill_between(
            x,
            p10,
            p90,
            color=outer_face_color,
            alpha=0.30,
            linewidth=0,
            zorder=1,
        )
        # Hatch overlay on top (no facecolor) so it stays visible even where bands overlap.
        outer_hatch = ax.fill_between(
            x,
            p10,
            p90,
            color="none",
            alpha=0.0,
            linewidth=0.0,
            zorder=3,
        )
        outer_hatch.set_hatch("///")
        outer_hatch.set_edgecolor(outer_edge_color)

        # Outline the outer band for visibility after downscaling.
        ax.plot(x, p10, color=outer_edge_color, alpha=0.95, linewidth=3.2, zorder=4)
        ax.plot(x, p90, color=outer_edge_color, alpha=0.95, linewidth=3.2, zorder=4)

        ax.plot(
            x,
            mean,
            color=mean_color,
            linewidth=4.2,
            linestyle="--",
            label="mean",
            zorder=5,
        )

        if y_ref is not None:
            ax.axhline(y_ref, color="#666666", linestyle=":", linewidth=2.0)

        # Add some headroom so the legend can sit inside the axes without covering the curves.
        y_min = np.nanmin([np.nanmin(p10), np.nanmin(mean)])
        y_max = np.nanmax([np.nanmax(p90), np.nanmax(mean)])
        # Add generous headroom so the legend fits without occluding curves.
        y_pad = 0.28 * (y_max - y_min + 1e-9)
        ax.set_ylim(y_min - 0.02 * (y_max - y_min + 1e-9), y_max + y_pad)

        # Note: this figure is downscaled heavily in the two-column PDF, so we use
        # larger font sizes for readability after inclusion.
        ax.set_xlabel("Timestep", fontsize=28)
        ax.set_ylabel(y_label, fontsize=28)

        ax.tick_params(axis="both", which="major", labelsize=22, width=1.6, length=6)
        ax.grid(True, which="major", alpha=0.25, linewidth=1.2)

        ax.set_xlim(1, 50)
        ax.set_xticks([1, 10, 20, 30, 40, 50])

        legend_handles = [
            Patch(
                facecolor=outer_face_color,
                alpha=0.25,
                edgecolor=outer_edge_color,
                linewidth=0.0,
                hatch="///",
                label="p10–p90",
            ),
            Line2D([0], [0], color=mean_color, linewidth=3.0, linestyle="--", label="mean"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="none",
            fontsize=18,
            columnspacing=1.0,
            handlelength=2.0,
            handletextpad=0.5,
            borderaxespad=0.0,
        )

        fig.subplots_adjust(left=0.11, right=0.995, bottom=0.21, top=0.90)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate motivation quantile-band plots from step stats CSVs.")
    parser.add_argument("--ratio-csv", type=Path, required=True, help="Path to latent_ratio_step_stats.csv")
    parser.add_argument("--cosine-csv", type=Path, required=True, help="Path to latent_cosine_step_stats.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory for PNGs (default: current directory).",
    )
    args = parser.parse_args()

    _plot_quantile_bands(
        stats_csv=args.ratio_csv,
        out_png=args.out_dir / "latent_ratio_step_quantile_bands.png",
        y_label="Norm ratio",
        y_ref=1.0,
    )
    _plot_quantile_bands(
        stats_csv=args.cosine_csv,
        out_png=args.out_dir / "latent_cosine_step_quantile_bands.png",
        y_label="Cosine mean",
        y_ref=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
