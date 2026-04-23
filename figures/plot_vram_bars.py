#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParsedToken:
    kind: Literal["value", "oom"]
    value: float | None = None


def _parse_vram_tokens(cell: object) -> list[ParsedToken]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell).strip()
    if not text:
        return []
    text = (
        text.replace(",", " ")
        .replace("，", " ")
        .replace(";", " ")
        .replace("\t", " ")
    )
    tokens: list[ParsedToken] = []
    for raw in text.split():
        if raw.upper() == "OOM":
            tokens.append(ParsedToken(kind="oom"))
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        tokens.append(ParsedToken(kind="value", value=value))
    return tokens


def _pick_preferred_order(
    available: list[str],
    preferred: list[str],
) -> list[str]:
    chosen = [x for x in preferred if x in available]
    remaining = [x for x in available if x not in set(chosen)]
    return chosen + sorted(remaining)


def _grouped_bars(
    *,
    ax,
    x_labels: list[str],
    series_labels: list[str],
    values_by_series: dict[str, np.ndarray],
    missing_kind_by_series: dict[str, np.ndarray] | None = None,
    colors_by_series: dict[str, tuple[float, float, float, float]] | None = None,
    hatches_by_series: dict[str, str] | None = None,
    ylabel: str,
    title: str,
    annotate: bool = True,
):
    x = np.arange(len(x_labels))
    bar_width = 0.8 / max(len(series_labels), 1)

    missing_markers: list[tuple[float, str]] = []

    def format_value(value: float) -> str:
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    for i, label in enumerate(series_labels):
        heights = values_by_series[label]
        offsets = x + (i - (len(series_labels) - 1) / 2) * bar_width

        draw_heights = np.where(np.isnan(heights), 0.0, heights)
        bars = ax.bar(
            offsets,
            draw_heights,
            width=bar_width,
            label=label,
            linewidth=1.0,
            color=(colors_by_series or {}).get(label),
            edgecolor="0.25",
            hatch=(hatches_by_series or {}).get(label, ""),
            alpha=1.0,
        )

        if annotate:
            ymax = ax.get_ylim()[1]
            y_offset = max(0.02 * ymax, 0.01)
            for j, bar in enumerate(bars):
                value = heights[j]
                if np.isnan(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    format_value(float(value)),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=90,
                    color="0.15",
                    fontweight="semibold",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                        "pad": 0.6,
                    },
                    clip_on=True,
                )

        if missing_kind_by_series is None:
            continue

        kinds = missing_kind_by_series[label]
        for j, bar in enumerate(bars):
            kind = kinds[j]
            if kind == "":
                continue
            bar.set_facecolor("none")
            bar.set_edgecolor("0.4")
            bar.set_hatch("//" if kind == "NA" else "xx")
            missing_markers.append((bar.get_x() + bar.get_width() / 2, kind))

    if annotate:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.35)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(False)
    ax.legend(frameon=False, ncol=max(1, len(series_labels)))

    if missing_markers:
        ymax = ax.get_ylim()[1]
        y = ymax * 0.02
        for x_pos, kind in missing_markers:
            ax.text(
                x_pos,
                y,
                kind,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="0.35",
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot VRAM grouped bar charts from a CSV like paper-dmd/data/plot_example.csv."
        )
    )
    parser.add_argument(
        "--csv",
        default="paper-dmd/data/plot_example.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--outdir",
        default="paper-dmd/figures",
        help="Output directory.",
    )
    parser.add_argument(
        "--vram-unit-column",
        default="VRAM_unit",
        help="Optional CSV column that stores VRAM units (Mi or Gi).",
    )
    parser.add_argument(
        "--default-vram-unit",
        default="Gi",
        choices=["Gi", "Mi"],
        help="Default VRAM unit when the unit column is absent.",
    )
    parser.add_argument(
        "--write-tex-table",
        default="paper-dmd/tables/vram_footprint.tex",
        help="Write a LaTeX table (path) summarizing plot-1 values; empty disables.",
    )
    parser.add_argument(
        "--taylorseer-order-for-method-plot",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Which max_order to use for the Taylorseer bar in the method comparison plot.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"model", "Method", "VRAM"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise SystemExit(f"Missing columns in CSV: {sorted(missing_cols)}")

    df["model"] = df["model"].astype(str).str.strip()
    df["Method"] = df["Method"].astype(str).str.strip()
    df["vram_tokens"] = df["VRAM"].apply(_parse_vram_tokens)
    if args.vram_unit_column in df.columns:
        df["vram_unit"] = df[args.vram_unit_column].astype(str).str.strip()
    else:
        df["vram_unit"] = args.default_vram_unit

    def unit_scale(unit: str) -> float:
        if unit == "Mi":
            return 1.0 / 1024.0
        if unit == "Gi":
            return 1.0
        raise ValueError(f"Unsupported VRAM unit: {unit!r}")

    preferred_method_order = [
        "Baseline",
        "Magcache",
        "Teacache",
        "DMD",
        "Taylorseer",
    ]
    preferred_model_order = [
        "Wan2.1-t2v-1.3B",
        "FLUX.1-dev",
        "HiDream-I1-full",
        "HuanyuanVideo",
    ]
    models = _pick_preferred_order(
        available=list(dict.fromkeys(df["model"].tolist())),
        preferred=preferred_model_order,
    )
    methods = _pick_preferred_order(
        available=list(dict.fromkeys(df["Method"].tolist())),
        preferred=preferred_method_order,
    )

    # ---- Plot 1: model x method (single VRAM per method) ----
    taylorseer_idx = args.taylorseer_order_for_method_plot - 1

    def pick_single_value(row: pd.Series) -> float:
        tokens: list[ParsedToken] = row["vram_tokens"]
        if not tokens:
            return np.nan
        if row["Method"] == "Taylorseer":
            if len(tokens) <= taylorseer_idx:
                return np.nan
            token = tokens[taylorseer_idx]
        else:
            token = tokens[0]
        if token.kind != "value":
            return np.nan
        assert token.value is not None
        return float(token.value) * unit_scale(str(row["vram_unit"]))

    def pick_single_kind(row: pd.Series) -> str:
        tokens: list[ParsedToken] = row["vram_tokens"]
        if not tokens:
            return "NA"
        if row["Method"] == "Taylorseer":
            if len(tokens) <= taylorseer_idx:
                return "NA"
            token = tokens[taylorseer_idx]
        else:
            token = tokens[0]
        if token.kind == "oom":
            return "OOM"
        if token.kind != "value":
            return "NA"
        return ""

    df_method = df.copy()
    df_method["vram_single"] = df_method.apply(pick_single_value, axis=1)
    df_method["vram_single_kind"] = df_method.apply(pick_single_kind, axis=1)

    pivot_method = df_method.pivot_table(
        index="model",
        columns="Method",
        values="vram_single",
        aggfunc="first",
    ).reindex(index=models, columns=methods)

    missing_kind_method = {}
    values_by_method = {}
    for method in methods:
        values = pivot_method[method].to_numpy(dtype=float)
        values_by_method[method] = values
        if method == "Taylorseer":
            kinds = (
                df_method[df_method["Method"] == "Taylorseer"]
                .pivot_table(
                    index="model",
                    values="vram_single_kind",
                    aggfunc="first",
                )
                .reindex(index=models)["vram_single_kind"]
                .astype(str)
                .to_numpy()
            )
            missing_kind_method[method] = kinds
        else:
            missing_kind_method[method] = np.where(np.isnan(values), "NA", "")

    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-muted")
    except Exception:
        pass

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    cmap_methods = plt.get_cmap("Set2")
    from matplotlib.colors import to_rgba

    method_hex_colors = {
        "Taylorseer": "#cc4c36",
        "Baseline": "#557f7c",
        "Teacache": "#a1c4bf",
        "Magcache": "#b6b6b6",
        "DMD": "#ebc9b9",
    }
    colors_by_method = {}
    hatches_by_method = {
        "Baseline": "",
        "Teacache": "////",
        "Magcache": "\\\\\\\\",
        "DMD": "xx",
        "Taylorseer": "++",
    }

    def normalize_hex_color(value: str) -> str:
        v = value.strip()
        if not v.startswith("#"):
            return v
        digits = v[1:]
        if len(digits) == 5:
            return f"#{digits}{digits[-1]}"
        return v

    for i, method in enumerate(methods):
        hex_color = method_hex_colors.get(method)
        if hex_color is not None:
            try:
                colors_by_method[method] = to_rgba(normalize_hex_color(hex_color))
                continue
            except ValueError:
                pass
        colors_by_method[method] = cmap_methods(i % 8)
    _grouped_bars(
        ax=ax1,
        x_labels=models,
        series_labels=methods,
        values_by_series=values_by_method,
        missing_kind_by_series=missing_kind_method,
        colors_by_series=colors_by_method,
        hatches_by_series=hatches_by_method,
        ylabel="Peak VRAM (GiB)",
        title="",
    )
    fig1.tight_layout()

    fig1_png = outdir / "vram_by_model_method.png"
    fig1_pdf = outdir / "vram_by_model_method.pdf"
    fig1.savefig(fig1_png, dpi=300)
    fig1.savefig(fig1_pdf)
    plt.close(fig1)

    # ---- Plot 2: model x max_order for Taylorseer ----
    df_taylorseer = df[df["Method"] == "Taylorseer"].copy()
    if df_taylorseer.empty:
        raise SystemExit("No rows found for Method == Taylorseer")

    max_orders = [1, 2, 3, 4]
    for idx, order in enumerate(max_orders):
        df_taylorseer[f"vram_order_{order}"] = df_taylorseer["vram_tokens"].apply(
            lambda toks: (
                float(toks[idx].value)
                if len(toks) > idx and toks[idx].kind == "value"
                else np.nan
            )
        )
        df_taylorseer[f"vram_order_{order}"] = df_taylorseer.apply(
            lambda row: (
                float(row[f"vram_order_{order}"]) * unit_scale(str(row["vram_unit"]))
                if not np.isnan(float(row[f"vram_order_{order}"]))
                else np.nan
            ),
            axis=1,
        )
        df_taylorseer[f"kind_order_{order}"] = df_taylorseer["vram_tokens"].apply(
            lambda toks: (
                "OOM"
                if len(toks) > idx and toks[idx].kind == "oom"
                else ("NA" if len(toks) <= idx or not toks else "")
            )
        )

    pivot_taylorseer = df_taylorseer.pivot_table(
        index="model",
        values=[f"vram_order_{o}" for o in max_orders],
        aggfunc="first",
    ).reindex(index=models)

    kinds_taylorseer = df_taylorseer.pivot_table(
        index="model",
        values=[f"kind_order_{o}" for o in max_orders],
        aggfunc="first",
    ).reindex(index=models)

    series_labels = [f"max_order={o}" for o in max_orders]
    values_by_order = {
        f"max_order={o}": pivot_taylorseer[f"vram_order_{o}"].to_numpy(dtype=float)
        for o in max_orders
    }
    missing_kind_by_order = {
        f"max_order={o}": kinds_taylorseer[f"kind_order_{o}"].astype(str).to_numpy()
        for o in max_orders
    }
    cmap_orders = plt.get_cmap("Blues")
    shades = cmap_orders(np.linspace(0.45, 0.85, len(max_orders)))
    colors_by_order = {label: tuple(shades[i]) for i, label in enumerate(series_labels)}

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    _grouped_bars(
        ax=ax2,
        x_labels=models,
        series_labels=series_labels,
        values_by_series=values_by_order,
        missing_kind_by_series=missing_kind_by_order,
        colors_by_series=colors_by_order,
        hatches_by_series={label: "" for label in series_labels},
        ylabel="Peak VRAM (GiB)",
        title="",
    )
    fig2.tight_layout()

    fig2_png = outdir / "vram_taylorseer_by_model_max_order.png"
    fig2_pdf = outdir / "vram_taylorseer_by_model_max_order.pdf"
    fig2.savefig(fig2_png, dpi=300)
    fig2.savefig(fig2_pdf)
    plt.close(fig2)

    if args.write_tex_table:
        tex_path = Path(args.write_tex_table)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        with tex_path.open("w", encoding="utf-8") as f:
            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Peak VRAM footprint (GiB) across models and methods.}\n")
            f.write("\\label{tab:app:vram:footprint}\n")
            f.write("\\adjustbox{max width=\\textwidth}{%\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\toprule\n")
            f.write(
                "Model & Baseline & TeaCache & MagCache & TaylorSeer (O="
                f"{args.taylorseer_order_for_method_plot}) & $D^3$-Cache \\\\\n"
            )
            f.write("\\midrule\n")
            method_map = {
                "Baseline": "Baseline",
                "Teacache": "Teacache",
                "Magcache": "Magcache",
                "Taylorseer": "Taylorseer",
                "DMD": "DMD",
            }
            for model in models:
                row = [model]
                for m in ["Baseline", "Teacache", "Magcache", "Taylorseer", "DMD"]:
                    col = method_map[m]
                    value = float(pivot_method.loc[model, col]) if col in pivot_method.columns else np.nan
                    if np.isnan(value):
                        if m == "Taylorseer":
                            kind = df_method[
                                (df_method["model"] == model)
                                & (df_method["Method"] == "Taylorseer")
                            ]["vram_single_kind"]
                            if not kind.empty and str(kind.iloc[0]) == "OOM":
                                row.append("OOM")
                            else:
                                row.append("---")
                        else:
                            row.append("---")
                    else:
                        row.append(f"{value:.2f}")
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}}\n")
            f.write("\\end{table*}\n")

    print(f"Wrote: {fig1_png}")
    print(f"Wrote: {fig1_pdf}")
    print(f"Wrote: {fig2_png}")
    print(f"Wrote: {fig2_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
