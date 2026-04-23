#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


VBENCH_DIMS = [
    "overall_consistency",
    "scene",
    "subject_consistency",
    "background_consistency",
    "dynamic_degree",
    "motion_smoothness",
    "imaging_quality",
    "aesthetic_quality",
]

VBENCH_DIMS_NO_SCENE = [
    "overall_consistency",
    "subject_consistency",
    "background_consistency",
    "dynamic_degree",
    "motion_smoothness",
    "imaging_quality",
    "aesthetic_quality",
]


@dataclass(frozen=True)
class MethodStyle:
    display: str
    color_key: str


METHOD_STYLES: dict[str, MethodStyle] = {
    "Baseline": MethodStyle(display="Baseline", color_key="Baseline"),
    "Teacache": MethodStyle(display="TeaCache", color_key="TeaCache"),
    "Magcache": MethodStyle(display="MagCache", color_key="MagCache"),
    "Taylorseer": MethodStyle(display="TaylorSeer", color_key="TaylorSeer"),
    "DMD": MethodStyle(display=r"$D^3$-Cache", color_key="D3-Cache"),
}


def _parse_run_suffix(run: str) -> str:
    run = str(run or "").strip()
    if not run:
        return ""
    return run.split("/")[-1].strip()


def _parse_config(method: str, run: str) -> str:
    suffix = _parse_run_suffix(run)
    method = str(method).strip()
    if method == "Baseline":
        return "Original"
    if method == "DMD":
        m = re.fullmatch(r"H(\d+)S(\d+)", suffix)
        if m:
            return f"H={m.group(1)},S={m.group(2)}"
    if method == "Teacache":
        m = re.fullmatch(r"r([0-9.]+)", suffix)
        if m:
            return f"r={m.group(1)}"
    if method == "Magcache":
        m = re.fullmatch(r"threshold([0-9.]+)", suffix)
        if m:
            return f"$\\delta$={m.group(1)}"
    if method == "Taylorseer":
        m = re.fullmatch(r"N(\d+)O(\d+)I(\d+)", suffix)
        if m:
            return f"N={m.group(1)},O={m.group(2)},I={m.group(3)}"
    return suffix or run


def _pick_flops_column(columns: list[str], override: str | None) -> str | None:
    if override:
        return override if override in columns else None
    for candidate in [
        "flops_p",
        "flops_peta",
        "flops_pf",
        "flops_pflops",
        "FLOPs_P",
        "FLOPs",
    ]:
        if candidate in columns:
            return candidate
    return None


def _method_order_key(method: str) -> int:
    order = ["Baseline", "Teacache", "Magcache", "Taylorseer", "DMD"]
    try:
        return order.index(method)
    except ValueError:
        return len(order)


def _format_float(value: float, digits: int = 3) -> str:
    if np.isnan(value):
        return "---"
    return f"{value:.{digits}f}"


def _format_speedup(value: float) -> str:
    if np.isnan(value):
        return "---"
    return f"{value:.2f}$\\times$"


def _dmd_hs_from_run(run: str) -> tuple[int | None, int | None]:
    suffix = _parse_run_suffix(run)
    m = re.fullmatch(r"H(\d+)S(\d+)", suffix)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _appendix_method_order_key(method: str) -> int:
    order = ["Baseline", "DMD", "Teacache", "Magcache", "Taylorseer"]
    try:
        return order.index(method)
    except ValueError:
        return len(order)


def _format_method_appendix(method: str, run: str) -> str:
    method = str(method).strip()
    suffix = _parse_run_suffix(run)

    if method == "Baseline":
        return "Baseline"
    if method == "DMD":
        h, s = _dmd_hs_from_run(run)
        if h is not None and s is not None:
            return rf"$D^3$-Cache ($\mathcal{{H}}{{=}}{h}$,$\mathcal{{S}}{{=}}{s}$)"
        return r"$D^3$-Cache"
    if method == "Teacache":
        m = re.fullmatch(r"r([0-9.]+)", suffix)
        if m:
            return rf"TeaCache ($r={m.group(1)}$)"
        return "TeaCache"
    if method == "Magcache":
        m = re.fullmatch(r"threshold([0-9.]+)", suffix)
        if m:
            return rf"MagCache ($\delta$={m.group(1)})"
        return "MagCache"
    if method == "Taylorseer":
        m = re.fullmatch(r"N(\d+)O(\d+)I(\d+)", suffix)
        if m:
            n, o, i = m.group(1), m.group(2), m.group(3)
            return rf"TaylorSeer ($\mathcal{{N}}{{=}}{n}$,$\mathcal{{O}}{{=}}{o}$,$\mathcal{{I}}{{=}}{i}$)"
        return "TaylorSeer"

    style = METHOD_STYLES.get(method, MethodStyle(method, method))
    return style.display


def _infer_method_from_run(run: str) -> str:
    suffix = _parse_run_suffix(run).strip()
    if suffix.casefold() in {"baseline", "original"}:
        return "Baseline"
    if re.fullmatch(r"H\d+S\d+", suffix):
        return "DMD"
    if re.fullmatch(r"threshold[0-9.]+", suffix):
        return "Magcache"
    if re.fullmatch(r"r[0-9.]+", suffix):
        return "Teacache"
    if re.fullmatch(r"N\d+O\d+I\d+", suffix):
        return "Taylorseer"
    return "Unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot efficiency + (VBench) visual-quality figures from "
            "paper-dmd/data/vbench_8dims_time_speedup.csv, and optionally write a LaTeX table."
        )
    )
    parser.add_argument(
        "--csv",
        default="paper-dmd/data/vbench_8dims_time_speedup.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--outdir",
        default="paper-dmd/figures",
        help="Output directory.",
    )
    parser.add_argument(
        "--write-tex-table",
        default="",
        help="Write a LaTeX table (path); empty disables.",
    )
    parser.add_argument(
        "--write-tex-appendix-table",
        default="paper-dmd/tables/wan21_vbench_main.tex",
        help=(
            "Write the appendix VBench table (8 dims + average over 7 non-scene dims). "
            "Empty disables."
        ),
    )
    parser.add_argument(
        "--appendix-csv",
        default="paper-dmd/data/vbench_scores_summary.csv",
        help="CSV path for the appendix VBench table.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip writing figures; useful when only generating LaTeX tables.",
    )
    parser.add_argument(
        "--flops-column",
        default="",
        help="Optional FLOPs column name (if present in CSV).",
    )
    parser.add_argument(
        "--title-prefix",
        default="Wan2.1",
        help="Figure title prefix.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help=(
            "Optional suffix appended to output filenames (e.g. 'subset' -> "
            "vbench_efficiency_subset.png)."
        ),
    )
    parser.add_argument(
        "--include-run",
        action="append",
        default=[],
        help=(
            "Repeatable substring filter on the 'run' column (case-insensitive). "
            "If provided, only rows whose run contains ANY of these substrings are kept."
        ),
    )
    parser.add_argument(
        "--include-run-file",
        default="",
        help=(
            "Optional path to a text file (one substring per line, # for comments) "
            "to extend --include-run."
        ),
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"method", "run", "avg_prompt_sec", "speedup_vs_baseline"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in CSV: {sorted(missing)}")

    df["method"] = df["method"].astype(str).str.strip()
    df["run"] = df["run"].astype(str).str.strip()

    include_runs: list[str] = [
        str(x).strip() for x in (args.include_run or []) if str(x).strip()
    ]
    if args.include_run_file:
        include_path = Path(args.include_run_file)
        if not include_path.exists():
            raise SystemExit(f"--include-run-file not found: {include_path}")
        for line in include_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            include_runs.append(line)

    if include_runs:
        run_cf = df["run"].astype(str).str.casefold()
        mask = np.zeros(len(df), dtype=bool)
        for needle in include_runs:
            needle_cf = needle.casefold()
            mask |= run_cf.str.contains(re.escape(needle_cf), regex=True).to_numpy()
        df = df.loc[mask].copy()
        if df.empty:
            raise SystemExit(
                "No rows matched --include-run filters. "
                f"Filters={include_runs}"
            )

    df["config"] = df.apply(lambda r: _parse_config(r["method"], r["run"]), axis=1)
    df["label"] = df.apply(
        lambda r: f"{METHOD_STYLES.get(r['method'], MethodStyle(r['method'], r['method'])).display}\n{r['config']}",
        axis=1,
    )

    if "vbench_score" not in df.columns:
        missing_dims = [c for c in VBENCH_DIMS_NO_SCENE if c not in df.columns]
        if missing_dims:
            raise SystemExit(
                "CSV is missing vbench_score and required VBench dims: "
                f"{missing_dims}"
            )
        df["vbench_score"] = df[VBENCH_DIMS_NO_SCENE].mean(axis=1)

    flops_col = _pick_flops_column(df.columns.tolist(), args.flops_column or None)
    if flops_col is None:
        df["__flops_missing__"] = np.nan
        flops_col = "__flops_missing__"

    df["__method_order__"] = df["method"].apply(_method_order_key)
    df = df.sort_values(
        by=["__method_order__", "speedup_vs_baseline", "avg_prompt_sec"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    suffix = f"_{args.name_suffix.strip()}" if args.name_suffix.strip() else ""
    eff_png = outdir / f"vbench_efficiency{suffix}.png"
    eff_pdf = outdir / f"vbench_efficiency{suffix}.pdf"
    vq_png = outdir / f"vbench_visual_quality{suffix}.png"
    vq_pdf = outdir / f"vbench_visual_quality{suffix}.pdf"

    if not args.no_figures:
        import matplotlib.pyplot as plt

        try:
            plt.style.use("seaborn-v0_8-muted")
        except Exception:
            pass

        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titleweight": "semibold",
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 10,
            }
        )
        # Stable per-method colors (shared across figures).
        palette = plt.get_cmap("Set2")
        color_keys = ["Baseline", "TeaCache", "MagCache", "TaylorSeer", "D3-Cache"]
        colors_by_key = {k: palette(i % 8) for i, k in enumerate(color_keys)}

        def row_color(method: str):
            style = METHOD_STYLES.get(method, MethodStyle(method, method))
            return colors_by_key.get(style.color_key, (0.35, 0.35, 0.35, 1.0))

        bar_colors = [row_color(m) for m in df["method"].tolist()]
        x = np.arange(len(df))

        # ---- Figure 1: Efficiency ----
        metrics = [
            (flops_col, "FLOPs", "FLOPs (arb.)" if flops_col == "__flops_missing__" else "FLOPs"),
            ("speedup_vs_baseline", "Speedup", "Speedup ($\\uparrow$)"),
            ("avg_prompt_sec", "Latency", "Latency (s) ($\\downarrow$)"),
        ]

        fig1, axes1 = plt.subplots(
            nrows=1,
            ncols=len(metrics),
            figsize=(max(10, 0.85 * len(df)), 3.8),
            sharex=True,
        )
        if len(metrics) == 1:
            axes1 = [axes1]

        for ax, (col, title, ylabel) in zip(axes1, metrics, strict=True):
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            draw_values = np.where(np.isnan(values), 0.0, values)
            bars = ax.bar(x, draw_values, color=bar_colors, alpha=0.92, linewidth=0.7)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle="--", alpha=0.25)

            if col == "__flops_missing__":
                ax.text(
                    0.5,
                    0.5,
                    "Missing FLOPs column\n(add e.g. flops_p to CSV)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.35",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 4},
                )
                for b in bars:
                    b.set_facecolor("none")
                    b.set_edgecolor("0.6")
                    b.set_hatch("//")

            if col in {"speedup_vs_baseline", "avg_prompt_sec"}:
                ymax = ax.get_ylim()[1]
                y_offset = max(0.02 * ymax, 0.01)
                for idx, b in enumerate(bars):
                    v = values[idx]
                    if np.isnan(v):
                        continue
                    txt = f"{v:.2f}" if col == "speedup_vs_baseline" else f"{v:.1f}"
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + y_offset,
                        txt,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                        color="0.2",
                        clip_on=True,
                    )

        for ax in axes1:
            ax.set_xticks(x)
            ax.set_xticklabels(df["label"].tolist(), rotation=25, ha="right")

        fig1.suptitle(f"{args.title_prefix}: Efficiency", y=1.02)
        fig1.tight_layout()

        fig1.savefig(eff_png, dpi=300, bbox_inches="tight")
        fig1.savefig(eff_pdf, bbox_inches="tight")
        plt.close(fig1)

        # ---- Figure 2: Visual quality (VBench) ----
        quality_cols = [c for c in VBENCH_DIMS if c in df.columns] + ["vbench_score"]
        quality_names = {
            "overall_consistency": "Overall",
            "scene": "Scene",
            "subject_consistency": "Subject",
            "background_consistency": "Background",
            "dynamic_degree": "Dynamic",
            "motion_smoothness": "Smoothness",
            "imaging_quality": "Imaging",
            "aesthetic_quality": "Aesthetic",
            "vbench_score": "VBench (avg, no-scene)",
        }

        quality = df[quality_cols].apply(pd.to_numeric, errors="coerce")
        data = quality.to_numpy(dtype=float).T  # (metrics, runs)

        fig2, ax2 = plt.subplots(
            figsize=(max(10, 0.9 * len(df)), 4.2),
        )
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color=(0.9, 0.9, 0.9, 1.0))
        im = ax2.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)

        ax2.set_yticks(np.arange(len(quality_cols)))
        ax2.set_yticklabels([quality_names.get(c, c) for c in quality_cols])
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["label"].tolist(), rotation=25, ha="right")
        ax2.set_title(f"{args.title_prefix}: Visual Quality (VBench)")

        cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
        cbar.set_label("Score")

        fig2.tight_layout()
        fig2.savefig(vq_png, dpi=300, bbox_inches="tight")
        fig2.savefig(vq_pdf, bbox_inches="tight")
        plt.close(fig2)

    # ---- Optional LaTeX table ----
    if args.write_tex_table:
        tex_path = Path(args.write_tex_table)
        tex_path.parent.mkdir(parents=True, exist_ok=True)

        def fmt_vbench(value: float) -> str:
            if np.isnan(value):
                return "---"
            return f"{value:.3f}"

        with tex_path.open("w", encoding="utf-8") as f:
            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Quantitative comparison on Wan2.1 (VBench + efficiency).}\n")
            f.write("\\label{tab:wan21-vbench}\n")
            f.write("\\adjustbox{max width=\\textwidth}{%\n")
            f.write("\\begingroup\n")
            f.write("\\setlength{\\tabcolsep}{0pt}\n")
            f.write("\\renewcommand{\\arraystretch}{1.08}\n")
            f.write("\\newcolumntype{L}{>{\\hspace{0.45em}}l<{\\hspace{0.45em}}}\n")
            f.write("\\newcolumntype{C}{>{\\hspace{0.35em}}c<{\\hspace{0.35em}}}\n")
            f.write("\\begin{tabular}{LCCC CCCCCCCCC}\n")
            f.write("\\toprule\n")
            f.write(
                "\\multirow{2}{*}{\\textbf{Method}}"
                " & \\multicolumn{3}{c|}{\\textbf{Efficiency}}"
                " & \\multicolumn{9}{c}{\\textbf{Visual Quality (VBench)}} \\\\\n"
            )
            f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-13}\n")
            f.write(
                " & \\textbf{FLOPs}$\\downarrow$"
                " & \\textbf{Speedup}$\\uparrow$"
                " & \\textbf{Latency (s)}$\\downarrow$"
                " & \\textbf{Overall}$\\uparrow$"
                " & \\textbf{Scene}$\\uparrow$"
                " & \\textbf{Subject}$\\uparrow$"
                " & \\textbf{Background}$\\uparrow$"
                " & \\textbf{Dynamic}$\\uparrow$"
                " & \\textbf{Smooth}$\\uparrow$"
                " & \\textbf{Imaging}$\\uparrow$"
                " & \\textbf{Aesthetic}$\\uparrow$"
                " & \\textbf{VBench}$\\uparrow$ \\\\\n"
            )
            f.write("\\midrule\n")

            for _, row in df.iterrows():
                method = str(row["method"])
                style = METHOD_STYLES.get(method, MethodStyle(method, method))
                method_tex = style.display
                config = str(row["config"])
                if method != "Baseline" and config:
                    method_tex = f"{method_tex} ({config})"

                flops_val = pd.to_numeric(row.get(flops_col), errors="coerce")
                if np.isnan(float(flops_val)) or flops_col == "__flops_missing__":
                    flops_tex = "---"
                else:
                    flops_tex = _format_float(float(flops_val), digits=2)

                speed_tex = _format_speedup(float(pd.to_numeric(row["speedup_vs_baseline"], errors="coerce")))
                lat_tex = _format_float(float(pd.to_numeric(row["avg_prompt_sec"], errors="coerce")), digits=2)

                vbench_vals = [
                    fmt_vbench(float(pd.to_numeric(row.get(c), errors="coerce")))
                    for c in VBENCH_DIMS
                ]
                vbench_avg = fmt_vbench(float(pd.to_numeric(row.get("vbench_score"), errors="coerce")))

                f.write(
                    " & ".join([method_tex, flops_tex, speed_tex, lat_tex, *vbench_vals, vbench_avg])
                    + " \\\\\n"
                )

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\endgroup}\n")
            f.write("\\end{table*}\n")

    # ---- Optional appendix LaTeX table (VBench 8 dims) ----
    if args.write_tex_appendix_table:
        appendix_csv_path = Path(args.appendix_csv)
        df_app = pd.read_csv(appendix_csv_path)
        df_app.columns = [str(c).strip() for c in df_app.columns]

        tex_path = Path(args.write_tex_appendix_table)
        tex_path.parent.mkdir(parents=True, exist_ok=True)

        if "run" not in df_app.columns:
            raise SystemExit(f"Appendix CSV is missing required column: run ({appendix_csv_path})")

        df_app["method"] = df_app["run"].apply(_infer_method_from_run)
        df_app["__app_method_order__"] = df_app["method"].apply(_appendix_method_order_key)
        hs = df_app["run"].apply(_dmd_hs_from_run)
        df_app["__dmd_h__"] = [h if h is not None else 10**9 for h, _ in hs]
        df_app["__dmd_s__"] = [s if s is not None else 10**9 for _, s in hs]

        # Compute the paper's VBench score: average over 7 non-scene dims.
        vb7_cols = [
            "overall_consistency",
            "subject_consistency",
            "background_consistency",
            "dynamic_degree",
            "motion_smoothness",
            "imaging_quality",
            "aesthetic_quality",
        ]
        missing_vb7 = [c for c in vb7_cols if c not in df_app.columns]
        if missing_vb7:
            raise SystemExit(
                f"Appendix CSV is missing required VBench columns: {missing_vb7} ({appendix_csv_path})"
            )
        df_app["vbench_score"] = df_app[vb7_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

        if "speedup" in df_app.columns:
            df_app["__sort_speedup__"] = pd.to_numeric(df_app["speedup"], errors="coerce")
        elif "speedup_vs_baseline" in df_app.columns:
            df_app["__sort_speedup__"] = pd.to_numeric(df_app["speedup_vs_baseline"], errors="coerce")
        else:
            df_app["__sort_speedup__"] = np.nan

        if "latency_sec" in df_app.columns:
            df_app["__sort_latency__"] = pd.to_numeric(df_app["latency_sec"], errors="coerce")
        elif "avg_prompt_sec" in df_app.columns:
            df_app["__sort_latency__"] = pd.to_numeric(df_app["avg_prompt_sec"], errors="coerce")
        else:
            df_app["__sort_latency__"] = np.nan

        df_app = df_app.sort_values(
            by=[
                "__app_method_order__",
                "__dmd_h__",
                "__dmd_s__",
                "__sort_speedup__",
                "__sort_latency__",
            ],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        def fmt_vbench(value: float) -> str:
            if np.isnan(value):
                return "---"
            return f"{value:.3f}"

        with tex_path.open("w", encoding="utf-8") as f:
            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Additional Wan2.1 results on VBench.}\n")
            f.write("\\label{tab:app:wan21-vbench-8dims}\n")
            f.write("\\adjustbox{max width=\\textwidth}{\n")
            f.write("\\begingroup\n")
            f.write("\\setlength{\\tabcolsep}{0pt}\n")
            f.write("\\renewcommand{\\arraystretch}{1.08}\n")
            f.write("\\newcolumntype{L}{>{\\hspace{0.45em}}l<{\\hspace{0.45em}}}\n")
            f.write("\\newcolumntype{C}{>{\\hspace{0.35em}}c<{\\hspace{0.35em}}}\n")
            f.write("\\begin{tabular}{L CCCCCCCCC}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Method}\n")
            f.write("    & \\tbox{\\textbf{Overall}\\\\\\textbf{Consistency}$\\uparrow$}\n")
            f.write("    & \\tbox{\\textbf{Scene}\\\\\\textbf{Consistency}$\\uparrow$}\n")
            f.write("    & \\tbox{\\textbf{Subject}\\\\\\textbf{Consistency}$\\uparrow$}\n")
            f.write("    & \\tbox{\\textbf{Background}\\\\\\textbf{Consistency}$\\uparrow$}\n")
            f.write("    & \\tbox{\\textbf{Dynamic}\\\\\\textbf{Degree}$\\uparrow$}\n")
            f.write("    & \\tbox{\\textbf{Motion}\\\\\\textbf{Smoothness}$\\uparrow$}\n")
            f.write("    & \\tbox{\\textbf{Imaging}\\\\\\textbf{Quality}$\\uparrow$}\n")
            f.write("    & \\tbox{\\textbf{Aesthetic}\\\\\\textbf{Quality}$\\uparrow$}\n")
            f.write("    & \\tbox{\\textbf{VBench}\\\\\\textbf{Score}$\\uparrow$} \\\\\n")
            f.write("\\midrule\n")

            vbench_cols = [
                "overall_consistency",
                "scene",
                "subject_consistency",
                "background_consistency",
                "dynamic_degree",
                "motion_smoothness",
                "imaging_quality",
                "aesthetic_quality",
            ]
            for _, row in df_app.iterrows():
                method_tex = _format_method_appendix(row["method"], row["run"])
                vals = [fmt_vbench(float(pd.to_numeric(row.get(c), errors="coerce"))) for c in vbench_cols]
                vbench_score = fmt_vbench(float(pd.to_numeric(row.get("vbench_score"), errors="coerce")))

                f.write(f"{method_tex}\n")
                f.write("    & " + "\n    & ".join([*vals, vbench_score]) + " \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\endgroup}\n")
            f.write("\\end{table*}\n")

        print(f"Wrote: {args.write_tex_appendix_table}")

    if not args.no_figures:
        print(f"Wrote: {eff_png}")
        print(f"Wrote: {eff_pdf}")
        print(f"Wrote: {vq_png}")
        print(f"Wrote: {vq_pdf}")
    if args.write_tex_table:
        print(f"Wrote: {args.write_tex_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
