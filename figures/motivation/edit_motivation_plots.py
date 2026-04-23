#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(font_path, size)


def stylize_plot(in_path: Path, out_path: Path) -> None:
    img = Image.open(in_path).convert("RGB")
    width, height = img.size

    draw = ImageDraw.Draw(img)

    font_axis_bold = _load_font(42, bold=True)
    font_axis = _load_font(34, bold=False)
    font_legend = _load_font(34, bold=False)

    is_cosine = "cosine" in in_path.name.lower()

    # ---- Remove old y-axis label (vertical) without touching y-ticks ----
    # In the original plots, y-tick labels start at x≈35+, so keep this narrow.
    draw.rectangle(
        [0, int(0.20 * height), 30, int(0.92 * height)],
        fill=(255, 255, 255),
    )

    # ---- Remove old legend (top-right for ratio, bottom-right for cosine) ----
    if is_cosine:
        # Bottom-right legend: avoid x tick labels near the very bottom.
        draw.rectangle(
            [int(0.74 * width), int(0.60 * height), width - 10, int(0.90 * height)],
            fill=(255, 255, 255),
        )
    else:
        draw.rectangle(
            [int(0.74 * width), int(0.03 * height), width - 10, int(0.24 * height)],
            fill=(255, 255, 255),
        )

    # Draw a new legend in one row at the top-right.
    legend_items = [
        ("patch", (185, 212, 230, 110), "p10–p90"),
        ("dash", (14, 70, 118, 255), "mean"),
    ]

    legend_y = int(0.04 * height)
    box_w, box_h = 54, 24
    sample_w = 72
    gap = 22

    def legend_item_width(kind: str, label: str) -> int:
        sample = box_w if kind == "patch" else sample_w
        return sample + 14 + int(draw.textlength(label, font=font_legend)) + gap

    total_w = sum(legend_item_width(kind, label) for kind, _, label in legend_items)
    cursor_x = int((width - total_w) / 2)

    # Add a subtle background behind the legend row for readability.
    legend_bg = [cursor_x - 12, legend_y - 10, cursor_x + total_w - gap + 12, legend_y + 10 + box_h + 10]
    draw.rectangle(legend_bg, fill=(255, 255, 255))

    for kind, color, label in legend_items:
        if kind == "patch":
            draw.rectangle(
                [cursor_x, legend_y + 10, cursor_x + box_w, legend_y + 10 + box_h],
                fill=color,
                outline=(40, 40, 40, 60),
            )
            cursor_x += box_w + 14
            draw.text((cursor_x, legend_y), label, fill=(20, 20, 20, 255), font=font_legend)
            cursor_x += int(draw.textlength(label, font=font_legend)) + gap
            continue

        x0 = cursor_x
        y0 = legend_y + 22
        x1 = cursor_x + sample_w
        if kind == "dash":
            dash = 10
            x = x0
            while x < x1:
                draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=6)
                x += dash * 2
        else:
            draw.line([(x0, y0), (x1, y0)], fill=color, width=7)
        cursor_x = x1 + 14
        draw.text((cursor_x, legend_y), label, fill=(20, 20, 20, 255), font=font_legend)
        cursor_x += int(draw.textlength(label, font=font_legend)) + gap

    # ---- Add / enlarge axis labels ----
    y_label = "Cosine mean" if is_cosine else "Norm ratio"
    yx, yy = int(0.06 * width), int(0.11 * height)
    y_w = int(draw.textlength(y_label, font=font_axis_bold))
    draw.rectangle([yx - 10, yy - 10, yx + y_w + 10, yy + 58], fill=(255, 255, 255))
    draw.text((yx, yy), y_label, fill=(20, 20, 20, 255), font=font_axis_bold)

    xlabel = "Timestep"
    x_w = draw.textlength(xlabel, font=font_axis_bold)
    draw.rectangle(
        [int(0.32 * width), height - 90, int(0.68 * width), height - 18],
        fill=(255, 255, 255),
    )
    draw.text(((width - x_w) / 2, height - 74), xlabel, fill=(20, 20, 20, 255), font=font_axis_bold)

    # Keep the original y-axis label text (e.g., "norm ratio" / "cosine_mean")
    # readable by slightly increasing its contrast with a small white pad behind.
    # This is a non-destructive nudge since we can't re-render tick labels.
    draw.rectangle([0, int(0.22 * height), 36, int(0.32 * height)], fill=(255, 255, 255))
    draw.text((6, int(0.24 * height)), y_label.lower().replace(" ", "_") if is_cosine else "norm ratio", fill=(20, 20, 20, 255), font=font_axis)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser(description="Restyle motivation plots (larger labels, horizontal legend).")
    parser.add_argument("--in", dest="in_path", required=True, help="Input PNG path.")
    parser.add_argument("--out", dest="out_path", required=True, help="Output PNG path.")
    args = parser.parse_args()

    stylize_plot(Path(args.in_path), Path(args.out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
