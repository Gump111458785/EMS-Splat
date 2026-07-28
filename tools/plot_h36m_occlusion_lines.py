from pathlib import Path
from xml.sax.saxutils import escape


OCCLUSION_RATIOS = [0.0, 0.2, 0.4, 0.6]

METRICS = [
    (
        "MPJPE",
        "mm, lower is better",
        {"Baseline": [27.74, 54.41, 126.62, 168.99], "Ours": [26.11, 39.17, 64.55, 83.46]},
    ),
    (
        "PA-MPJPE",
        "mm, lower is better",
        {"Baseline": [29.91, 61.27, 132.46, 159.46], "Ours": [27.85, 42.03, 66.59, 83.19]},
    ),
    (
        "PCK@150",
        "%, higher is better",
        {"Baseline": [99.19, 90.75, 69.30, 57.77], "Ours": [99.41, 98.20, 93.85, 89.21]},
    ),
]

COLORS = {"Baseline": "#6B7280", "Ours": "#D55E00"}
WIDTH = 1320
HEIGHT = 440
PANEL_W = 380
PANEL_H = 260
LEFT = 70
TOP = 96
GAP = 45


def nice_limits(values):
    lo = min(values)
    hi = max(values)
    pad = (hi - lo) * 0.12 or 1
    return max(0, lo - pad), hi + pad


def scale_x(x, panel_left):
    return panel_left + (x - min(OCCLUSION_RATIOS)) / (max(OCCLUSION_RATIOS) - min(OCCLUSION_RATIOS)) * PANEL_W


def scale_y(y, ymin, ymax):
    return TOP + PANEL_H - (y - ymin) / (ymax - ymin) * PANEL_H


def line(points, color):
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />'
    )


def circle(x, y, color):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="white" stroke="{color}" stroke-width="3" />'


def text(x, y, value, size=13, anchor="middle", weight="400", color="#111827", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{color}"{transform}>{escape(value)}</text>'
    )


def panel(metric_idx, title, subtitle, series):
    panel_left = LEFT + metric_idx * (PANEL_W + GAP)
    values = [v for points in series.values() for v in points]
    ymin, ymax = nice_limits(values)
    parts = []

    parts.append(f'<rect x="{panel_left}" y="{TOP}" width="{PANEL_W}" height="{PANEL_H}" fill="#FAFAFA" />')
    for i in range(5):
        y = TOP + i * PANEL_H / 4
        tick_value = ymax - i * (ymax - ymin) / 4
        parts.append(f'<line x1="{panel_left}" y1="{y:.1f}" x2="{panel_left + PANEL_W}" y2="{y:.1f}" stroke="#D1D5DB" stroke-dasharray="4 5" />')
        parts.append(text(panel_left - 12, y + 4, f"{tick_value:.0f}", size=11, anchor="end", color="#4B5563"))

    parts.append(f'<line x1="{panel_left}" y1="{TOP + PANEL_H}" x2="{panel_left + PANEL_W}" y2="{TOP + PANEL_H}" stroke="#111827" stroke-width="1.5" />')
    parts.append(f'<line x1="{panel_left}" y1="{TOP}" x2="{panel_left}" y2="{TOP + PANEL_H}" stroke="#111827" stroke-width="1.5" />')

    for ratio in OCCLUSION_RATIOS:
        x = scale_x(ratio, panel_left)
        parts.append(f'<line x1="{x:.1f}" y1="{TOP + PANEL_H}" x2="{x:.1f}" y2="{TOP + PANEL_H + 6}" stroke="#111827" />')
        parts.append(text(x, TOP + PANEL_H + 24, f"{ratio:g}", size=12, color="#374151"))

    for method, values in series.items():
        points = [(scale_x(x, panel_left), scale_y(y, ymin, ymax)) for x, y in zip(OCCLUSION_RATIOS, values)]
        parts.append(line(points, COLORS[method]))
        for x, y, value in [(x, y, v) for (x, y), v in zip(points, values)]:
            parts.append(circle(x, y, COLORS[method]))
            offset = -12 if method == "Baseline" else 20
            parts.append(text(x, y + offset, f"{value:.2f}", size=10, color=COLORS[method]))

    parts.append(text(panel_left + PANEL_W / 2, TOP - 26, title, size=18, weight="700"))
    parts.append(text(panel_left + PANEL_W / 2, TOP - 8, subtitle, size=12, color="#4B5563"))
    parts.append(text(panel_left + PANEL_W / 2, TOP + PANEL_H + 48, "Occlusion Ratio", size=13, color="#111827"))
    return "\n".join(parts)


def main():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white" />',
        text(WIDTH / 2, 38, "Human3.6M Occlusion Robustness", size=22, weight="700"),
    ]

    for idx, metric in enumerate(METRICS):
        parts.append(panel(idx, *metric))

    legend_x = WIDTH - 210
    legend_y = 32
    for i, method in enumerate(("Baseline", "Ours")):
        y = legend_y + i * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{COLORS[method]}" stroke-width="3" />')
        parts.append(circle(legend_x + 16, y, COLORS[method]))
        parts.append(text(legend_x + 44, y + 5, method, size=13, anchor="start", color="#111827"))

    parts.append("</svg>")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h36m_occlusion_line_chart.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
