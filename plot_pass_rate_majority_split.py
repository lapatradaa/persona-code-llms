"""Same data/logic as plot_pass_rate_majority.py, but saves each attribute
(Gender, ReviewStyle, Seniority) as its own standalone PDF instead of a 1x3 grid
in one file -- e.g. for dropping a single chart into a slide/doc without cropping.

Text sizing: title/axis-label/tick-label/legend fontsize=14; only the per-bar
pass_rate value annotations (the "0.99" text at the end of each bar) are smaller,
fontsize=12, so they read as a value label rather than competing with the axis text.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

ATTRS = ["Gender", "ReviewStyle", "Seniority"]

CANDIDATE_MODELS = ["qwen3-coder-next", "deepseek-v4-flash"]
MODEL_DIRS = {m: f"results_majority/{m}" for m in CANDIDATE_MODELS}

MODEL_COLORS = {
    "qwen3-coder-next": "#2a78d6",
    "deepseek-v4-flash": "#e34948",
}
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"

LABEL_FONTSIZE = 16
VALUE_FONTSIZE = 14


def summary_path(model_dir, attr):
    key = attr.lower()
    return f"{model_dir}/k1_{key}/summary_k1_{key}_by_persona.csv"


available = {
    (model, attr): summary_path(MODEL_DIRS[model], attr)
    for model in CANDIDATE_MODELS
    for attr in ATTRS
    if os.path.exists(summary_path(MODEL_DIRS[model], attr))
}
models_present = [m for m in CANDIDATE_MODELS if any(a for (mm, a) in available if mm == m)]
if not models_present:
    raise SystemExit("No summary_k1_<attr>_by_persona.csv files found for any model/attribute yet.")

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

title_models = " vs ".join(models_present)
bar_height = 0.72 / max(len(models_present), 1)
gap = 0.02

for attr in ATTRS:
    attr_models = [m for m in models_present if (m, attr) in available]
    if not attr_models:
        print(f"{attr}: no data yet -- skipped")
        continue

    frames = {m: pd.read_csv(available[(m, attr)]).set_index("Value")["pass_rate"]
              for m in attr_models}
    combo_values = frames[attr_models[0]].sort_values(ascending=True).index.tolist()
    y = range(len(combo_values))

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(combo_values) + 1.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(attr_models)
    for i, model in enumerate(attr_models):
        offset = (i - (n - 1) / 2) * (bar_height + gap)
        values = frames[model].reindex(combo_values)
        bars = ax.barh([yi + offset for yi in y], values, height=bar_height,
                        color=MODEL_COLORS[model], label=model, zorder=3)
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                        f"{value:.2f}", va="center", ha="left",
                        fontsize=VALUE_FONTSIZE, color=TEXT_PRIMARY)

    ax.set_yticks(list(y))
    ax.set_yticklabels(combo_values)
    ax.set_xlim(0.5, 1.12)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("pass rate", fontsize=LABEL_FONTSIZE, color=TEXT_MUTED)
    ax.tick_params(axis="y", labelsize=LABEL_FONTSIZE, colors=TEXT_PRIMARY, length=0)
    ax.tick_params(axis="x", labelsize=LABEL_FONTSIZE, colors=TEXT_MUTED, length=0)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if n > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in attr_models]
        fig.legend(handles, attr_models, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=n, frameon=False, fontsize=LABEL_FONTSIZE, labelcolor=TEXT_PRIMARY)

    out_path = f"pass_rate_majority_{attr.lower()}_" + "_vs_".join(attr_models) + ".pdf"
    fig.savefig(out_path, facecolor=SURFACE)

    os.makedirs("docs/figures", exist_ok=True)
    png_path = f"docs/figures/pass_rate_{attr.lower()}.png"
    fig.savefig(png_path, facecolor=SURFACE, dpi=200)

    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {png_path}")
