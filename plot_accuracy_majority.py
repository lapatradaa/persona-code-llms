"""Accuracy by persona attribute (Baseline, Gender, ReviewStyle, Seniority), comparing
whichever models currently have data.

Reads the majority-voted (2-of-3 rounds) results in results_majority/, mirroring
plot_pass_rate_majority_split.py's data source but plotting the accuracy column
(vs. ground_truth) instead of pass_rate (vs. baseline). Baseline is the single
no-persona accuracy from summary_baseline_overview.csv; the other three panels read
summary_k1_<attr>_by_persona.csv.
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


def summary_path(model_dir, attr):
    key = attr.lower()
    return f"{model_dir}/k1_{key}/summary_k1_{key}_by_persona.csv"


def baseline_path(model_dir):
    return f"{model_dir}/baseline/summary_baseline_overview.csv"


baseline_available = {
    model: baseline_path(MODEL_DIRS[model])
    for model in CANDIDATE_MODELS
    if os.path.exists(baseline_path(MODEL_DIRS[model]))
}
available = {
    (model, attr): summary_path(MODEL_DIRS[model], attr)
    for model in CANDIDATE_MODELS
    for attr in ATTRS
    if os.path.exists(summary_path(MODEL_DIRS[model], attr))
}
models_present = [
    m for m in CANDIDATE_MODELS
    if m in baseline_available or any(a for (mm, a) in available if mm == m)
]
if not models_present:
    raise SystemExit("No summary CSVs found for any model/attribute yet.")

print(f"Plotting {len(models_present)} model(s): {', '.join(models_present)}")

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

# Labels read tiny once this 38in-wide figure is shrunk to the paper's ~7.16in
# column width -- what matters for legibility there is fontsize_pt / fig_width_inch.
# Scaling fonts up all the way to match 10pt body text needs more horizontal room
# per panel than 4-across has (long labels like "SecurityFocused" start colliding
# with the neighboring panel); FONT_SCALE below is the largest bump that still
# fits this 4-panel-in-a-row layout without overlap.
FONT_SCALE = 1.4

panels = ["Baseline"] + ATTRS
# Baseline only has one bar per model (vs. many persona values in the other panels),
# so give it a narrower column instead of the same width as the rest. Each panel
# pair gets its own blank spacer column (instead of one uniform wspace) -- all three
# spacer columns are the same width so the gaps between all four panels look equal.
GAP = 0.45
fig, all_axes = plt.subplots(1, 7, figsize=(38, 15), facecolor=SURFACE,
                              gridspec_kw={"width_ratios": [0.35, GAP, 1, GAP, 1, GAP, 1],
                                           "wspace": 0})
for spacer_i in (1, 3, 5):
    all_axes[spacer_i].set_visible(False)
axes = [all_axes[0], all_axes[2], all_axes[4], all_axes[6]]
title_models = " vs ".join(models_present)

gap = 0.05

# Gender/ReviewStyle have 12 categories each vs. Seniority's 9, so they read as
# thinner-barred by default -- give them a bit more bar thickness specifically
# (matches plot_pass_rate_majority.py's sizing so bar weight feels consistent
# with the larger label font).
BAR_HEIGHT_SCALE = {"Gender": 0.92, "ReviewStyle": 0.92, "Seniority": 0.8}

for ax, attr in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    bar_height = BAR_HEIGHT_SCALE.get(attr, 0.8) / max(len(models_present), 1)

    if attr == "Baseline":
        attr_models = [m for m in models_present if m in baseline_available]
        if not attr_models:
            ax.set_title("Baseline (no data yet)", fontsize=12, color=TEXT_MUTED, loc="left", pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        # vertical bars (flipped from the horizontal layout of the other panels) --
        # reads narrower/slimmer since there's only one bar per model to show.
        values = {m: pd.read_csv(baseline_available[m])["accuracy"].iloc[0] for m in attr_models}
        x = range(len(attr_models))
        bars = ax.bar(list(x), [values[m] for m in attr_models], width=0.55,
                       color=[MODEL_COLORS[m] for m in attr_models], zorder=3)
        for bar, model in zip(bars, attr_models):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{values[model]:.2f}", va="bottom", ha="center",
                    fontsize=20 * FONT_SCALE, color=TEXT_PRIMARY)
        # no x-tick labels here -- the legend above already maps color to model,
        # and the model names were the thing getting clipped against the left
        # edge of the figure at this font size.
        ax.set_xticks(list(x))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 0.75)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75])
        title = "Baseline" if len(attr_models) == len(models_present) else f"Baseline ({', '.join(attr_models)} only)"
        ax.set_title(title, fontsize=24 * FONT_SCALE, color=TEXT_PRIMARY, loc="left", pad=4 * FONT_SCALE)
        ax.set_ylabel("accuracy", fontsize=22 * FONT_SCALE, color=TEXT_MUTED)
        ax.tick_params(axis="x", labelsize=20 * FONT_SCALE, colors=TEXT_PRIMARY, length=0)
        ax.tick_params(axis="y", labelsize=20 * FONT_SCALE, colors=TEXT_MUTED, length=0)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8 * FONT_SCALE, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS_COLOR)
        continue

    attr_models = [m for m in models_present if (m, attr) in available]
    if not attr_models:
        ax.set_title(f"{attr} (no data yet)", fontsize=12, color=TEXT_MUTED, loc="left", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        continue

    frames = {m: pd.read_csv(available[(m, attr)]).set_index("Value")["accuracy"]
              for m in attr_models}
    combo_values = frames[attr_models[0]].sort_values(ascending=True).index.tolist()
    y = range(len(combo_values))
    n = len(attr_models)
    for i, model in enumerate(attr_models):
        offset = (i - (n - 1) / 2) * (bar_height + gap)
        vals = frames[model].reindex(combo_values)
        bars = ax.barh([yi + offset for yi in y], vals, height=bar_height,
                        color=MODEL_COLORS[model], label=model, zorder=3)
        for bar, value in zip(bars, vals):
            if pd.notna(value):
                ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                        f"{value:.2f}", va="center", ha="left",
                        fontsize=16 * FONT_SCALE, color=TEXT_PRIMARY)
    ax.set_yticks(list(y))
    ax.set_yticklabels(combo_values)
    ax.margins(y=0.035)
    ax.set_xlim(0, 0.75)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75])
    title = attr if attr_models == models_present else f"{attr} ({', '.join(attr_models)} only)"
    ax.set_title(title, fontsize=24 * FONT_SCALE, color=TEXT_PRIMARY, loc="left", pad=4 * FONT_SCALE)
    ax.set_xlabel("accuracy", fontsize=22 * FONT_SCALE, color=TEXT_MUTED)
    ax.tick_params(axis="y", labelsize=22 * FONT_SCALE, colors=TEXT_PRIMARY, length=0)
    ax.tick_params(axis="x", labelsize=20 * FONT_SCALE, colors=TEXT_MUTED, length=0)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8 * FONT_SCALE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)

if len(models_present) > 1:
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in models_present]
    fig.legend(handles, models_present, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(models_present), frameon=False, fontsize=22 * FONT_SCALE, labelcolor=TEXT_PRIMARY)

# tight_layout would recompute wspace and clobber the per-gap spacer columns above,
# so set margins directly instead.
fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.06)

out_path = "accuracy_majority_" + "_vs_".join(models_present) + ".pdf"
fig.savefig(out_path, facecolor=SURFACE)
print(f"Saved {out_path}")

png_path = "accuracy_majority_" + "_vs_".join(models_present) + ".png"
fig.savefig(png_path, facecolor=SURFACE, dpi=200)
print(f"Saved {png_path}")
