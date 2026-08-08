"""Plot pass_rate by persona attribute (Gender, ReviewStyle, Seniority), comparing
whichever models currently have data. Majority-vote variant of plot_pass_rate.py --
reads from results_majority/ (majority_vote.py's 2-of-3 consensus across
results/, results_round2/, results_round3/) instead of the single round1 results/
tree, so the plotted pass_rate is the denoised 3-round consensus rather than one
run's per-call jitter.

Reads summary_k1_<attr>_by_persona.csv for each attribute under each model's
results_majority/<model>/k1_<attr>/ folder (run_line_experiment.py + analyze_line.py's
user_prompt.py-sourced MR/Attribute/Value schema -- not the older result/ tree's
k/combo_id/combo_attrs/combo_values schema) and renders a 1x3 grid of grouped
horizontal bar charts, one per attribute, one bar per available model per persona
value.

A model only needs SOME of the three attributes' summaries to appear -- missing
ones are skipped per-panel, so this can be re-run at any point in a multi-model,
multi-attribute campaign (e.g. qwen-only today, qwen+DeepSeek once DeepSeek's
sweeps finish) without editing the script.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

ATTRS = ["Gender", "ReviewStyle", "Seniority"]

# candidate models, in a fixed preferred order -- only ones with data actually get
# plotted. Order also fixes color assignment (never cycled/reassigned by what's present).
CANDIDATE_MODELS = ["qwen3-coder-next", "deepseek-v4-flash"]
MODEL_DIRS = {m: f"results_majority/{m}" for m in CANDIDATE_MODELS}

# dataviz reference palette (references/palette.md), slots 1 (blue) + 8 (red) --
# validated via scripts/validate_palette.js: CVD dE 21.6 (protan), normal-vision dE 32.3,
# both well clear of the 8/15 floors.
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


# which (model, attr) pairs actually have a summary file on disk right now
available = {
    (model, attr): summary_path(MODEL_DIRS[model], attr)
    for model in CANDIDATE_MODELS
    for attr in ATTRS
    if os.path.exists(summary_path(MODEL_DIRS[model], attr))
}
models_present = [m for m in CANDIDATE_MODELS if any(a for (mm, a) in available if mm == m)]
if not models_present:
    raise SystemExit("No summary_k1_<attr>_by_persona.csv files found for any model/attribute yet.")

print(f"Plotting {len(models_present)} model(s): {', '.join(models_present)}")
for attr in ATTRS:
    missing = [m for m in models_present if (m, attr) not in available]
    if missing:
        print(f"  {attr}: no data yet for {', '.join(missing)} -- panel will show only "
              f"{', '.join(m for m in models_present if m not in missing)}")

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor=SURFACE)
title_models = " vs ".join(models_present)

gap = 0.1  # surface gap between bars in a group

# Gender/ReviewStyle have 12 categories each vs. Seniority's 9, so they read as
# thinner-barred by default -- give them a bit more bar thickness specifically.
BAR_HEIGHT_SCALE = {"Gender": 0.8, "ReviewStyle": 0.8, "Seniority": 0.72}

for ax, attr in zip(axes.flat, ATTRS):
    bar_height = BAR_HEIGHT_SCALE[attr] / max(len(models_present), 1)
    attr_models = [m for m in models_present if (m, attr) in available]
    if not attr_models:
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{attr} (no data yet)", fontsize=12, color=TEXT_MUTED, loc="left", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        continue

    frames = {m: pd.read_csv(available[(m, attr)]).set_index("Value")["pass_rate"]
              for m in attr_models}

    # sort by the first available model's pass_rate so bar order is stable/meaningful
    combo_values = frames[attr_models[0]].sort_values(ascending=True).index.tolist()
    y = range(len(combo_values))

    ax.set_facecolor(SURFACE)
    n = len(attr_models)
    for i, model in enumerate(attr_models):
        offset = (i - (n - 1) / 2) * (bar_height + gap)
        values = frames[model].reindex(combo_values)
        bars = ax.barh([yi + offset for yi in y], values, height=bar_height,
                        color=MODEL_COLORS[model], label=model, zorder=3)
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{value:.2f}", va="center", ha="left", fontsize=20, color=TEXT_PRIMARY)

    ax.set_yticks(list(y))
    ax.set_yticklabels(combo_values)
    ax.margins(y=0.02)
    ax.set_xlim(0.7, 1.05)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    title = attr if n == len(models_present) else f"{attr} ({', '.join(attr_models)} only)"
    ax.set_title(title, fontsize=24, color=TEXT_PRIMARY, loc="left", pad=4)
    ax.set_xlabel("pass rate", fontsize=22, color=TEXT_MUTED)
    ax.tick_params(axis="y", labelsize=22, colors=TEXT_PRIMARY, length=0)
    ax.tick_params(axis="x", labelsize=20, colors=TEXT_MUTED, length=0)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)

if len(models_present) > 1:
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in models_present]
    fig.legend(handles, models_present, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(models_present), frameon=False, fontsize=22, labelcolor=TEXT_PRIMARY)

fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=6)

out_path = "pass_rate_majority_" + "_vs_".join(models_present) + ".pdf"
fig.savefig(out_path, facecolor=SURFACE)
print(f"Saved {out_path}")

png_path = "pass_rate_majority_" + "_vs_".join(models_present) + ".png"
fig.savefig(png_path, facecolor=SURFACE, dpi=200)
print(f"Saved {png_path}")
