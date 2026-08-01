"""
Analyze a run_line_experiment.py results CSV (persona-only, e.g. results_k1_gender.csv)
against its baseline CSV (results_baseline.csv), which -- since run_line_experiment.py
made baseline opt-in and never duplicates it into sweep files -- now always lives in a
separate file. Mirrors analyze.py's metrics (accuracy/precision/recall/f1 vs
ground_truth, pass_rate vs baseline) adapted to this schema: entry_idx is the join key
(row_id+line_number would also work -- entry_idx is simpler since both files always
come from the same, unsampled, full dataset_location_issue.json).

Schema note: this matches user_prompt.py's 10-MR results (MR/Attribute/Value columns),
not the older persona_lib/generate_personas k=0..15 combo schema (k/combo_id/
combo_attrs/combo_values) -- don't point --results/--baseline at CSVs from before
run_line_experiment.py switched to user_prompt.py.

GUARDRAIL: pass_rate is only valid when results and baseline cover the identical set of
entry_idx with identical ground_truth (i.e. same dataset run) -- validate_paired_baseline
checks this and refuses to compute pass_rate otherwise, rather than silently comparing
against a mismatched baseline.

Usage:
  python3 analyze_line.py --results result/qwen3-coder-next/k1_gender/results_k1_gender.csv \\
    --baseline result/qwen3-coder-next/baseline/results_baseline.csv --prefix k1_gender
"""
import argparse
import os
import sys

import pandas as pd

GROUP_COLS = ["persona_row_id", "MR", "Attribute", "Value"]


def confusion_counts(df):
    """Return TP, FP, TN, FN, unparsed (predicted_label is null), and derived metrics."""
    unparsed = df["predicted_label"].isna().sum()
    parsed = df.dropna(subset=["predicted_label"])
    tp = ((parsed["ground_truth"] == 1) & (parsed["predicted_label"] == 1)).sum()
    fp = ((parsed["ground_truth"] == 0) & (parsed["predicted_label"] == 1)).sum()
    tn = ((parsed["ground_truth"] == 0) & (parsed["predicted_label"] == 0)).sum()
    fn = ((parsed["ground_truth"] == 1) & (parsed["predicted_label"] == 0)).sum()
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and precision == precision and recall == recall else float("nan"))
    return {"n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn, "unparsed": unparsed,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def validate_paired_baseline(persona_df, baseline_df):
    """Refuse pass_rate unless baseline and persona rows cover the identical entry_idx
    set with identical ground_truth -- guards against comparing against a baseline from
    a different dataset/session."""
    if baseline_df.empty:
        return "no rows found in the baseline file."
    b_gt = baseline_df[["entry_idx", "ground_truth"]].drop_duplicates()
    p_gt = persona_df[["entry_idx", "ground_truth"]].drop_duplicates()
    b_idx, p_idx = set(b_gt["entry_idx"]), set(p_gt["entry_idx"])
    if b_idx != p_idx:
        return ("baseline and persona rows cover different entry_idx sets "
                f"(persona-only: {sorted(p_idx - b_idx)}, baseline-only: {sorted(b_idx - p_idx)}). "
                "They must come from the SAME dataset run.")
    merged = p_gt.merge(b_gt, on="entry_idx", suffixes=("_persona", "_baseline"))
    mismatches = merged[merged["ground_truth_persona"] != merged["ground_truth_baseline"]]
    if len(mismatches):
        return ("ground_truth differs between baseline and persona rows for entry_idx "
                f"{mismatches['entry_idx'].tolist()} -- different dataset samples/snapshots; "
                "pass_rate would be meaningless.")
    return None


def compute_pass_rate(persona_df, baseline_df, group_cols=None):
    baseline_pred = baseline_df[["entry_idx", "predicted_label"]].rename(
        columns={"predicted_label": "baseline_predicted_label"})
    merged = persona_df.merge(baseline_pred, on="entry_idx", how="inner").dropna(
        subset=["predicted_label", "baseline_predicted_label"])
    merged = merged.copy()
    merged["passed"] = merged["predicted_label"] == merged["baseline_predicted_label"]
    if group_cols is None:
        return merged["passed"].mean() if len(merged) else float("nan")
    return merged.groupby(group_cols)["passed"].mean().rename("pass_rate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="persona/attribute-sweep results CSV, e.g. results_k1_goal.csv")
    ap.add_argument("--baseline", required=True, help="baseline results CSV, e.g. results_baseline.csv")
    ap.add_argument("--prefix", required=True,
                     help="output filename prefix, e.g. k1_goal -> summary_k1_goal_by_persona.csv / "
                          "summary_k1_goal_by_overview.csv")
    ap.add_argument("--out-dir", default=None, help="default: alongside --results")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        sys.exit(f"ERROR: {args.results} not found.")
    if not os.path.exists(args.baseline):
        sys.exit(f"ERROR: {args.baseline} not found.")

    persona_df = pd.read_csv(args.results)
    baseline_df = pd.read_csv(args.baseline)
    # baseline_df only needs entry_idx/ground_truth/predicted_label (validate_paired_
    # baseline, compute_pass_rate) -- it may come from an older-schema baseline file
    # that never had Attribute/Value columns at all, so those two are only touched on
    # persona_df, which does need them for GROUP_COLS.
    for df in (persona_df, baseline_df):
        df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    persona_df["Attribute"] = persona_df["Attribute"].fillna("").astype(str)
    persona_df["Value"] = persona_df["Value"].fillna("").astype(str)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    n_unparsed = persona_df["predicted_label"].isna().sum()
    print(f"Loaded {len(persona_df):,} persona rows from {args.results} ({n_unparsed:,} unparsed) "
          f"and {len(baseline_df):,} baseline rows from {args.baseline}.")

    error = validate_paired_baseline(persona_df, baseline_df)
    if error:
        print(f"\nWARNING: refusing to compute pass_rate -- {error}")

    # per-persona (finest grain: one row per persona_row_id/k/combo_id/combo_attrs/combo_values)
    by_persona = persona_df.groupby(GROUP_COLS, dropna=False).apply(
        lambda g: pd.Series(confusion_counts(g)), include_groups=False
    ).reset_index()
    if not error:
        pr = compute_pass_rate(persona_df, baseline_df, group_cols=GROUP_COLS).reset_index()
        by_persona = by_persona.merge(pr, on=GROUP_COLS, how="left")
    else:
        by_persona["pass_rate"] = float("nan")
    metric_cols = ["accuracy", "precision", "recall", "f1", "pass_rate"]
    for c in metric_cols:
        by_persona[c] = by_persona[c].round(4)
    by_persona = by_persona[GROUP_COLS + ["n", "unparsed"] + metric_cols]
    by_persona = by_persona.sort_values("Value")
    by_persona_path = os.path.join(out_dir, f"summary_{args.prefix}_by_persona.csv")
    by_persona.to_csv(by_persona_path, index=False)
    print(f"Wrote {by_persona_path}  ({len(by_persona)} personas)")

    # overview: one pooled row for the whole attribute sweep. Baseline is the same
    # fixed reference for every k/attribute run -- it lives once in the baseline
    # file's own directory (summary_baseline_overview.csv), not repeated here.
    persona_cc = confusion_counts(persona_df)
    persona_pass_rate = round(compute_pass_rate(persona_df, baseline_df), 4) if not error else float("nan")
    overview = pd.DataFrame([{
        "n": persona_cc["n"], "TP": persona_cc["TP"], "FP": persona_cc["FP"],
        "TN": persona_cc["TN"], "FN": persona_cc["FN"], "unparsed": persona_cc["unparsed"],
        "accuracy": round(persona_cc["accuracy"], 4), "precision": round(persona_cc["precision"], 4),
        "recall": round(persona_cc["recall"], 4), "f1": round(persona_cc["f1"], 4),
        "pass_rate": persona_pass_rate,
    }])
    overview_path = os.path.join(out_dir, f"summary_{args.prefix}_by_overview.csv")
    overview.to_csv(overview_path, index=False)
    print(f"Wrote {overview_path}")
    print("\nOverview (personas_all, pooled):")
    print(overview.to_string(index=False))

    # baseline's own summary lives once, in its own directory -- kept in sync on every run.
    baseline_cc = confusion_counts(baseline_df)
    baseline_summary = pd.DataFrame([{
        "n": baseline_cc["n"], "TP": baseline_cc["TP"], "FP": baseline_cc["FP"],
        "TN": baseline_cc["TN"], "FN": baseline_cc["FN"], "unparsed": baseline_cc["unparsed"],
        "accuracy": round(baseline_cc["accuracy"], 4), "precision": round(baseline_cc["precision"], 4),
        "recall": round(baseline_cc["recall"], 4), "f1": round(baseline_cc["f1"], 4),
    }])
    baseline_summary_path = os.path.join(os.path.dirname(os.path.abspath(args.baseline)),
                                          "summary_baseline_overview.csv")
    baseline_summary.to_csv(baseline_summary_path, index=False)
    print(f"Wrote {baseline_summary_path}  (kept in sync, same for every attribute/k run)")


if __name__ == "__main__":
    main()
