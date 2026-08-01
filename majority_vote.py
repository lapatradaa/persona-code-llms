"""
Collapse the same experiment run across multiple rounds (e.g. round1/round2/round3 of
run_line_experiment.py, same model/attr/dataset) into one majority-voted results.csv.

For each (persona_row_id, entry_idx) pair, takes predicted_label from every round file
and keeps the majority value (>=2 of 3 agreeing). Ties (no single majority label) are
written as unparsed (empty predicted_label), matching analyze_line.py's existing
"unparsed" handling, since a tied vote isn't a real prediction. ground_truth must be
identical across rounds for a pair -- it always should be, since every round runs off
the same dataset+entry-seed -- a mismatch means the rounds aren't comparable and that
pair is skipped with a warning rather than silently guessed.

All other columns (MR, Attribute, Value, persona_prompt, line_number, ground_truth,
row_id) are copied from the first round file, since they never vary by round. reason/
raw_response are replaced with a short record of what each round predicted, for
traceability back to the per-round CSVs.

Usage:
  python3 majority_vote.py \\
    --round-files results/qwen3-coder-next/k1_gender/results_k1_gender.csv \\
                  results_round2/qwen3-coder-next/k1_gender/results_k1_gender.csv \\
                  results_round3/qwen3-coder-next/k1_gender/results_k1_gender.csv \\
    --out results_majority/qwen3-coder-next/k1_gender/results_k1_gender.csv
"""
import argparse
import csv
import os
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

RESULT_COLS = ["entry_idx", "row_id", "persona_row_id", "MR", "Attribute", "Value",
               "persona_prompt", "line_number", "ground_truth", "predicted_label",
               "reason", "raw_response"]


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def majority_label(labels):
    """labels: list of raw predicted_label strings (possibly '' for unparsed).
    Returns the winning int label, or None if no single majority (tie or all-unparsed)."""
    valid = [int(l) for l in labels if l not in (None, "")]
    if not valid:
        return None
    counts = Counter(valid)
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round-files", nargs="+", required=True,
                     help="results CSVs, one per round, same model/attr/dataset, same schema")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.round_files) < 2:
        sys.exit("ERROR: need at least 2 round files to vote across.")

    round_rows = [load(p) for p in args.round_files]
    indexed = [{(int(r["persona_row_id"]), int(r["entry_idx"])): r for r in rows} for rows in round_rows]

    key_sets = [set(idx.keys()) for idx in indexed]
    common_keys = set.intersection(*key_sets)
    all_keys = set.union(*key_sets)
    missing = all_keys - common_keys
    if missing:
        print(f"WARNING: {len(missing)} (persona_row_id, entry_idx) pairs missing from at least "
              f"one round file -- skipped (not present in all {len(args.round_files)} rounds).",
              file=sys.stderr)

    out_rows = []
    gt_mismatches = 0
    ties = 0
    for key in sorted(common_keys):
        rows_for_key = [idx[key] for idx in indexed]
        gts = {int(r["ground_truth"]) for r in rows_for_key}
        if len(gts) > 1:
            gt_mismatches += 1
            continue
        labels = [r["predicted_label"] for r in rows_for_key]
        maj = majority_label(labels)
        if maj is None:
            ties += 1
        base = rows_for_key[0]
        row = {col: base.get(col, "") for col in RESULT_COLS}
        row["predicted_label"] = maj if maj is not None else ""
        row["reason"] = f"majority vote across {len(args.round_files)} rounds; per-round predicted_label={labels}"
        row["raw_response"] = f"per-round predicted_label={labels}"
        out_rows.append(row)

    if gt_mismatches:
        print(f"WARNING: {gt_mismatches} pairs had mismatched ground_truth across rounds -- skipped.",
              file=sys.stderr)
    if ties:
        print(f"NOTE: {ties} pairs had no majority (tie) and were written as unparsed.", file=sys.stderr)

    out_rows.sort(key=lambda r: (int(r["persona_row_id"]), int(r["row_id"]), int(r["entry_idx"])))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} majority-voted rows to {args.out}")


if __name__ == "__main__":
    main()
