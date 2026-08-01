"""
Line-level issue-localization runs: for each (code, line_number) entry in a
dataset_location_issue.json-style file, ask whether that specific line contains an
issue, under each of the 10 MR personas from user_prompt.py (MR1 Gender ... MR10
Domain) plus an optional no-persona baseline.

  - Dataset: JSON list of {row_id, code, line_number, issue}.
  - Personas: user_prompt.build() -- always exactly 10 fixed rows, one per MR/
    attribute, each already carrying its own SystemPrompt (persona identity) and
    UserPrompt (TASK_BLOCK, the task instructions). There is no combo/value sweep here
    (that was generate_personas.build() + persona_lib's job, for the older k=0..15
    multi-attribute study) -- every persona row is one attribute at its representative
    ontology value, matching Table I.
  - Prompt: SystemPrompt and UserPrompt go out as separate SystemMessage/HumanMessage
    (persona identity vs. task instructions, per system-vs-user-prompt best practice).
    UserPrompt's literal "{code}"/"{line}" placeholders get substituted via str.replace
    per entry -- NOT str.format, since task_prompt.md's own output-format example
    ("{decision}, {reason}") must stay literal, and source code routinely contains its
    own '{'/'}'. The baseline row's SystemPrompt is BASELINE_SYSTEM_PROMPT ("You are a
    software engineer.") -- the same generic base identity every persona sentence
    already builds on top of, just with no attribute-specific clause -- so it goes out
    as a SystemMessage + HumanMessage pair exactly like every real persona, never a
    bare HumanMessage. This must stay identical across every model's baseline run --
    never model-specific wording -- so baselines are comparable across models.
  - results.csv's "persona_prompt" column logs only the system prompt (the persona
    identity actually sent) -- not the user prompt, which is task_prompt.md and is
    identical across every row anyway, so logging it per-row would be redundant. For
    baseline this column is "You are a software engineer." (BASELINE_SYSTEM_PROMPT),
    same for every model.
  - Response: "{decision}, {reason}" (e.g. "1, this line creates a null pointer
    exception.").

Dry-run by default (no API calls, no cost) -- pass --live to execute. Baseline is
opt-in via --with-baseline, never implicit -- a persona sweep run can never
accidentally re-bill baseline calls just because a flag was forgotten. Run baseline
once per model (--with-baseline, no other persona rows to add since build() is fixed),
then every later sweep omits --with-baseline entirely and compares against that
baseline file by joining on entry_idx/row_id -- no duplication.

Note: this results.csv schema (entry_idx, row_id, persona_row_id, MR, Attribute,
Value, persona_prompt, ...) does NOT match the older generate_personas/persona_lib-era
schema (k, combo_id, combo_attrs, combo_values, one column per attribute) -- point
--out-dir/--results-name at a fresh file, not an existing results.csv from before this
script switched personas sources.

Example (OpenRouter):
  # one-time baseline for a model:
  python3 run_line_experiment.py --live --with-baseline \\
    --model openai:qwen/qwen3-coder-next --base-url https://openrouter.ai/api/v1 \\
    --api-key-env OPENROUTER_API_KEY --out-dir result/qwen3-coder-next \\
    --results-name results_mr_baseline.csv --run-info-name run_info_mr_baseline.json

  # later attribute sweeps, same model -- no --with-baseline, so baseline is never re-billed:
  python3 run_line_experiment.py --live --workers 8 \\
    --model openai:qwen/qwen3-coder-next --base-url https://openrouter.ai/api/v1 \\
    --api-key-env OPENROUTER_API_KEY --out-dir result/qwen3-coder-next \\
    --results-name results_mr.csv --run-info-name run_info_mr.json
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

# A model occasionally loops and produces a runaway response (observed: one reply
# >250KB) that blows past Python's default 131072-byte csv field limit and crashes
# resume/sort on that row.
csv.field_size_limit(sys.maxsize)

from user_prompt import ATTRS, BASELINE_SYSTEM_PROMPT, TASK_BLOCK, build, build_attribute_sweep

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "issue_location", "dataset_location_issue.json")
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")

RESULT_COLS = ["entry_idx", "row_id", "persona_row_id", "MR", "Attribute", "Value",
               "persona_prompt", "line_number", "ground_truth", "predicted_label",
               "reason", "raw_response"]

BASELINE_ROW = {"RowID": 0, "MR": "baseline", "Attribute": "", "Value": "",
                 "SystemPrompt": BASELINE_SYSTEM_PROMPT, "UserPrompt": TASK_BLOCK}

RESPONSE_RE = re.compile(r"^\s*([01])\s*,\s*(.*)", re.DOTALL)


def load_api_key(api_key_env="OPENAI_API_KEY"):
    """Project .env wins over an inherited shell env var -- otherwise a stale/
    placeholder value exported in the shell profile silently shadows the real key."""
    if os.path.exists(ENV_PATH):
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH, override=True)
    return os.environ.get(api_key_env) or None


def load_entries(dataset_path, n=None, seed=0):
    """All entries by default (dataset_location_issue.json is already a curated,
    class-balanced sample). --n-entries/--entry-seed optionally subsample for cheap
    smoke tests."""
    with open(dataset_path, encoding="utf-8") as f:
        entries = json.load(f)
    if n is not None and n < len(entries):
        entries = random.Random(seed).sample(entries, n)
    return entries


def build_chat_model(model, api_key, max_output_tokens=None, base_url=None, reasoning=None):
    """reasoning controls OpenRouter's unified `reasoning` field for hybrid reasoning
    models (e.g. deepseek/deepseek-v4-flash): "off" disables the hidden thinking pass
    entirely (this task only needs a one-line decision + reason, not a CoT), so it can
    run at qwen3-coder-next-like speed instead of burning most of max_output_tokens on
    invisible reasoning tokens before ever reaching the visible answer. "low"/"medium"/
    "high" cap its effort instead of disabling it. Non-reasoning models ignore the field."""
    from langchain.chat_models import init_chat_model
    kwargs = dict(temperature=0, api_key=api_key)
    if max_output_tokens:
        kwargs["max_tokens"] = max_output_tokens
    if base_url:
        kwargs["base_url"] = base_url
    if reasoning:
        if reasoning == "off":
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        else:
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning}}
    return init_chat_model(model, **kwargs)


def build_prompt(user_prompt, code, line_number):
    """Substitute {code}/{line} into the user prompt (TASK_BLOCK) only -- those
    placeholders never appear in a system prompt (persona sentences)."""
    return user_prompt.replace("{code}", code).replace("{line}", str(line_number))


def call_llm(chat_model, system_prompt, user_prompt, retries=2):
    """A flaky network/gateway response (e.g. a malformed chunked response from
    OpenRouter under load) must not crash the whole batch -- that would discard
    every other call already in flight in the same ThreadPoolExecutor round, wasting
    real money. Treat any exception the same as an unparsed response: recorded, not
    fatal, retryable on the next resumed run. Also retries on an empty response --
    a reasoning model can exhaust its token budget mid-thought and return no visible
    content at all; a retry is cheaper than silently losing the row."""
    # Every row (including baseline) has a non-empty system_prompt now, so this always
    # sends a SystemMessage + HumanMessage pair -- the `if system_prompt` guard is just
    # a safety fallback to a bare HumanMessage if system_prompt is ever empty.
    messages = ([SystemMessage(content=system_prompt)] if system_prompt else []) + [HumanMessage(content=user_prompt)]
    text = ""
    for attempt in range(retries + 1):
        try:
            resp = chat_model.invoke(messages)
            text = resp.content or ""
        except Exception as exc:
            return None, None, f"[run_line_experiment ERROR: {type(exc).__name__}: {exc}]"
        if text.strip():
            break
    m = RESPONSE_RE.match(text)
    if m:
        return int(m.group(1)), m.group(2).strip(), text
    return None, None, text


def load_done_pairs(results_path):
    """(persona_row_id, entry_idx) pairs already present in results.csv, for resume."""
    if not os.path.exists(results_path):
        return set()
    done = set()
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((int(row["persona_row_id"]), int(row["entry_idx"])))
    return done


def iter_pairs(personas, entries, done_pairs):
    for p in personas:
        for ei, e in enumerate(entries):
            if (p["RowID"], ei) in done_pairs:
                continue
            yield p, ei, e


def write_run_info(out_dir, model, extra, filename="run_info.json"):
    info = {"model": model, "timestamp": datetime.now(timezone.utc).isoformat(), **extra}
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path


def sort_results_file(results_path):
    with open(results_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["persona_row_id"]), int(r["row_id"]), int(r["entry_idx"])))
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()
        writer.writerows(rows)


def run_sync(chat_model, personas, entries, results_path, workers):
    done_pairs = load_done_pairs(results_path)
    pairs = list(iter_pairs(personas, entries, done_pairs))
    total_planned = len(personas) * len(entries)
    if done_pairs:
        print(f"Resuming: {len(done_pairs):,}/{total_planned:,} pairs already done, "
              f"{len(pairs):,} remaining.")
    if not pairs:
        print("Nothing to do -- all pairs already present in results.csv.")
        return

    file_exists = os.path.exists(results_path)
    lock = threading.Lock()
    f = open(results_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(RESULT_COLS)
        f.flush()

    done = 0

    def work(item):
        p, ei, e = item
        user_prompt = build_prompt(p["UserPrompt"], e["code"], e["line_number"])
        pred, reason, raw = call_llm(chat_model, p["SystemPrompt"], user_prompt)
        return p, ei, e, pred, reason, raw

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(work, item) for item in pairs]
            for fut in as_completed(futures):
                p, ei, e, pred, reason, raw = fut.result()
                row = ([ei, e["row_id"], p["RowID"], p["MR"], p["Attribute"], p["Value"],
                        p["SystemPrompt"].replace("\n", " "),
                        e["line_number"], e["issue"], pred,
                        reason.replace("\n", " ") if reason else reason,
                        raw.replace("\n", " ")])
                with lock:
                    writer.writerow(row)
                    done += 1
                    if done % 100 == 0:
                        f.flush()
                        f.close()
                        sort_results_file(results_path)
                        f = open(results_path, "a", newline="", encoding="utf-8")
                        writer = csv.writer(f)
                        print(f"  {done:,}/{len(pairs):,} calls done... (sorted)")
    finally:
        f.close()
    print(f"\nWrote {done:,} new rows to {results_path}")

    if done % 100 != 0:
        sort_results_file(results_path)


def main():
    ap = argparse.ArgumentParser(description="Run the 10 MR personas (+ optional no-persona baseline) on the line-level issue task.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--n-entries", type=int, default=None, help="subsample this many entries (default: all)")
    ap.add_argument("--entry-seed", type=int, default=0)
    ap.add_argument("--model", default="openai:qwen/qwen3-coder-next",
                     help="'provider:model' for LangChain init_chat_model, e.g. openai:qwen/qwen3-coder-next")
    ap.add_argument("--base-url", default=None,
                     help="OpenAI-compatible endpoint override, e.g. https://openrouter.ai/api/v1 for OpenRouter")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--reasoning", choices=["off", "low", "medium", "high"], default=None,
                     help="OpenRouter 'reasoning' field for hybrid reasoning models (e.g. "
                          "deepseek/deepseek-v4-flash). This task only needs a one-line decision "
                          "+ reason, so 'off' skips the hidden thinking pass entirely -- use it to "
                          "bring a reasoning model's latency in line with a non-reasoning model like "
                          "qwen3-coder-next. Ignored by models that don't support it.")
    ap.add_argument("--attr", choices=ATTRS, default=None,
                     help="restrict to a full per-value sweep of one attribute (e.g. Gender -- "
                          "all 12 values) instead of the default 10-row build() (one representative "
                          "value per attribute).")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-name", default="results.csv")
    ap.add_argument("--run-info-name", default="run_info.json")
    ap.add_argument("--with-baseline", action="store_true",
                     help="include a fresh no-persona baseline call via the API (costs money). "
                          "Omit this for persona sweeps once a baseline already exists in its own "
                          "results file (e.g. result/<model>/results_mr_baseline.csv) -- baseline is "
                          "opt-in, never implicit, specifically so a persona run can't accidentally "
                          "re-bill baseline. Compare a sweep against baseline later by joining on "
                          "entry_idx/row_id across files, rather than duplicating baseline rows into "
                          "every sweep's output.")
    ap.add_argument("--baseline-only", action="store_true",
                     help="run ONLY the no-persona baseline row -- no MR personas at all (implies "
                          "--with-baseline; --attr is ignored if also given). Use this to (re)collect "
                          "just the baseline (e.g. after a prompt-design change) without re-running "
                          "every persona sweep too.")
    ap.add_argument("--live", action="store_true", help="actually call the API (costs money)")
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        sys.exit(f"ERROR: dataset not found at {args.dataset}")

    if args.baseline_only:
        personas = [BASELINE_ROW]
        n_personas = 0
        baseline_note = " (baseline only)"
    else:
        personas = build_attribute_sweep(args.attr) if args.attr else build()
        n_personas = len(personas)
        if args.with_baseline:
            personas = [BASELINE_ROW] + personas
        baseline_note = " + 1 baseline" if args.with_baseline else ""
    print(f"Built {n_personas:,} MR personas in memory{baseline_note}.")

    entries = load_entries(args.dataset, args.n_entries, args.entry_seed)
    print(f"Loaded {len(entries)} entries from {os.path.relpath(args.dataset, SCRIPT_DIR)}"
          + (f" (subsampled from full set, seed={args.entry_seed})" if args.n_entries else " (all)"))

    os.makedirs(args.out_dir, exist_ok=True)
    total_calls = len(personas) * len(entries)
    print(f"Model: {args.model}" + (f" (base_url={args.base_url})" if args.base_url else ""))
    print(f"Total planned LLM calls: {len(personas):,} x {len(entries)} = {total_calls:,}")

    results_path = os.path.join(args.out_dir, args.results_name)

    if not args.live:
        has_baseline = args.with_baseline or args.baseline_only
        sample_idx = 1 if has_baseline else 0
        print("\nDRY RUN -- no API calls made. Re-run with --live to execute.")
        non_baseline = personas[sample_idx:]
        print(f"\nSystem prompt for each of the {len(non_baseline)} non-baseline persona(s) this run would use "
              "(this is exactly what gets written to the persona_prompt column; the user prompt -- "
              "task_prompt.md with {code}/{line} filled in per-entry -- is never logged, only sent):")
        for p in non_baseline:
            print(f"\n  --- RowID={p['RowID']} MR={p['MR']} Attribute={p['Attribute']} Value={p['Value']} ---")
            print("  System: " + p["SystemPrompt"])
        if len(personas) > sample_idx:
            p, e = personas[sample_idx], entries[0]
            user_prompt = build_prompt(p["UserPrompt"], e["code"], e["line_number"])
            print(f"\nSample fully-substituted prompt (entry 0, truncated) for persona RowID={p['RowID']}:")
            print(f"  System: {p['SystemPrompt']}")
            print(f"  User: {user_prompt[:500]}")
            print(f"  Ground truth issue label: {e['issue']}")
        if has_baseline:
            b_user = build_prompt(personas[0]["UserPrompt"], entries[0]["code"], entries[0]["line_number"])
            b_system_display = personas[0]["SystemPrompt"] or "(none -- sent as a single HumanMessage)"
            print(f"\nBaseline prompt (truncated):\n  System: {b_system_display}\n  User: {b_user[:500]}")
        print(f"\nWould write results to: {results_path}")
        return

    api_key = load_api_key(args.api_key_env)
    if not api_key:
        sys.exit(f"ERROR: {args.api_key_env} not found in environment or .env file.")
    chat_model = build_chat_model(args.model, api_key, args.max_output_tokens, args.base_url, args.reasoning)

    run_sync(chat_model, personas, entries, results_path, args.workers)
    write_run_info(args.out_dir, args.model,
                   {"dataset": os.path.relpath(args.dataset, SCRIPT_DIR),
                    "n_entries": len(entries), "entry_seed": args.entry_seed,
                    "attr": args.attr, "with_baseline": args.with_baseline,
                    "baseline_only": args.baseline_only,
                    "max_output_tokens": args.max_output_tokens, "reasoning": args.reasoning},
                   filename=args.run_info_name)
    print(f"\nDone. Results: {results_path}")


if __name__ == "__main__":
    main()
