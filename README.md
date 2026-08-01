# FAIRCR

Testing fairness of LLM code-review agents via persona prompting on issue-localization
tasks. An LLM is asked to decide whether a code snippet/line contains an issue, once
with no persona (baseline) and once per demographic/role **persona** system prompt
(e.g. gender, seniority, review style). Comparing persona runs against the paired
baseline reveals whether the model's issue-detection behavior shifts based on who it
"thinks" it is.

This repo is a replication package: it contains the full pipeline (persona
generation → LLM calls → metrics → significance tests → plots) needed to reproduce
the experiments and figures.

## Two experiment tracks

The repo contains two generations of the same idea, both still runnable:

| | **Line-level / MR sweep (current)** | **Snippet-level / combo sweep (legacy)** |
|---|---|---|
| Runner | `run_line_experiment.py` | `run_experiment.py` |
| Persona source | `user_prompt.py` | `persona_lib.py` + `generate_personas.py` |
| Design | 10 single-attribute Metamorphic Relations (MR1 Gender … MR10 Domain), one representative value each, plus optional full per-value sweep of one attribute | Exhaustive/sampled combos of 1–15 attributes at once (k=1..15), up to 2,000 personas |
| Task | Per-line binary issue label + one-sentence reason | Per-snippet binary issue label |
| Analysis | `analyze_line.py`, `stat_test_line.py` | `analyze.py`, `stat_test.py` |
| Results live in | `results*/`, `results_majority/` | (not currently populated) |

Both share the same underlying idea (a no-persona baseline paired against persona
variants, compared via **pass_rate** and McNemar/sign tests) and the same LLM-calling
scaffolding (LangChain `init_chat_model`, resumable/threaded sync mode, `.env`-based
API keys). The line-level track is the one with populated `results*/` folders and is
the active one; the combo-sweep track is kept for the earlier, coarser-grained study
(see `claude_code_prompt.md` for its original design rationale).

## Directory structure

```
FAIRCR_LANGCHAIN/
├── README.md
├── requirements.txt              # langchain, langchain-openai, langchain-google-genai, openai, python-dotenv, rdflib
├── .env                          # OPENAI_API_KEY, OPENROUTER_API_KEY (not committed — see below)
├── issue_location -> /Users/lapatrada/Desktop/FAIRCR/issue_location   # symlink, see "External data" below
│
├── task_prompt.md                # {code}/{line} task instructions shared by both tracks
├── persona_prompt.md             # notes on the multi-attribute identity-sentence template
│
├── persona_lib.py                # combo-sweep: ATTRS (15), VALUE_POOL (from ontology), combo_prompt()
├── generate_personas.py          # combo-sweep: build() — exhaustive/sampled k=1..15 persona rows
├── run_experiment.py             # combo-sweep: k=0 baseline + all personas -> results/results.csv
├── collect_batch.py              # combo-sweep: turn a finished OpenAI Batch API job into results.csv
├── analyze.py                    # combo-sweep: accuracy/precision/recall/f1 + pass_rate vs baseline
├── stat_test.py                  # combo-sweep: McNemar + sign test per persona, BH-FDR corrected
│
├── user_prompt.py                # MR sweep: ATTRS (10), VALUE_POOL (from ontology), build()/build_attribute_sweep()
├── run_line_experiment.py        # MR sweep: 10 MR personas (+ optional baseline) -> results.csv
├── analyze_line.py               # MR sweep: accuracy/precision/recall/f1 + pass_rate vs baseline
├── stat_test_line.py             # MR sweep: McNemar + sign test per persona, BH-FDR corrected
├── majority_vote.py              # MR sweep: collapse round1/round2/round3 into a 2-of-3 majority vote
│
├── plot_accuracy.py              # MR sweep: accuracy by attribute, all models overlaid
├── plot_pass_rate.py             # MR sweep: pass_rate by attribute (single round), all models overlaid
├── plot_pass_rate_majority.py    # same, reading the majority-voted results_majority/ tree
├── plot_pass_rate_majority_split.py  # same, one standalone PDF per attribute instead of a 1x3 grid
│
├── results/                      # MR sweep, round 1
├── results_round2/                # MR sweep, round 2 (same design, re-run for majority voting)
├── results_round3/                # MR sweep, round 3
├── results_majority/              # 2-of-3 majority vote across the three rounds above (majority_vote.py output)
│   └── <model>/                  # e.g. qwen3-coder-next, deepseek-v4-flash
│       ├── baseline/
│       │   ├── results_baseline.csv
│       │   ├── run_info_baseline.json
│       │   └── summary_baseline_overview.csv
│       └── k1_<attribute>/       # e.g. k1_gender, k1_reviewstyle, k1_seniority
│           ├── results_k1_<attribute>.csv
│           ├── run_info_k1_<attribute>.json
│           ├── summary_k1_<attribute>_by_persona.csv
│           └── summary_k1_<attribute>_by_overview.csv
│
├── stat_tests_qwen3codernext_seed1/   # stat_test_line.py output, one CSV per attribute
├── pass_rate_majority_*.pdf       # figures rendered by the plot_* scripts (checked in for convenience)
│
├── claude_code_prompt.md         # design notes from the original (combo-sweep) rebuild
├── nohup.out, round3_baseline_qwen.log   # raw logs from past batch runs (not needed to reproduce)
└── __pycache__/                  # build artifact, safe to delete
```

Files prefixed with `_` (`_persona_lib_test.py`, `_plt.py`, `_plt_check.py`,
`_uprompt_check.py`) are scratch/backup copies used while iterating on prompt wording.
They are not imported by any runner script and can be ignored (or deleted) when
reading the pipeline.

## External data dependencies (required, not included in this repo)

Two things live **outside** this directory and must be present at the exact paths
below for the scripts to run — a portability gap worth fixing before handing this
package to someone else, but documented here so it's reproducible as-is today:

1. **Persona ontology** (defines each attribute's value pool), loaded by `rdflib` at
   import time:
   - `persona_lib.py` → `/Users/lapatrada/Desktop/fairness in code review/ontology/Persona_Ontology_V3.ttl`
   - `user_prompt.py` → `/Users/lapatrada/Desktop/fairness in code review/ontology/Persona_Ontology_V5.ttl`
2. **Issue-localization dataset**, reached via the `issue_location` symlink:
   - `issue_location/dataset-issue-location.csv` — snippet-level dataset (combo-sweep track)
   - `issue_location/dataset_location_issue.json` — line-level dataset, `{row_id, code, line_number, issue}` (MR-sweep track)

To replicate on another machine: copy the ontology `.ttl` file(s) and the dataset
files somewhere, then update `ONTOLOGY_PATH` in `persona_lib.py`/`user_prompt.py` and
`DEFAULT_DATASET` in `run_experiment.py`/`run_line_experiment.py` (or replace
`issue_location` with a real copy/symlink) to point at the new locations.

## Running the pipeline

### Setup

```bash
pip install -r requirements.txt
# .env in this directory, containing:
#   OPENAI_API_KEY=...
#   OPENROUTER_API_KEY=...   # only needed for --base-url https://openrouter.ai/api/v1
```

### Line-level / MR sweep (current track)

```bash
# 1. One-time baseline for a model (no persona, dry-run by default -- add --live to spend money)
python3 run_line_experiment.py --live --with-baseline \
  --model openai:qwen/qwen3-coder-next --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --out-dir results/qwen3-coder-next/baseline \
  --results-name results_baseline.csv --run-info-name run_info_baseline.json

# 2. Per-attribute persona sweep, same model (baseline is opt-in, never re-billed here)
python3 run_line_experiment.py --live --workers 8 \
  --model openai:qwen/qwen3-coder-next --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --out-dir results/qwen3-coder-next/k1_gender \
  --results-name results_k1_gender.csv --run-info-name run_info_k1_gender.json

# 3. Metrics: accuracy/precision/recall/f1 + pass_rate vs baseline
python3 analyze_line.py --results results/qwen3-coder-next/k1_gender/results_k1_gender.csv \
  --baseline results/qwen3-coder-next/baseline/results_baseline.csv --prefix k1_gender

# 4. Significance tests (McNemar + sign test, BH-FDR corrected)
python3 stat_test_line.py --results results/qwen3-coder-next/k1_gender/results_k1_gender.csv \
  --baseline results/qwen3-coder-next/baseline/results_baseline.csv --prefix k1_gender

# 5. Optional: after collecting 3 rounds (results/, results_round2/, results_round3/),
#    collapse into a 2-of-3 majority vote, then plot
python3 majority_vote.py --round-files \
  results/qwen3-coder-next/k1_gender/results_k1_gender.csv \
  results_round2/qwen3-coder-next/k1_gender/results_k1_gender.csv \
  results_round3/qwen3-coder-next/k1_gender/results_k1_gender.csv \
  --out results_majority/qwen3-coder-next/k1_gender/results_k1_gender.csv

python3 plot_pass_rate_majority.py   # or plot_accuracy.py / plot_pass_rate.py / plot_pass_rate_majority_split.py
```

### Snippet-level / combo sweep (legacy track)

```bash
# Dry run (no API calls) -- shows a sample persona x snippet pairing
python3 run_experiment.py

# Sync, resumable, concurrent
python3 run_experiment.py --live --workers 8 --model openai:gpt-5.4-mini

# Or the OpenAI Batch API (~50% cheaper, for the full 2,000 x 50 run)
python3 run_experiment.py --batch --submit
python3 collect_batch.py <batch_id>

python3 analyze.py results/results.csv
python3 stat_test.py results/results.csv
```

## Key design guarantees to preserve

- **Paired baseline**: every results CSV's k=0 (or MR-sweep's `baseline`) rows come
  from the *same* model, snippets/entries, and session as the personas it's compared
  against. `analyze.py`/`analyze_line.py` refuse to compute `pass_rate` otherwise
  (see `validate_paired_baseline`) — don't hand-edit results files across runs.
- **Personas are built in memory** from `persona_lib.py`/`user_prompt.py` every run,
  never read from a cached CSV, so they can't go stale relative to the ontology.
- **Dry-run by default**: every runner script requires `--live` (or `--batch
  --submit`) to actually spend money.
