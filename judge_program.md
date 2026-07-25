# judge_program.md — personalized AI-judge autoresearch loop

Goal: build an AI judge that predicts an **individual rater's** 1–6 ratings
(sporty, luxurious, modern, rugged, preference) for cars, and iteratively improve
it until the held-out **overall MAE** is as low as possible. This mirrors
Karpathy's `autoresearch`, but the editable artifact is `judge_config.py` (a
`JudgeConfig`) instead of `train.py`, and the metric is MAE instead of val_bpb.

## Setup

1. **Data**: the merged long csv
   `car_ratings_long_July8_July22_merged_with_color_and_Q23.csv` (rater × car
   ratings + demographics + Q23 + owned brands/body-styles + Big Five).
2. **Images**: `selected_2000_isometric_upload_chunks_renamed/chunk_XX/car_N.png`.
3. **Key**: put `OPENAI_API_KEY` and `OPENAI_MODEL` in a git-ignored `.env`
   (see `.env.example`). Offline runs need no key: use `--mock` or `--dry-run`.
4. **Branch**: work on a branch, e.g. `git checkout -b autojudge/<tag>`.
5. **Read the in-scope files**: `car_judge/config.py` (the `JudgeConfig` levers),
   `car_judge/personalized_judge.py`, `car_judge/inner_loop.py` (the metric —
   treat as ground truth, do not weaken it), and `judge_config.py` (what you edit).

## The metric (inner loop) — DO NOT weaken

`run_inner` reserves a fixed held-out set of `--test-size` cars per rater (stable
seed, identical items across every config), predicts each held-out car using only
that rater's OTHER cars as context plus whatever `JudgeConfig` enables, and reports
**overall MAE** = per rater, mean |pred−true| over their held-out (car × dimension)
cells, then averaged across raters (so every rater counts equally). Lower is better.
Also watch `within1`, per-dimension MAE, and `unparsed` (parse failures inflate MAE).

## What you CAN change

Only `judge_config.py` (the `CONFIG = JudgeConfig(...)`). Every lever is fair game:
`n_context`, `context_strategy` (random / first / diverse / rag_profile /
rag_image), `context_order`, `include_demographics` (+ `profile_fields`),
`include_q23`, `include_owned`, `prompt_variant` (default / concise / persona),
`extra_instructions`, `temperature`.

## What you CANNOT change

- `car_judge/inner_loop.py` metric definition (the ground-truth score).
- The held-out split logic / `--test-size` / `--split-seed` mid-comparison (that
  would make runs non-comparable).
- Don't add heavy dependencies without noting it.

## The experiment loop

The first run is always the **baseline**: `judge_config.py` as shipped
(`n_context=0`, no side-info). Then LOOP:

1. Note the current git commit and the best MAE so far (top of `results.tsv`).
2. Edit `judge_config.py` with ONE idea (rename `name` to describe it).
3. `git add -A && git commit -m "<idea>"`.
4. Run one round (start cheap: few raters, small test-size, then scale up):
   ```
   python run_autojudge.py --data <long.csv> --image-root <chunks> \
     --raters 20 --test-size 20 --split-seed 0 > run.log 2>&1
   ```
5. Read the score: `grep "^\[inner\] MAE" run.log` (or the last `results.tsv` row).
6. If `run.log` has no MAE line, the run errored — `tail -n 40 run.log`, fix if
   trivial, else set status `crash` in `results.tsv` and move on.
7. **Keep or revert**: if overall MAE improved vs the best so far, keep the commit
   (set that `results.tsv` row's `status` to `keep`); if equal/worse, revert with
   `git checkout judge_config.py` (status `discard`).
8. Repeat. Vary ONE lever at a time so you can attribute changes; occasionally
   combine the best-known levers.

**Cost discipline**: live runs cost API money and scale as
`raters × test_size × (1 + n_context images)`. Validate every new idea with
`--mock` first (free), then a small live run (e.g. `--raters 5 --test-size 6`),
then scale to your evaluation size. Keep the same `--raters/--test-size/--split-seed`
within a comparison set.

## First batch of experiments — a concrete ladder

Run these in order. Each phase keeps its winner and carries it into the next, so
`N*` = the best context size found in Phase 1 and `S*` = the best context strategy
found in Phase 2. One lever changes per experiment, so every result is attributable.

### Automated vs. hand-driven

Two ways to run this ladder:

- **Automated** — `python sweep_autojudge.py … --target-mae 1.0 --keep-going`
  runs every phase below by itself, keeps the best, and stops when overall MAE
  drops under the target (or hits `--max-experiments`). Fastest to kick off.
- **Hand-driven** — edit `judge_config.py` one lever at a time and rerun
  `run_autojudge.py`, using `git` to keep/revert. More control, good for probing
  a specific idea. Both share the same metric and `results.tsv`.

### Fixed evaluation protocol (do NOT change mid-batch)

`--test-size 20` (so `min_cars` = 21, auto-derived → ~731 raters pass) and
`--split-seed 0`, held constant across the whole batch — changing the held-out
set mid-comparison makes runs non-comparable. Use a fixed `--raters` for the
ladder (start at **20**; scale only the final winner). Validate every config with
`--mock` first (free, deterministic), then run live. The per-run command is:

```
python run_autojudge.py --data <long.csv> --image-root <chunks> \
  --raters 20 --test-size 20 --split-seed 0 > run.log 2>&1
```

### RAG with CLIP embeddings (one-time setup for `rag_clip`)

The `rag_clip` strategy picks context cars nearest the target in CLIP space, using
the precomputed `data/clip_embeddings.npy` (18,053 × 768 over the full candidate
pool). It needs a per-car index built once by joining those embeddings to the 2000
uploaded cars. That join requires **`clip_dataset_map.csv`** (the row→image map
aligned to the embeddings — the file the embeddings were saved with):

```
python scripts/build_clip_car_index.py \
  --embeddings data/clip_embeddings.npy \
  --map <path>/clip_dataset_map.csv \
  --car-mapping ../selected_2000_isometric_upload_chunks_renamed/car_name_mapping.csv \
  --out data/car_clip_embeddings.npz
```

It prints match coverage (all 1,302 rated cars should map). Until this file exists,
`rag_clip` transparently falls back to `rag_image` (a lightweight pixel feature),
so the ladder still runs — just rerun Phase 2 once the CLIP index is built.

**Context-size cap (important):** with `test_size=20` the context pool is each
rater's *remaining* cars — 14 for July-22 raters (34−20), and as few as 1 for the
shallow July-8 raters. `select_context` caps `n_context` at the pool size per
rater, so effective context tops out near 12–14, and July-8 raters contribute
small-context predictions. Don't set `n_context` above ~12 here; if you want
larger context, raise the rater floor or run July-22 only.

### Phase 0 — Baseline (already shipped)

`JudgeConfig(name="e0_baseline", n_context=0)` — record its MAE. Every later run
must beat this number.

### Phase 1 — Does the person's own context help? (size sweep, random)

- `JudgeConfig(name="e1_ctx4",  n_context=4,  context_strategy="random")`
- `JudgeConfig(name="e2_ctx8",  n_context=8,  context_strategy="random")`
- `JudgeConfig(name="e3_ctx12", n_context=12, context_strategy="random")`

Keep the size with the lowest MAE → call it **N\***. (If MAE is still dropping at
12, you're likely pool-capped — note it and stop at 12.)

### Phase 2 — Smarter context selection at N\*

- `JudgeConfig(name="e4_diverse",  n_context=N*, context_strategy="diverse")`
- `JudgeConfig(name="e5_ragclip",  n_context=N*, context_strategy="rag_clip",   context_order="similar_last")`
- `JudgeConfig(name="e6_ragprof",  n_context=N*, context_strategy="rag_profile", context_order="similar_last")`

`rag_clip` = nearest cars by CLIP embedding (build the index first, above; it
falls back to a pixel feature if the index is missing). Compare all three against
the Phase-1 winner at the same N\*. Keep the best strategy → **S\***.

### Phase 3 — Ordering control (only if S\* is a `rag_*`)

- `JudgeConfig(name="e7_shuffle", n_context=N*, context_strategy=S*, context_order="shuffle")`

If `similar_last` (from Phase 2) beats `shuffle`, ordering genuinely helps — keep
`similar_last`. If not, drop it (simpler is better).

### Phase 4 — Person side-info, one at a time, on top of {N\*, S\*}

- `e8_q23`   → add `include_q23=True`            (cheapest tokens, usually highest signal)
- `e9_demo`  → add `include_demographics=True`   (age / sex / kids / driving / Big-Five)
- `e10_owned`→ add `include_owned=True`          (brands / body styles / hobbies)
- `e11_all`  → add all three together

Keep whichever subset lowers MAE — more text is not always better. (Q23 is the
rater's own statement of what they look at, so it often personalizes cheaply.)

### Phase 5 — Prompt wording, on the best config so far

- `e12_persona`  → `prompt_variant="persona"`
- `e13_instr`    → `extra_instructions="Match this specific person's numbers, including their leniency and quirks — do not substitute your own opinion. When unsure, imitate the pattern in their example ratings."`
- `e14_concise`  → `prompt_variant="concise"` (token-saving control; keep only if MAE is within noise AND it's meaningfully cheaper)

### Phase 6 — Lock and scale

Assemble the best lever combination, then re-run at `--raters 100` (or more) and
also `--split-seed 1` to confirm the winner holds beyond the fixed 20-rater ladder
set and isn't seed-luck. That is the number to report.

**Per-experiment cost** ≈ `raters × test_size × (1 + n_context)` image inputs.
At `--raters 20 --test-size 20`, a config is 400 target predictions; an N=12
context config sends ~13 images each. Mock-verify, then spend.

## results.tsv

Tab-separated, one row per run (appended automatically). Columns:
`timestamp  commit  name  config  score_mae  within1  n_raters  n_preds  unparsed
status  note`. You set `status` to `keep` / `discard` / `crash` after comparing.
