# AI Judge Experiments — Personalized VLM Car-Aesthetics Judges

Early experiment harness for the **AI Judges++** paper. It queries an open
**Qwen3.5 vision-language model** (served elsewhere via vLLM's OpenAI-compatible
endpoint) to predict human aesthetic ratings of cars, in two conditions:

1. **No-context judge** — 0-shot. The model sees only the target car image and
   the rating rubric, and returns five ratings.
2. **In-context judge** — few-shot. The model first sees several example cars
   *that one specific rater scored*, each followed by that rater's ratings,
   then sees a target car and predicts that rater's ratings for it.

This is the machinery for experiments **E1 / E2** in the experiment plan
(personalization + information ablation). The VLM itself is assumed to be hosted
on a separate GPU box — this repo is the **client**.

> Status: early test build. It runs against the real Prolific survey export but
> the model calls are pointed at whatever `--base-url` you give. A `--dry-run`
> mode builds and inspects every request **without** contacting a server.

## Rating task

Each car is rated on five 1–6 integer scales:

| dimension    | prompt meaning                          |
|--------------|-----------------------------------------|
| `sporty`     | how sporty the car looks                |
| `luxurious`  | how luxurious the car looks             |
| `modern`     | how modern the car looks                |
| `rugged`     | how rugged the car looks                |
| `preference` | how much the rater likes the car        |

`1` = "not at all / strongly dislike", `6` = "extremely / strongly like".

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Serving the model (reference, runs on the GPU box, not here)

Qwen3.5 is natively multimodal — vision is built into the base checkpoint, so
there is no separate `-VL` variant to serve.

```bash
# --limit-mm-per-prompt : max_shots + 1; ICL sends many images
# --reasoning-parser    : only needed for --enable-thinking runs, so that the
#                         <think> block is split off instead of prefixing the JSON
vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 \
  --limit-mm-per-prompt image=21 \
  --reasoning-parser qwen3
```

Then point the client at it (OpenAI-compatible API):

```bash
export VLM_BASE_URL="http://<gpu-host>:8000/v1"
export VLM_API_KEY="EMPTY"                # vLLM ignores the key by default
export VLM_MODEL="Qwen/Qwen3.5-9B"
```

## Run

Dry run (no server needed — inspect the requests that *would* be sent):

```bash
python -m car_judge.run_experiment \
  --data "../Car Aesthetics Ratings - Prolific_July 8, 2026_10.14.csv" \
  --image-root "../selected_2000_isometric_upload_chunks_renamed" \
  --raters 2 --test-size 8 --context-size 10 \
  --judges no_context in_context \
  --dry-run
```

Live run against a served Qwen model:

```bash
python -m car_judge.run_experiment \
  --data "../Car Aesthetics Ratings - Prolific_July 8, 2026_10.14.csv" \
  --image-root "../selected_2000_isometric_upload_chunks_renamed" \
  --base-url "$VLM_BASE_URL" --model "$VLM_MODEL" \
  --raters 5 --test-size 8 --context-size 10 \
  --judges no_context in_context \
  --out outputs/predictions.csv
```

Output is a tidy CSV: one row per (rater × target car × dimension × judge) with
the predicted rating, the true held-out rating, and the run configuration.

## Context-size sweep + statistical evaluation

To sweep context sizes (e.g. N = 0, 5, 10, 15) with hold-out testing and then
run the full statistical comparison against human ratings (weighted kappa,
ICC, Spearman, MAE/RMSE, Bland–Altman, TOST equivalence, paired Wilcoxon):

```bash
python -m car_judge.run_context_sweep \
  --data ".../Prolific.csv" --image-root ".../chunks" \
  --raters 5 --context-sizes 0 5 10 15 --holdout-mode fixed \
  --test-size 8 --repeats 3 --out outputs/sweep_predictions.csv

python -m evaluation.evaluate_predictions \
  --predictions outputs/sweep_predictions.csv \
  --reference rater --data ".../Prolific.csv" \
  --out-dir evaluation/outputs/sweep_rater
```

See [evaluation/README.md](evaluation/README.md) for hold-out modes
(fixed / hold-multiple-out / hold-one-out), the `car_mean` consensus
reference, and how to read the outputs.

## Personalized-judge autoresearch loop (GPT, inner + outer loop)

The newer harness learns to match **one individual rater's** ratings and
iteratively improves the judge, in the style of Karpathy's `autoresearch`. It
uses the merged long csv (`car_ratings_long_..._with_color_and_Q23.csv`), which
carries each rater's demographics, Q23 free-text, owned brands/body-styles, and
Big Five — so the judge can be personalized.

- **Baseline** — `JudgeConfig(n_context=0)`: the model sees only the target image
  and the rubric and returns the five ratings.
- **Inner loop** (`car_judge/inner_loop.py`) — score one `JudgeConfig` against
  real raters. Fixed held-out set per rater; predict each held-out car from that
  rater's *other* cars + enabled side-info; report **overall MAE** (macro-averaged
  across raters so each person counts equally) plus per-dimension MAE and within-1.
- **Outer loop** (agent-driven, see `judge_program.md`) — edit `judge_config.py`
  (the one editable artifact, like autoresearch's `train.py`), rerun, and keep the
  change only if MAE dropped. Every run appends a row to `results.tsv`.

### Keys — never pasted, always from env / .env

```bash
cp .env.example .env            # then edit: OPENAI_API_KEY=... and OPENAI_MODEL=gpt-5.6
```

`.env` is git-ignored; the key stays on your machine. `OPENAI_BASE_URL` optionally
retargets Azure / vLLM / a proxy. Offline modes need no key at all.

### Run

```bash
# offline, free, deterministic (validate any config end-to-end):
python run_autojudge.py --data ../car_ratings_long_..._Q23.csv \
  --image-root ../selected_2000_isometric_upload_chunks_renamed \
  --raters 10 --test-size 20 --mock

# dry-run: build every request, size images, no server:
python run_autojudge.py --data <long.csv> --image-root <chunks> --dry-run

# live (after filling .env): one experiment round -> results.tsv
python run_autojudge.py --data <long.csv> --image-root <chunks> \
  --raters 25 --test-size 20
```

Then follow `judge_program.md`: change one lever in `judge_config.py`, rerun,
`git commit` if MAE improved (keep) or `git checkout judge_config.py` (revert).

### RAG context via CLIP (one-time index build)

`context_strategy="rag_clip"` selects context cars nearest the target in CLIP
space, from `data/clip_embeddings.npy` (18,053×768 over the full pool). Build the
per-car index once — it joins the embeddings to the 2,000 cars via
`clip_dataset_map.csv` (the row→image map saved with the embeddings) and
`car_name_mapping.csv`:

```bash
python scripts/build_clip_car_index.py \
  --embeddings data/clip_embeddings.npy \
  --map <path>/clip_dataset_map.csv \
  --car-mapping ../selected_2000_isometric_upload_chunks_renamed/car_name_mapping.csv \
  --out data/car_clip_embeddings.npz
```

All 1,302 rated cars are in the mapping, so coverage should be complete. Until the
index exists, `rag_clip` falls back to a lightweight pixel feature (`rag_image`).

### Automated sweep — iterate until MAE < target

`sweep_autojudge.py` runs the whole ladder by itself (no per-round editing): it
greedily searches one lever at a time, carries the best forward, logs every run to
`results.tsv`, and **stops as soon as the best overall MAE drops below
`--target-mae`** (default 1.0). `--keep-going` adds a randomized combination search
after the ladder; `--max-experiments` is a hard safety cap so it can't loop or
spend forever.

```bash
# validate the sweep mechanics FREE first:
python sweep_autojudge.py --data ../car_ratings_long_..._Q23.csv \
  --image-root ../selected_2000_isometric_upload_chunks_renamed \
  --raters 20 --test-size 20 --target-mae 1.0 --mock

# live sweep until MAE < 1.0 (or the cap), best config -> best_judge_config.json:
python sweep_autojudge.py --data <long.csv> --image-root <chunks> \
  --raters 20 --test-size 20 --target-mae 1.0 --keep-going --max-experiments 30
```

Cost ≈ `raters × test_size` model calls per experiment (each sending `1 + n_context`
images), so a 30-experiment live sweep at 20×20 is ~240k image inputs — mock first.

### JudgeConfig levers (what the outer loop mutates)

`n_context` · `context_strategy` (random / first / diverse / rag_clip /
rag_image / rag_profile) · `context_order` · `include_demographics` (+ `profile_fields`) ·
`include_q23` · `include_owned` · `prompt_variant` (default / concise / persona) ·
`extra_instructions` · `temperature`.

## Design notes (from the experiment plan)

- **Fixed held-out test set per rater.** For each rater a `--test-size` set of
  cars is reserved with a fixed seed. Context of any size is drawn only from the
  *remaining* cars, so metrics stay comparable across shot counts.
- **Image-before-text in each demonstration** (per Qin et al.): each exemplar is
  an image part followed by its ratings text.
- **Temperature 0** by default for reproducibility.
- **The no-context judge rates each car once.** It is 0-shot — the prompt is the
  rubric plus the target image, with no rater identity — so its prediction is a
  function of the image alone and cannot differ between two raters who happen to
  share a car. Repeat targets are served from cache. The predictions CSV still
  carries one row per (rater × car × dimension), scored against *that* rater's
  ratings, so per-rater metrics are unaffected. The in-context judge deliberately
  does **not** cache: its ratings are conditioned on the rater's exemplars, so the
  same car legitimately gets a different prediction per rater.
- **`--enable-thinking`** toggles Qwen's reasoning mode (for the E2 reasoning
  lever); off by default. Note this is a real lever on Qwen3.5 — the old
  Qwen2.5-VL chat template ignored the flag, so any pre-Qwen3.5 "thinking" run
  was in fact a non-thinking run and is not comparable. Reasoning also spends
  the same budget as the answer, so raise `max_tokens` (512 by default) when
  enabling it, or the JSON can get truncated away.

## Layout

```
car_judge/
  config.py          RunConfig + JudgeConfig (the editable-judge dataclass)
  env.py             OpenAI creds from env / .env (key never hard-coded)
  data.py            Prolific export loader + car-image index + splits
  long_data.py       merged long-csv loader (ratings + demographics + Q23 + owned)
  client.py          OpenAI-compatible client (+ dry-run + deterministic mock)
  prompts.py         rubric + config-driven message builder (persona/Q23/owned)
  context_selection.py  context strategies: random/first/diverse/rag_profile/rag_image
  personalized_judge.py JudgeConfig-driven judge (baseline .. full personalization)
  inner_loop.py      score a JudgeConfig -> overall MAE across raters/metrics
  parsing.py         robust numeric-rating JSON parser
  judges.py          original NoContextJudge / InContextJudge (Qwen path)
  metrics.py         MAE / exact / within-1 (pure python)
  run_experiment.py / run_context_sweep.py  original Qwen CLIs
judge_config.py      THE EDITABLE ARTIFACT the outer loop mutates
judge_program.md     instructions for the agent-driven outer loop
run_autojudge.py     outer-loop driver: run one round -> results.tsv
evaluation/          statistical evaluation CLI (kappa, ICC, TOST, ...)
tests/               offline unit tests, incl. hermetic test_autojudge.py
```
