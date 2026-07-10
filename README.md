# AI Judge Experiments — Personalized VLM Car-Aesthetics Judges

Early experiment harness for the **AI Judges++** paper. It queries an open
**Qwen vision-language model** (served elsewhere via vLLM's OpenAI-compatible
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

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --limit-mm-per-prompt image=21          # max_shots + 1; ICL sends many images
```

Then point the client at it (OpenAI-compatible API):

```bash
export VLM_BASE_URL="http://<gpu-host>:8000/v1"
export VLM_API_KEY="EMPTY"                # vLLM ignores the key by default
export VLM_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
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

## Design notes (from the experiment plan)

- **Fixed held-out test set per rater.** For each rater a `--test-size` set of
  cars is reserved with a fixed seed. Context of any size is drawn only from the
  *remaining* cars, so metrics stay comparable across shot counts.
- **Image-before-text in each demonstration** (per Qin et al.): each exemplar is
  an image part followed by its ratings text.
- **Temperature 0** by default for reproducibility.
- **`--enable-thinking`** toggles Qwen's reasoning mode (for the E2 reasoning
  lever); off by default.

## Layout

```
car_judge/
  config.py          run configuration dataclass (+ env defaults)
  data.py            Prolific export loader + car-image index + splits
  client.py          Qwen VLM client (openai SDK -> vLLM endpoint)
  prompts.py         rubric system prompt + base64 message builders
  parsing.py         robust numeric-rating JSON parser
  judges.py          NoContextJudge, InContextJudge
  metrics.py         MAE / exact / within-1 (pure python)
  run_experiment.py  CLI driver
tests/               offline unit tests (no server needed)
```
