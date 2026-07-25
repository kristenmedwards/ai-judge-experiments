"""THE EDITABLE ARTIFACT — this is the file the outer loop mutates.

Analogous to Karpathy autoresearch's ``train.py``: an experimenter (you, or a
coding agent following ``judge_program.md``) changes ``CONFIG`` below, reruns the
inner loop via ``run_autojudge.py``, and keeps the edit only if overall MAE drops.

Everything about the AI judge lives in this one JudgeConfig. Start from the
no-context baseline, then turn levers on one at a time.

Levers you can change:
  name                 label that lands in results.tsv (rename per experiment)
  n_context            # of the rater's own cars shown as few-shot exemplars (0 = baseline)
  context_strategy     random | first | diverse | rag_profile | rag_image
  context_order        as_selected | similar_last | shuffle
  include_demographics show age/sex/kids/driving/Big-Five block
  profile_fields       which demographic fields to include
  include_q23          show the rater's own "what I look at when rating" text
  include_owned        show brands / body styles / hobbies they reported
  prompt_variant       default | concise | persona
  extra_instructions   free-form text appended to the system prompt
  temperature          sampling temperature for THIS judge (endpoint temp is separate)
"""

from car_judge.config import JudgeConfig

# --- Phase 0 baseline: no context, image only. Run this first. ---
CONFIG = JudgeConfig(
    name="e0_baseline",
    n_context=0,
    context_strategy="random",
    include_demographics=False,
    include_q23=False,
    include_owned=False,
    prompt_variant="default",
)

# The ladder from judge_program.md (uncomment ONE at a time; keep if MAE improves).
# Replace N*/S* with the winners you find as you go.
#
# Phase 1 — context size sweep (random):
# CONFIG = JudgeConfig(name="e1_ctx4",  n_context=4,  context_strategy="random")
# CONFIG = JudgeConfig(name="e2_ctx8",  n_context=8,  context_strategy="random")
# CONFIG = JudgeConfig(name="e3_ctx12", n_context=12, context_strategy="random")
#
# Phase 2 — smarter context at N* (rag_clip = nearest by CLIP embedding; build the
# index first with scripts/build_clip_car_index.py, else it falls back to pixels):
# CONFIG = JudgeConfig(name="e5_ragclip", n_context=8, context_strategy="rag_clip",
#                      context_order="similar_last")
#
# Phase 4 — person side-info on top of {N*, S*}:
# CONFIG = JudgeConfig(name="e8_q23", n_context=8, context_strategy="rag_clip",
#                      context_order="similar_last", include_q23=True)
# CONFIG = JudgeConfig(name="e11_all", n_context=8, context_strategy="rag_clip",
#                      context_order="similar_last", include_q23=True,
#                      include_demographics=True, include_owned=True)
#
# Phase 5 — prompt wording on the best config so far:
# CONFIG = JudgeConfig(name="e12_persona", n_context=8, context_strategy="rag_clip",
#                      context_order="similar_last", include_q23=True,
#                      prompt_variant="persona")
