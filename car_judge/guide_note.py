"""Guide-conditioned "taste card" — programs ../rater_matching_guide.md into text.

Given a rater's profile, deterministically compute the guide's per-person model
(attribute weights, baseline, spread, endowment bumps, perception shifts,
confidence) and render it as a compact instruction card for the judge prompt.

This is the difference between dumping raw demographics (tried; within noise)
and telling the model what the demographics MEAN: "older male pickup owner"
becomes "rugged weight ~0.55 vs 0.28 average".

Contract: pure function of the profile dict — no randomness, no access to the
target car or any true ratings (mirrors the no-leakage rule in
context_selection.py). Numbers come from the guide's tables (§2-§6).
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# Population reference points (guide §0 and observed medians in the long CSV;
# BFI is a 1-5 scale in this data).
POP_PREF_MEAN = 3.3
BFI_MED = {"openness": 4.0, "conscientiousness": 4.0, "extraversion": 2.5,
           "agreeableness": 4.0, "neuroticism": 2.5}
MED_AGE = 37.0

# §3 population-average attribute weights (within-person corr with preference).
BASE_WEIGHTS = {"luxurious": 0.64, "modern": 0.61, "sporty": 0.57, "rugged": 0.28}

# §4(a) luxury-brand set for the luxurious-weight bump.
LUX_BRANDS = ("bmw", "mercedes_benz", "audi", "lexus", "cadillac",
              "lincoln", "genesis", "infiniti", "acura", "volvo")

# §4(b) endowment bumps by owned body style (sedan ~0 → omitted).
ENDOWMENT = [("pickup_truck", "pickup truck", 1.0),
             ("convertible", "convertible", 0.6),
             ("sport_utility_vehicle", "SUV", 0.4),
             ("coupe", "coupe", 0.3),
             ("hatchback", "hatchback", 0.2)]


def _num(profile: Dict, key: str) -> Optional[float]:
    v = profile.get(key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN -> None


def _bfi(profile: Dict, trait: str) -> float:
    """Centered BFI score (0 = population median for that trait)."""
    v = _num(profile, f"bfi_{trait}")
    return (v - BFI_MED[trait]) if v is not None else 0.0


def _owns(profile: Dict, key: str) -> bool:
    return str(profile.get(key, "")) in ("1", "1.0", "True", "true")


def _income_tier(profile: Dict) -> int:
    """-1 low (<$30k), 0 mid/unknown, +1 high (>=$100k). Parses the messy label."""
    s = str(profile.get("prolific_household_income_us_participants_only", "") or "")
    nums = [int(n) for n in re.findall(r"\d{4,6}", s)]
    if not nums:
        return 0
    top = max(nums)
    if "less than" in s.lower():
        return -1
    if top >= 100000:
        return 1
    if top < 30000:
        return -1
    return 0


def _driving_level(profile: Dict) -> int:
    """+2 daily, +1 weekly, 0 monthly/unknown, -1 rarely/never."""
    s = str(profile.get("driving_frequency", "") or "").lower()
    if "daily" in s:
        return 2
    if "week" in s:
        return 1
    if "rarely" in s or "never" in s:
        return -1
    return 0


def compute_taste(profile: Dict) -> Dict:
    """The guide's numeric per-person model. Deterministic; see module docstring."""
    age = _num(profile, "prolific_age") or MED_AGE
    sex = str(profile.get("prolific_sex", "") or "").strip().lower()
    o, c = _bfi(profile, "openness"), _bfi(profile, "conscientiousness")
    e, a = _bfi(profile, "extraversion"), _bfi(profile, "agreeableness")
    n = _bfi(profile, "neuroticism")
    inc, drv = _income_tier(profile), _driving_level(profile)

    w = dict(BASE_WEIGHTS)
    # §3 sex row (M .32/.24 rugged, .62/.52 sporty around the base).
    if sex == "male":
        w["rugged"] += 0.04
        w["sporty"] += 0.05
    elif sex == "female":
        w["rugged"] -= 0.04
        w["sporty"] -= 0.05
    # §3 age — the strongest single moderator, rugged only (rho=+0.31).
    w["rugged"] += max(-0.15, min(0.20, 0.010 * (age - MED_AGE)))
    # §3 income.
    if inc > 0:
        w["modern"] += 0.06; w["luxurious"] += 0.05
        w["sporty"] += 0.04; w["rugged"] += 0.02
    elif inc < 0:
        w["modern"] -= 0.03; w["luxurious"] -= 0.03; w["sporty"] -= 0.02
    # §3 driving frequency.
    if drv == 2:
        w["sporty"] += 0.05; w["rugged"] += 0.04
        w["luxurious"] += 0.04; w["modern"] += 0.04
    elif drv == 1:
        for k in w:
            w[k] += 0.02
    elif drv < 0:
        for k in w:
            w[k] -= 0.04
    # §3 personality rows: E raises all, N flattens all, C raises, O slight anti.
    for k in w:
        w[k] += 0.03 * e - 0.04 * n + 0.02 * c
    w["sporty"] += 0.01 * c - 0.015 * o
    w["modern"] -= 0.015 * o
    # §4(a) ownership -> weights.
    if _owns(profile, "body_style_owned_pickup_truck"):
        w["rugged"] += 0.14
    if _owns(profile, "brand_owned_jeep"):
        w["rugged"] += 0.10
    if _owns(profile, "body_style_owned_sport_utility_vehicle"):
        w["rugged"] += 0.10
    if _owns(profile, "body_style_owned_coupe"):
        w["sporty"] += 0.07
    if _owns(profile, "brand_owned_tesla"):
        w["modern"] += 0.10
    if any(_owns(profile, f"brand_owned_{b}") for b in LUX_BRANDS):
        w["luxurious"] += 0.05
    for k in w:
        w[k] = round(max(-0.05, min(0.80, w[k])), 2)

    # §2 baseline and spread.
    baseline = round(max(2.6, min(4.2,
        POP_PREF_MEAN + 0.15 * a + 0.15 * e - 0.06 * n)), 1)
    spread = 0.5 * c + 0.5 * o          # >0 wide, <0 compressed
    # §6 legibility: stable/conscientious/extraverted/older -> model fits well.
    legibility = -0.4 * n + 0.3 * c + 0.2 * e + 0.01 * (age - MED_AGE)

    bumps = [(label, val) for key, label, val in ENDOWMENT
             if _owns(profile, f"body_style_owned_{key}")]
    return {"weights": w, "baseline": baseline, "spread": spread,
            "legibility": legibility, "bumps": bumps,
            "age": age, "sex": sex, "income_tier": inc}


def render_guide_note(profile: Dict) -> str:
    """Render the taste card as compact prompt text."""
    t = compute_taste(profile)
    w, lines = t["weights"], []

    lines.append("TASTE CARD for this person (derived from a study of 776 raters "
                 "— treat as priors that tilt predictions, not rules):")
    lines.append(f"- Baseline liking: about {t['baseline']}/6. Center their "
                 "preference predictions there, not at the scale midpoint.")

    if t["spread"] >= 0.25:
        lines.append("- Scale use: they genuinely use the whole 1-6 range, "
                     "including 1s and 6s. Do NOT compress toward the middle.")
    elif t["spread"] <= -0.75:
        lines.append("- Scale use: they avoid extremes; keep most predictions "
                     "within 2-5.")

    ordered = sorted(w, key=w.get, reverse=True)
    wtxt = ", ".join(f"{k} {w[k]:+.2f}" for k in ordered)
    lines.append(f"- Preference drivers — how strongly each styling attribute "
                 f"moves THEIR liking (population avg: luxurious .64, modern .61, "
                 f"sporty .57, rugged .28): {wtxt}.")
    dr = w["rugged"] - BASE_WEIGHTS["rugged"]
    if dr >= 0.10:
        lines.append("- Rugged is unusually important to them: rugged looks "
                     "should clearly RAISE their preference.")
    elif dr <= -0.10:
        lines.append("- Rugged matters little or even negatively for them: a "
                     "rugged look does NOT raise their preference (refined "
                     "styling does).")

    if t["bumps"]:
        b = "; ".join(f"if the target car is a {lbl}, add about +{v:.1f} to "
                      f"preference" for lbl, v in t["bumps"])
        lines.append(f"- Ownership bump ({', '.join(l for l, _ in t['bumps'])} "
                     f"owner): {b}.")

    percep = []
    if t["age"] >= 50:
        percep.append("they see cars as somewhat MORE sporty and rugged than "
                      "younger raters do — nudge those two attribute ratings up")
    elif t["age"] <= 27:
        percep.append("they see cars as slightly LESS sporty/rugged than older "
                      "raters do")
    if t["income_tier"] > 0:
        percep.append("they rate cars slightly less modern than average")
    if percep:
        lines.append("- Perception: " + "; ".join(percep) + ".")

    if t["legibility"] >= 0.4:
        lines.append("- Reliability: their taste tracks visible styling closely "
                     "— trust this card and the styling of the car.")
    elif t["legibility"] <= -0.5:
        lines.append("- Reliability: their taste is idiosyncratic; styling "
                     "explains less. Stay closer to their baseline and lean "
                     "harder on their example ratings than on this card.")

    lines.append("Use the card together with their example ratings: infer their "
                 "actual leniency and taste from the examples, and use the card "
                 "as the prior. When the examples clearly contradict the card, "
                 "trust the examples.")
    return "\n".join(lines)
