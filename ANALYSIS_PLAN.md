# Analysis Plan — Car Aesthetics Preference Study

**Dataset (July 15 export):** 140 respondents × 34 cars = 4,760 car-ratings. Each car rated 1–6 on 4 attributes (Sporty, Luxurious, Modern, Rugged) + Preference. All finished, 100% progress, both attention checks passed by all 140. Design: 4 fixed "anchor" cars seen by everyone + ~30 randomized; 1,300 of 2,000-car pool appear.

---

## 0. Data prep & known gaps (do first)

- **Build one tidy long table:** row per `subject × car × dimension` with rating, subject columns (demographics, hobbies, personality), and car columns (labels, once derived). Backbone for everything below.
- **Normalize:** add within-subject z-scored ratings alongside raw. Z-scored = primary for variance/hypothesis analyses; raw = descriptives and flatliner checks.
- **Blockers to resolve before analysis:**
  - *Prolific demographics don't match this batch* — the on-disk export (112 rows) has **zero** ID overlap with the 140 survey IDs. Re-pull the demographic export for this submission (Age, Sex, Ethnicity, Nationality, Country, Employment, Student status live only there). Join key = Q19 (`ExternalReference` is redacted).
  - *No car metadata* — derive `body_style`, `seats`, `doors` from `actual_car_name` (make/model/year) in `car_name_mapping.csv`; verify the truck/SUV boundary. Required for the field-assumption tests.
  - *Sparse per-car coverage* — outside the 4 anchors, cars average ~3.2 ratings. Only anchors are population-powered; decide whether per-car population analyses use anchors only or accept noisy means.

## 1. QC gate & exclusions

- Attention/AI checks (Q22, Q33): report pass counts; currently 0 failures (confirm this export isn't already pre-filtered, so the denominator is real).
- Flatliners: within-subject rating variance across the 34 cars; flag/exclude subjects with near-zero spread (e.g. all 3s) on preference and attributes.
- Freeze exclusion list, then run everything downstream on the clean set.

## 2. Survey-quality / descriptive analyses (collaborator #1–5)

- **Variance across cars (#2):** distribution + IQR/variance of per-car mean/median ratings — tests whether the car set is varied enough. Run for preference and each attribute.
- **Variance within cars (#3):** IQR/distribution of ratings within each car across subjects — tests individualization.
- **Variance within subject (#4):** spread across each subject's 34 cars — doubles as the flatliner check in §1.
- **Demographics & ordering (#5):** distribution plots for all questionnaire items; if option order was fixed, ordinal regression of selection ~ option position (check first whether Qualtrics randomized order — if so, this is moot).
- Use within-subject z-scored ratings for #2–#4.

## 3. Polarizing vs. agreed-upon cars (core framing)

- **Per-car consensus metric:** within-car preference dispersion (SD/IQR) and inter-rater agreement (ICC / weighted kappa on overlapping raters).
- **Agreed-upon:** low dispersion + extreme mean → universally liked or disliked.
- **Polarizing:** high dispersion despite mid mean → divides raters.
- Rank cars on this axis; inspect exemplars at each extreme; repeat for attributes vs preference.
- **Why it matters:** polarizing cars are where personalization has headroom; agreed-upon cars a population model already nails. This defines the subset where the in-context judge should beat the no-context judge.

## 4. Field-assumption hypotheses (collaborator assumptions #1–4)

- **Family size → larger cars (#1):** preference ~ (#kids × seats/doors).
- **Rural → trucks (#2):** preference for trucks, rural vs non-rural drivers.
- **Attribute correlations (#3):** pairwise Spearman across the 4 attributes on car-mean ratings (non-orthogonality, e.g. sporty↔rugged).
- **Outdoor hobby → rugged (#4):** preference ~ outdoorsy-hobby × rugged-rating interaction.
- **Method:** prefer linear mixed models with random intercepts for subject **and** car (data is crossed) over collapsed Mann-Whitney/OLS, which pseudo-replicate. Keep simple tests as an easy-to-read first pass. Control FDR across the hypothesis family.

## 5. AI-judge evaluation (existing metrics pipeline)

- **Judges:** no-context (0-shot, image only) vs in-context (few-shot on the target rater's exemplars) — E1/E2.
- **Accuracy metrics** (per dimension + overall): MAE, exact, within-1, bias, Pearson/Spearman; weighted kappa & ICC(2,1) vs human; MAE vs mean/median baseline; Bland–Altman, TOST equivalence, paired Wilcoxon.
- **Diagnostics already surfaced (no-context):** systematic over-prediction (+0.30 overall, +0.90 on preference), range compression to the middle, preference r≈0.25 (weakest), rugged r≈0.51 (strongest); barely beats constant baseline on `modern`/`preference`.
- **Personalization test:** does in-context beat no-context, and specifically on the polarizing cars from §3? Report per-rater lift and context-size sweep (N = 0/5/10/15).
- **Reference options:** per-rater ratings vs `car_mean` consensus; fixed held-out test set per rater.

## 6. Cross-cutting statistical notes

- Ratings are ordinal 1–6; z-scoring treats as interval (pragmatic) — note in write-ups.
- Crossed random effects (subject × car) throughout §4–§5.
- FDR correction across attribute and hypothesis families.
- Report raw + z-scored where scale-use differences could bias conclusions.
