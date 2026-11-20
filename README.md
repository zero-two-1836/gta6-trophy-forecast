# GTA VI trophy rarity forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zero-two-1836/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zero-two-1836/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)


Entry for make.fun competition `gta6-trophy-rarity-forecast-01`:
forecast, for each of GTA VI's 55 PS5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's
target, from the house's reads of the trophy page up to the case's
checkpoint. The final case targets 2026-11-26T00:00:00Z, a week
after launch.

## Files

- `main.py` — the entry (Python 3.13, standard library only).
  Run: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`
- `validate.py` — local validation only, not part of the entry:
  practice scoring against reconstructed answers, synthetic
  final-like scenarios, format and determinism checks.
  Run: `python3 validate.py`
- `LICENSE` — MIT, from the kit's starter.

## Model (see the `main.py` docstring for the full description)

1. **Distinct snapshots.** Consecutive identical reads collapse
   to one: Exophase refreshes a game's counts every few hours,
   so a flat history is one stale snapshot, not stability.
2. **Logit trend on the recent window.** Per trophy, recency-
   weighted least squares of logit(rate) on ln(1 + hours since
   release), fitted on the last 24 h of distinct snapshots
   (full history if that window is too thin). Slopes shrink
   toward the pooled median slope; the forecast anchors on the
   last observed rate.
3. **Dilution when the history is flat.** A refresh adds many new
   players who have earned almost nothing, so rates fall: the
   median mixes incumbents with newcomers at a fraction rho of
   the incumbents' rate, with the tracked population grown by a
   launch prior (0.39/h at release, decaying with age).
4. **Launch prior for the tracked population.** When the history
   is flat, N is grown from the first snapshot at the decaying
   launch rate (the first refresh revealed 638 -> 3,868 players);
   in informative histories the fitted slope carries the growth
   and the prior only corrects a stale last snapshot.
5. **Tier caps on the median:** platinum <= 1.0%, gold <= 6%,
   silver <= 30%, bronze <= 90%. A trophy at ~100% stays near
   100% (everyone earns the opening trophy within minutes).
6. **Spread** widens honestly with the horizon: random-walk term
   plus slope standard error, fit residual, refresh uncertainty
   and a base term. A 90-minute forecast gets sigma ~0.35; the
   4.5-day final gets sigma above 1.

## Validation

Practice answers are not shipped with the kit, so they are
reconstructed from the totals competition.yaml states for the
first refresh (players 638 -> 3,868, trophies earned 943 ->
4,276, every rate but the 100% one fell), under three variants
(pro-rata new earners, +1.5 per trophy, none). The flat
persistence baseline reproduces the house's reported practice
loss of 1.51 exactly on average across the three, which sanity-
checks the reconstruction.

### Results (`python3 validate.py`)

| Scenario | Flat baseline | Trend baseline | This entry |
|---|---|---|---|
| Practice p00/p01, mean over 3 reconstructions | 1.510 | 1.510 | **0.184** |
| Synthetic final, base (seed 7) | 1.480 | 0.928 | **0.847** |
| Synthetic final, fast N growth | 1.232 | 0.797 | **0.739** |
| Synthetic final, slow N / fast progression | 1.845 | 1.064 | **0.970** |
| Synthetic final, slow everything | 1.540 | 0.970 | **0.872** |

Practice typical miss 0.04–0.06 pp (flat: 0.68 pp), 90%
coverage 0.98 on the reconstructions. On 15 further synthetic
final scenarios (seeds 1–5 x 3 regimes) this entry beats the
trend baseline in 14 and the flat baseline in all 15.

The synthetic finals are a generative cohort model (2 days of
30-minute reads with 4-hour refreshes, cut at 59.5 h, target
day 7); they are a robustness check, not a claim about the real
final, whose answer is not known to anyone before the target.

## Assumptions and limitations

- The practice win rests on the launch prior: a flat history
  early after release is a stale pre-refresh snapshot, and the
  first refresh dilutes every rate ~5x. If Exophase refreshed
  continuously instead, the dilution would be wrong — but the
  kit documents the step refresh, and the flat baseline's
  practice loss (1.51, reproduced exactly) confirms it.
- The final's history will be informative, so the recent-window
  trend carries it; the trend assumes growth stays roughly
  linear in logit vs ln(1 + hours), which saturates gracefully
  but cannot foresee a late-story difficulty spike or a viral
  grind spot.
- Tier caps are priors, not data: if a gold trophy turns out
  to be an early-story award, the 6% cap could bind wrongly.
  The 100% trophy's 99% median assumes nearly every tracked
  player plays at least the opening minutes.
- The synthetic scenarios use invented trophy archetypes; the
  real trophy list's difficulty ordering is unknown beyond
  tier, so per-trophy medians lean on each trophy's own short
  history plus shrinkage to the pooled slope.
