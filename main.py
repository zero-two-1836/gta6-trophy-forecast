"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PS5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its Exophase earn rate -- the percent of
tracked players who have earned it -- at the case's target, from the
house's reads of Exophase's trophy page up to the case's checkpoint.

The core observation is that an earn rate is a ratio, earners /
players-tracked, whose two sides move for different reasons, and
that Exophase only refreshes a game's counts every few hours, so
long runs of identical reads are one stale snapshot, not stability:

1. Distinct snapshots.  Consecutive reads with identical players
   tracked and identical rates collapse to their first appearance.
   Everything below is fitted on distinct snapshots only.  A flat
   history (a single distinct snapshot) is treated as stale data
   from before the last refresh, not as a stable rate.

2. Logit trend on the recent window.  Per trophy, weighted least
   squares of logit(rate) on x = ln(1 + hours since release),
   fitted on the distinct snapshots of the last 24 hours (the
   full history when the recent window holds fewer than 3
   snapshots).  A launch's growth bends in log-time -- the
   recent slope is what the next days inherit, while the
   full-history slope averages away the acceleration -- so the
   recent window wins.  Each trophy's slope is shrunk toward
   the pooled median slope across trophies (a trophy's short,
   steppy series is noisy; trophies share the launch's regime),
   and the forecast is anchored on the last observed rate:
   median = logit(r_last) + b_shrunk * (x_target - x_last).

   (A count model -- extrapolating earners E and players N
   separately, rate = E / N -- was tried and dropped: on
   synthetic launch-week scenarios it scored far worse than
   the trend, because E's growth in log-time is curved and a
   straight line under-forecasts it.  The population N is
   still fitted on its own, for step 4 and for the flat-case
   dilution in step 3.)

3. Dilution when the history is flat.  A refresh that adds many
   new players pulls every rate down, because newcomers have
   earned almost nothing yet.  With N_target from the launch
   prior in step 4, the median rate is the incumbent rate mixed
   with newcomers who have earned the trophy at a fraction rho
   of the incumbents' rate:
       r = r_last * (N_last + rho * (N_target - N_last)) / N_target
       rho = 0.8 * (1 - exp(-H / 60h)),  H the horizon in hours.
   Newcomers over a 90-minute horizon have barely played (rho
   ~ 0.02); over the 4.5-day final horizon they have had days
   (rho ~ 0.67).  A trophy at ~100% is exempt: everyone earns the
   opening trophy within minutes, so it stays near 100%.

4. Launch prior for the tracked population.  On launch night the
   page sat unchanged for hours while the game launched, and the
   first refresh revealed a ~6x larger tracked population
   (638 -> 3,868 players, documented in the kit).  When the
   history is flat, the population at the target is therefore
   taken to have grown since the first snapshot at a rate that
   decays with age since release,
       g(t) = 0.39 * exp(-t / 36h) per hour,
   i.e. N_target = N_last * exp(g_bar * (staleness + H)), with
   g_bar the mean of g over the interval and the growth factor
   capped at 12x.  The same decaying rate, applied only to the
   staleness gap (capped at 6 h, factor at most 1.5x), nudges
   N_target up in informative histories whose last snapshot is
   stale.  By day 2 the fitted slope carries the growth and the
   prior is a small correction.

5. Plausibility caps on the median.  A week after launch only a
   small share of players can have finished everything: platinum
   <= 1.0%, gold <= 6%, silver <= 30%, bronze <= 90%.  These are
   upper bounds on the median only; the upper quantiles may exceed
   them.

6. Spread.  Quantiles are the median in logit space plus
   z_tau * sigma, with
       sigma^2 = (0.5*sqrt(d))^2 + (se_slope * dx)^2
                 + resid^2 + refresh^2 + 0.20^2,
   d the horizon in days, se_slope the shrunk slope's standard
   error times the extrapolation distance dx in x, resid the
   window fit's residual spread, and
   refresh = 0.55 * sqrt(1 - exp(-H / R)) the uncertainty from
   Exophase refreshes not yet seen, R the median gap between
   distinct snapshots (6 h when there is none).  The spread
   therefore widens honestly with the horizon: a 90-minute
   forecast gets sigma ~ 0.35, the 4.5-day final more than 1.

Standard library only, Python 3.13, deterministic, no network.
The grader runs `python3 -P -s /bot/main.py`; it reads every
<id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out
to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Tier caps on the median rate (percent) at the target.
TIER_CAP = {"platinum": 1.0, "gold": 6.0, "silver": 30.0, "bronze": 90.0}
# A trophy at or above this rate is treated as "everyone earns it
# within minutes" and exempted from dilution.
NEARLY_EVERYONE = 99.5
NEARLY_EVERYONE_MEDIAN = 99.0

# Recency half-life (hours) for the weighted fits.
HALF_LIFE_H = 36.0
# Shrinkage strength: slope weight n / (n + SHRINK_K).
SHRINK_K = 3.0
# Launch prior: tracked-population growth rate per hour at release.
G0 = 0.39
G_DECAY_H = 36.0
# Refresh uncertainty scale (logit) and default refresh gap (hours).
REFRESH_SIGMA = 0.55
REFRESH_GAP_DEFAULT_H = 6.0
BASE_SIGMA = 0.20
RW_SIGMA_PER_SQRT_DAY = 0.5


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    c = clamp_pct(pct)
    return math.log(c / (100.0 - c))


def inv_logit(y: float) -> float:
    y = max(-20.0, min(20.0, y))
    return 100.0 / (1.0 + math.exp(-y))


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def read_time(h: dict) -> datetime:
    """A read's time: its clock slot first (the authoritative read
    time), falling back to `at`."""
    return parse_time(h.get("slot") or h["at"])


def hours_since_release(t: datetime) -> float:
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def x_of(t: datetime) -> float:
    """Log-time coordinate: ln(1 + hours since release)."""
    return math.log1p(hours_since_release(t))


def distinct_snapshots(history):
    """Collapse consecutive identical reads to their first appearance."""
    out = []
    prev = None
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or []))
        if key != prev:
            out.append(h)
            prev = key
    return out


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, se_b)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    sw = sum(ws)
    if sw <= 0:
        ws = [1.0] * n
        sw = float(n)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-12:
        return my, 0.0, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        dof = max(1, n - 2)
        var_b = (resid2 / dof) / sxx
        se_b = math.sqrt(max(0.0, var_b))
    else:
        se_b = 0.0
    return a, b, se_b


def recency_weights(times, last_time):
    """Weights decaying with age from the last snapshot (36 h half-life)."""
    out = []
    for t in times:
        age_h = max(0.0, (last_time - t).total_seconds() / 3600.0)
        out.append(0.5 ** (age_h / HALF_LIFE_H))
    return out


def growth_rate_mean(t0: datetime, t1: datetime) -> float:
    """Mean of g(t) = G0 * exp(-t/36h) over [t0, t1] (hours since release)."""
    a = hours_since_release(t0)
    b = hours_since_release(t1)
    if b <= a:
        return G0 * math.exp(-a / G_DECAY_H)
    # Integral of G0*exp(-t/36) dt from a to b, divided by (b - a).
    integ = G0 * G_DECAY_H * (math.exp(-a / G_DECAY_H) - math.exp(-b / G_DECAY_H))
    return integ / (b - a)


def population_forecast(snaps, checkpoint, target):
    """Forecast players tracked at the target.

    Returns (N_target, staleness_hours).  Fits log N on x over the
    distinct snapshots when there are 3+; otherwise the launch
    prior carries the growth since the first snapshot.  A stale
    last snapshot (no refresh for a while) adds a capped catch-up.
    """
    first_t = read_time(snaps[0])
    last_t = read_time(snaps[-1])
    n_last = float(snaps[-1]["players_tracked"])
    staleness_h = max(0.0, (checkpoint - last_t).total_seconds() / 3600.0)
    horizon_h = max(0.0, (target - checkpoint).total_seconds() / 3600.0)

    if len(snaps) >= 3:
        times = [read_time(h) for h in snaps]
        xs = [x_of(t) for t in times]
        ys = [math.log(max(1.0, float(h["players_tracked"]))) for h in snaps]
        ws = recency_weights(times, times[-1])
        a, g, _se = wls(xs, ys, ws)
        n_tgt = math.exp(a + g * x_of(target))
        # Staleness catch-up: the displayed N is from the last
        # refresh, so the true N at the checkpoint is already
        # higher; grow it at the (decaying) launch rate, capped.
        if staleness_h > 1.0:
            g_bar = growth_rate_mean(last_t, checkpoint)
            n_tgt *= math.exp(g_bar * min(staleness_h, 6.0))
            n_tgt = min(n_tgt, n_last * 12.0)
        return max(n_last, n_tgt), staleness_h

    # Flat history: the launch prior.  The snapshot is as old as
    # the first read (or older); grow from it at the decaying
    # launch rate over the whole span to the target, capped 12x.
    g_bar = growth_rate_mean(first_t, target)
    span_h = max(0.0, (target - first_t).total_seconds() / 3600.0)
    n_tgt = n_last * math.exp(g_bar * span_h)
    return min(n_tgt, n_last * 12.0), staleness_h


def trophy_forecasts(case, snaps, checkpoint, target, n_target):
    """Per-trophy median logit and spread at the target.

    Returns a list, in trophy order, of (median_logit, sigma).
    """
    horizon_h = max(0.0, (target - checkpoint).total_seconds() / 3600.0)
    horizon_d = horizon_h / 24.0
    x_tgt = x_of(target)
    last = snaps[-1]
    n_last = float(last["players_tracked"])
    times = [read_time(h) for h in snaps]
    xs = [x_of(t) for t in times]
    ws = recency_weights(times, times[-1])
    informative = len(snaps) >= 3

    # Median gap between distinct snapshots, for the refresh term.
    gaps = [
        (times[i + 1] - times[i]).total_seconds() / 3600.0
        for i in range(len(times) - 1)
    ]
    gap_h = sorted(gaps)[len(gaps) // 2] if gaps else REFRESH_GAP_DEFAULT_H
    gap_h = min(max(gap_h, 0.5), 24.0)
    refresh = REFRESH_SIGMA * math.sqrt(1.0 - math.exp(-horizon_h / gap_h)) \
        if horizon_h > 0 else 0.0
    rw = RW_SIGMA_PER_SQRT_DAY * math.sqrt(max(horizon_d, 0.0))

    # ---- per-trophy trend fits on the recent window ----
    # Window: distinct snapshots from the last 24 h; the full
    # history when that window holds fewer than 3 snapshots.
    last_time = times[-1]
    win_idx = [j for j, t in enumerate(times)
               if (last_time - t).total_seconds() / 3600.0 <= 24.0]
    if len(win_idx) < 3:
        win_idx = list(range(len(snaps)))
    slopes = []
    per_trophy = []
    for i, trophy in enumerate(case["trophies"]):
        pts = []
        for j in win_idx:
            r = snaps[j]["rates"][i]
            if r is None:
                continue
            pts.append((xs[j], logit(r), ws[j]))
        if len(pts) >= 2:
            a, b, se = wls([p[0] for p in pts], [p[1] for p in pts],
                           [p[2] for p in pts])
            slopes.append(b)
            # Residual spread of the window fit (weighted RMS).
            sw = sum(p[2] for p in pts)
            resid = math.sqrt(
                sum(p[2] * (p[1] - (a + b * p[0])) ** 2 for p in pts) / sw
            ) if sw > 0 else 0.0
        else:
            a, b, se, resid = 0.0, 0.0, 0.0, 0.0
        per_trophy.append((a, b, se, resid, len(pts)))
    slopes.sort()
    pool = slopes[len(slopes) // 2] if slopes else 0.0

    rho = 0.8 * (1.0 - math.exp(-horizon_h / 60.0)) if horizon_h > 0 else 0.0
    out = []
    for i, trophy in enumerate(case["trophies"]):
        _a, b, se, resid, npts = per_trophy[i]
        r_last = last["rates"][i]
        if r_last is None:
            r_last = 0.01
        cap = TIER_CAP.get(trophy.get("tier", "bronze"), 90.0)

        if r_last >= NEARLY_EVERYONE:
            # Everyone earns the opening trophy within minutes:
            # it stays near 100% whatever the population does.
            med = logit(NEARLY_EVERYONE_MEDIAN)
            sigma = math.sqrt(0.05 ** 2 + rw ** 2 * 0.25)
            out.append((med, sigma))
            continue

        if informative and npts >= 3:
            # Shrink the slope toward the pooled median, then
            # anchor the forecast on the last observed rate.
            w = npts / (npts + SHRINK_K)
            b_s = w * b + (1 - w) * pool
            med_logit = logit(r_last) + b_s * (x_tgt - xs[-1])
            dx = max(0.0, x_tgt - xs[-1])
            se_term = se * dx
            sigma = math.sqrt(rw ** 2 + se_term ** 2 + resid ** 2
                              + refresh ** 2 + BASE_SIGMA ** 2)
        else:
            # Flat history: dilute the last rate with newcomers.
            factor = (n_last + rho * max(0.0, n_target - n_last)) / max(1.0, n_target)
            r_med = clamp_pct(r_last * factor)
            med_logit = logit(r_med)
            sigma = math.sqrt(rw ** 2 + refresh ** 2 + BASE_SIGMA ** 2)

        # Tier cap on the median (logit of the cap).
        med_logit = min(med_logit, logit(cap))
        out.append((med_logit, sigma))
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    snaps = distinct_snapshots(case["history"])
    n_target, _stale = population_forecast(snaps, checkpoint, target)
    meds = trophy_forecasts(case, snaps, checkpoint, target, n_target)
    rows = []
    for trophy, (med_logit, sigma) in zip(case["trophies"], meds):
        qs = []
        for z in Z:
            q = inv_logit(med_logit + z * sigma)
            qs.append(round4(min(100.0, max(0.0, q))))
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
