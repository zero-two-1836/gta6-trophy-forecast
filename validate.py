"""Local validation for the trophy forecaster (NOT part of the entry).

1. Practice answers are reconstructed, because the kit does not ship
   them: competition.yaml states the refresh between 04:00Z and
   04:30Z took players 638 -> 3,868 and trophies earned 943 -> 4,276,
   every rate but the 100% one fell, and the 05:00Z read was
   unchanged.  Per-trophy actual counts are not published, so three
   reconstructions are scored; the flat baseline must reproduce the
   house's reported practice loss ~1.51 on all of them (a sanity
   check that the reconstruction is in the right ballpark).
     R1: new earners proportional to old counts (total new excl.
         the 100% trophy = 103, spread pro-rata)
     R2: +1.5 new earners per trophy
     R3: 0 new earners
2. Synthetic final-like scenarios: a generative cohort model with
   two days of 30-minute reads and 4-hour refreshes, cut at 59.5 h,
   target day 7.  Scenarios vary dilution vs progression strength.
3. Format checks on the program's own outputs: valid JSON, 55
   rows, quantiles non-decreasing in 0..100, size under the limit,
   deterministic across two runs.

Usage: python3 validate.py  (runs main.py on the kit practice
inputs first, then scores everything)
"""

import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.expanduser(
    "~/workspace/makefun-gta6-trophy-forecast/"
    "gta6-trophy-rarity-forecast-01-kit"
)
Z = [-1.645, -0.674, 0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(p):
    c = clamp(p)
    return math.log(c / (100 - c))


def inv_logit(y):
    return 100 / (1 + math.exp(-max(-20, min(20, y))))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if v == int(v) else v


def pinball(forecast, actual):
    """forecast: {id: [q5]}; actual: {id: rate}. Mean pinball, logit scale."""
    tot, n = 0.0, 0
    for tid, y in actual.items():
        ly = logit(y)
        for tau, q in zip(LEVELS, forecast[tid]):
            d = ly - logit(q)
            tot += 2 * max(tau * d, (tau - 1) * d)
            n += 1
    return tot / n


def flat_baseline(case):
    """House persistence baseline, from the kit's starter."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    out = {}
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i]
        lg = logit(r if r is not None else 0)
        out[t["id"]] = [round4(inv_logit(lg + z * spread)) for z in Z]
    return out


def trend_baseline(case):
    """House curve baseline, approximated: OLS logit vs ln(1+h)
    over the last 24 h of reads, read at the target; spread is the
    random walk's plus the fit residual's.  Fewer than 3 reads
    falls back to persistence."""
    from datetime import datetime

    rel = datetime.fromisoformat("2026-11-19T00:00:00+00:00")

    def hrs(s):
        return max(0.0, (datetime.fromisoformat(s.replace("Z", "+00:00")) - rel).total_seconds() / 3600)

    tgt = hrs(case["target"])
    xt = math.log1p(tgt)
    hist = case["history"]
    cut = hrs(hist[-1]["slot"]) - 24
    use = [h for h in hist if hrs(h["slot"]) >= cut]
    if len(use) < 3:
        return flat_baseline(case)
    spread_rw = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    out = {}
    for i, t in enumerate(case["trophies"]):
        xs, ys = [], []
        for h in use:
            r = h["rates"][i]
            if r is None:
                continue
            xs.append(math.log1p(hrs(h["slot"])))
            ys.append(logit(r))
        if len(xs) < 3:
            return flat_baseline(case)
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 1e-12 else 0.0
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2)) if n > 2 else 0.0
        lg = a + b * xt
        sp = math.sqrt(spread_rw ** 2 + sd ** 2)
        out[t["id"]] = [round4(inv_logit(lg + z * sp)) for z in Z]
    return out


def run_entry(outdir):
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = os.path.join(KIT, "input")
    env["MAKEFUN_OUTPUT_DIR"] = outdir
    os.makedirs(outdir, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(HERE, "main.py")],
                   env=env, check=True)


def load_forecast(path):
    d = json.load(open(path))
    return {r["id"]: r["q"] for r in d["trophies"]}


def practice_answers(case, mode):
    """Reconstruct the refreshed page's rates from the kit's totals."""
    last = case["history"][-1]
    n0, n1 = 638, 3868
    actual = {}
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i]
        if r is not None and r >= 99.99:
            actual[t["id"]] = 100.0
            continue
        c = round(r * n0 / 100)
        if mode == "R1":
            c1 = c * 1.34
        elif mode == "R2":
            c1 = c + 1.5
        else:
            c1 = c
        actual[t["id"]] = 100.0 * c1 / n1
    return actual


def gen_scenario(seed, n_growth, prog_scale, refresh_h=4.0):
    """Generative cohort model: 2 days of reads, target day 7."""
    rng = random.Random(seed)
    arch = []
    for i in range(55):
        if i == 0:
            L, a50, k = 0.001, 400, 0.03      # platinum
        elif i == 1:
            L, a50, k = 0.90, 20, 0.10        # the 100% oddity
        elif i == 13:
            L, a50, k = 0.002, 500, 0.03      # gold, late
        elif i < 14:
            L, a50, k = rng.uniform(0.25, 0.6), rng.uniform(40, 120), 0.05
        elif i % 5 == 0:
            L, a50, k = rng.uniform(0.02, 0.08), rng.uniform(200, 400), 0.03
        else:
            L, a50, k = rng.uniform(0.05, 0.35), rng.uniform(60, 250), 0.04
        arch.append((L * prog_scale if i != 1 else L, a50, k))

    def n_true(t):
        return 600 * (1 + t / 3.0) ** n_growth

    def frac(age, L, a50, k):
        if age <= 0:
            return 0.0
        return L / (1 + math.exp(-k * (age - a50)))

    def true_rate(t, idx):
        L, a50, k = arch[idx]
        tot, wsum = 0.0, 0.0
        steps = 24
        for j in range(steps + 1):
            s = t * j / steps
            w = n_true(s + 0.5) - n_true(max(0.0, s - 0.5))
            tot += w * frac(t - s, L, a50, k)
            wsum += w
        return tot / max(wsum, 1e-9) * 100

    trophies = [{"id": i + 1, "name": "T%d" % (i + 1), "description": "",
                 "tier": "platinum" if i == 0 else "gold" if i == 13 else "bronze",
                 "points": 15, "secret": False} for i in range(55)]
    hist = []
    last_disp = None
    t = 0.0
    while t <= 59.5:
        if last_disp is None or (t % refresh_h) < 0.31 or t == 0.0:
            nd = n_true(t)
            rates = [round(true_rate(t, i), 2) for i in range(55)]
            last_disp = (nd, rates)
        nd, rates = last_disp
        earned = sum(round(r * nd / 100) for r in rates)
        slot = "2026-11-%02dT%02d:%02d:00Z" % (19 + int(t // 24), int(t % 24), int((t % 1) * 60))
        hist.append({"slot": slot,
                     "at": slot,
                     "players_tracked": int(nd),
                     "trophies_earned": earned, "rates": list(rates)})
        t += 0.5
    nd = n_true(168.0)
    actual = {i + 1: round(true_rate(168.0, i), 2) for i in range(55)}
    case = {"schema": "makefun/gta6-trophies-case/v1", "id": "f00",
            "at": "2026-11-21T11:30:00Z", "horizon_minutes": 6510,
            "target": "2026-11-26T00:00:00Z", "levels": LEVELS,
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19", "source": ""},
            "trophies": trophies, "history": hist, "notes": ""}
    return case, actual


def ours(case):
    import importlib.util
    spec = importlib.util.spec_from_file_location("entry", os.path.join(HERE, "main.py"))
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    f = entry.forecast_case(case, case["id"])
    return {r["id"]: r["q"] for r in f["trophies"]}


def main():
    outdir = os.path.join(HERE, "out")
    run_entry(outdir)
    # Determinism: run twice, compare bytes.
    outdir2 = os.path.join(HERE, "out2")
    run_entry(outdir2)
    print("== FORMAT + DETERMINISM ==")
    for fn in ["p00", "p01"]:
        b1 = open(os.path.join(outdir, fn + ".out"), "rb").read()
        b2 = open(os.path.join(outdir2, fn + ".out"), "rb").read()
        d = json.loads(b1)
        rows = d["trophies"]
        ok = (len(rows) == 55 and set(d.keys()) == {"trophies"}
              and all(set(r.keys()) == {"id", "q"} for r in rows)
              and all(len(r["q"]) == 5 and all(0 <= q <= 100 for q in r["q"])
                      and all(r["q"][k] <= r["q"][k + 1] for k in range(4)) for r in rows)
              and len(b1) <= 65536 and b1 == b2)
        print(f"{fn}: valid={ok} bytes={len(b1)} deterministic={b1 == b2}")

    print("== PRACTICE (reconstructed answers) ==")
    totals = {"flat": [], "trend": [], "ours": []}
    for fn in ["p00", "p01"]:
        case = json.load(open(os.path.join(KIT, "input", fn + ".in")))
        fc = load_forecast(os.path.join(outdir, fn + ".out"))
        for mode in ["R1", "R2", "R3"]:
            act = practice_answers(case, mode)
            lf = pinball(flat_baseline(case), act)
            lt = pinball(trend_baseline(case), act)
            lo = pinball(fc, act)
            totals["flat"].append(lf)
            totals["trend"].append(lt)
            totals["ours"].append(lo)
            print(f"{fn} {mode}: flat={lf:.3f} trend={lt:.3f} OURS={lo:.3f}  (house reports flat 1.51)")
        act = practice_answers(case, "R1")
        fb = flat_baseline(case)
        miss = sum(abs(act[i] - fb[i][2]) for i in act) / len(act)
        print(f"  flat typical miss R1: {miss:.2f} pp (house reports 0.68)")
        miss_o = sum(abs(act[i] - fc[i][2]) for i in act) / len(act)
        cov = sum(1 for i in act if fc[i][0] <= act[i] <= fc[i][4]) / len(act)
        print(f"  OURS typical miss R1: {miss_o:.2f} pp, 90% coverage R1: {cov:.2f}")

    print("== SYNTHETIC FINAL (2d history -> day 7) ==")
    for name, kw in [("base dilution>prog", dict(n_growth=1.1, prog_scale=1.0)),
                     ("fast N growth", dict(n_growth=1.6, prog_scale=1.0)),
                     ("slow N, fast prog", dict(n_growth=0.6, prog_scale=1.4)),
                     ("slow everything", dict(n_growth=0.7, prog_scale=0.6))]:
        case, act = gen_scenario(7, **kw)
        lf = pinball(flat_baseline(case), act)
        lt = pinball(trend_baseline(case), act)
        o = ours(case)
        lo = pinball(o, act)
        miss = sum(abs(act[i] - o[i][2]) for i in act) / len(act)
        print(f"{name}: flat={lf:.3f} trend={lt:.3f} OURS={lo:.3f} (ours median miss {miss:.2f}pp)")

    n = len(totals["ours"])
    print("== PRACTICE TOTALS (mean over 2 cases x 3 reconstructions) ==")
    for k, v in totals.items():
        print(f"  {k}: {sum(v) / len(v):.4f}")


if __name__ == "__main__":
    main()
