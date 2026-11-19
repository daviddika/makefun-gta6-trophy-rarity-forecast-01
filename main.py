"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Model idea
----------
A printed earn rate is a ratio:  rate = earners / players_tracked.  The two
sides move for different reasons, so one candidate models them separately
and divides at the end, instead of extrapolating the ratio itself.

* players_tracked P(t): the page only changes when Exophase refreshes, so the
  history is a step function.  Collapse it to distinct states, fit
  log P against log(1 + hours since release) by least squares (launch-week
  growth decelerates, so a power law fits better than an exponential), and
  extrapolate to the target, discounted by the expected staleness of the
  answering read (the answer is the page as of its last refresh before the
  target, so the trend is read half a typical refresh interval early).
* earners E_i(t) per trophy: implied by rate_i * P / 100 (the implied counts
  sum exactly to the page's trophies_earned, so the reconstruction is sound).
  Earners only grow, so fit log(E_i + 0.5) against log(1 + hours) per trophy
  over the last 24 h of states, shrinking each slope toward its tier's
  median slope (weak fits borrow strength from their tier).  Where a trophy
  has too few usable reads, a class prior slope (story missions earn
  fastest, platinum slowest) stands in.  The trend-like candidates all fit
  the last 24 h only: the early launch surge makes a whole-history slope
  far too steep to extrapolate 4.5 days out.
* rate_i(target) = 100 * E_i / P, clamped to the printable 0.01..99.99.
  The one trophy sitting at 100% ("Just Like Paradise") survived a 6x player
  refresh still at 100%, so it is forecast tight against 100%.

That earners/players candidate is one of five in an ensemble.  The others
are a saturating progression curve (a trophy's rate rises toward a ceiling
as the cohort plays through the game: r(h) = R * h / (h + tau), fitted in
log space over a grid of tau), a logit-linear trend, a downside-only decay
(logit(rate) linear in hours, used only while the rate is falling -- a
fall that is exponential in h is invisible to the log1p(h) candidates),
and plain persistence.
Each candidate is backtested per trophy -- refit on the first 70% of the
history, scored on the last 30% -- and the ensemble weights candidates by
that held-out error, so a trophy whose rates are falling is forecast by the
model that actually predicted its falls, and a trophy that saturates is
forecast by the curve that saturates.  The spread (logit scale) combines
the ensemble's backtest error, the disagreement between candidates, the
per-hour logit volatility of the page (refresh/staleness uncertainty) and a
model floor, so honest 90% coverage is near 0.9 instead of the house
baseline's ~0.

Flat histories (the practice cases: every read identical, 638 players)
carry no trend signal at all, so the program switches to a refresh-mixture
prior instead of pretending a flat line is a forecast: the page has been
unchanged for the whole observed span, so a refresh before the target is a
real possibility.  With probability p (from a lognormal refresh-interval
prior conditioned on the current state's age) the refresh lands and every
rate is diluted by ~players growth / earners growth; with probability 1-p
the page is still frozen and the rates are exactly the last ones.  The five
quantiles are the quantiles of that two-component mixture, computed on a
grid -- a wide, asymmetric band whose median sits between the two modes.

Standard library only, deterministic (no randomness, no wall-clock reads
beyond the case file), fast: a few thousand arithmetic operations.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release of the game, from the case's game.release_date (2026-11-19).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return clamp(100.0 / (1.0 + math.exp(-lg)))
    e = math.exp(lg)
    return clamp(100.0 * e / (1.0 + e))


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (same convention as the house starter)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_sd, xbar, sxx, n)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return ybar, 0.0, 0.0, xbar, 0.0, n
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    a = ybar - b * xbar
    dof = max(n - 2, 1)
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / dof)
    return a, b, resid, xbar, sxx, n


def trophy_class(trophy: dict) -> str:
    """Progression class from tier + description. Story missions are earned
    early and fast; grinds, 100% completion and the platinum crawl."""
    if trophy["tier"] == "platinum":
        return "platinum"
    desc = trophy["description"]
    if trophy["id"] == 14 or "100% Completion" in desc:
        return "completion"
    if "final mission" in desc and trophy["tier"] == "gold":
        return "final"
    if desc.startswith("Complete Chapter") or "final mission for" in desc:
        return "mission"
    if desc.startswith("Complete "):
        return "story"
    if any(k in desc for k in ("250 Stunts", "100 locations", "Collectible Card",
                               "50 Challenges", "100 vehicles", "5 Bank Vaults",
                               "all of Leonida", "every spot", "20 different races",
                               "gold medal", "all Shooting Range")):
        return "grind"
    return "activity"


# Prior log-earner growth per hour of staleness, by class (flat-history mode).
PRIOR_EARNER_RATE = {
    "story": 0.045, "mission": 0.045, "activity": 0.030, "grind": 0.020,
    "silver": 0.025, "gold": 0.015, "platinum": 0.010, "completion": 0.012,
    "final": 0.020,
}
# Fallback log-earner slope d ln E / d ln(1+h) by class (fit mode, weak fits).
PRIOR_SLOPE = {
    "story": 1.2, "mission": 1.0, "activity": 0.7, "grind": 0.5,
    "silver": 0.7, "gold": 0.4, "platinum": 0.2, "completion": 0.3,
    "final": 0.5,
}


def distinct_states(case):
    """Collapse the history to distinct page states (consecutive identical
    reads are one refresh state), keeping each state's first-seen time."""
    states = []
    for read in sorted(case["history"], key=lambda r: r["at"]):
        rates = tuple(read["rates"])
        key = (read["players_tracked"], rates)
        if states and states[-1]["key"] == key:
            continue
        states.append({
            "key": key,
            "at": parse_time(read["at"]),
            "players": read["players_tracked"],
            "rates": rates,
        })
    return states


def hours_since_release(t: datetime) -> float:
    return max((t - RELEASE).total_seconds() / 3600.0, 0.0)


def mixture_quantiles(p_refresh, mu_refresh, sd_refresh, sd_frozen=0.05):
    """Quantiles of the two-component mixture for a log-rate shift:
    with probability p_refresh the shift is N(mu_refresh, sd_refresh)
    (a refresh dilutes the rates); otherwise it is N(0, sd_frozen)
    (the page is still frozen).  Inverted on a grid, deterministically."""
    lo, hi, step = -6.0, 2.0, 0.01
    grid = []
    d = lo
    while d <= hi + 1e-9:
        cdf = (p_refresh * norm_cdf((d - mu_refresh) / sd_refresh)
               + (1.0 - p_refresh) * norm_cdf(d / sd_frozen))
        grid.append((d, cdf))
        d += step
    out = []
    for level in LEVELS:
        q = grid[-1][0]
        for d, cdf in grid:
            if cdf >= level:
                q = d
                break
        out.append(q)
    return out


def forecast_flat(case, states):
    """Flat history: no trend signal.  Refresh-mixture prior (see module
    docstring).  Returns rows for every trophy."""
    last = states[-1]
    checkpoint = parse_time(case["at"])
    horizon_h = case["horizon_minutes"] / 60.0
    age_h = max((checkpoint - states[0]["at"]).total_seconds() / 3600.0, 0.25)
    stale_h = age_h + horizon_h

    # P(refresh before the target): refresh interval ~ LogNormal(median
    # 4.5 h, sigma 0.5), conditioned on the current state already being
    # `age_h` old.  Clamped so the mixture never collapses to one mode.
    mu_i, sd_i = math.log(4.5), 0.5
    p_old = 1.0 - norm_cdf((math.log(age_h) - mu_i) / sd_i)
    p_by_target = norm_cdf((math.log(stale_h) - mu_i) / sd_i)
    p_refresh = min(0.9, max(0.1, (p_by_target - (1.0 - p_old)) / max(p_old, 1e-9)))

    # Players jump at a refresh grows with the staleness it reveals;
    # the one refresh on record was 6.06x after ~4.5 h of staleness.
    mu_log_players = min(3.0, max(0.5, 1.8 * (stale_h / 4.5) ** 0.6))
    sd_log_players = 0.5
    sd_log_earners = 0.35

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 1.0
        if last_rate >= 99.99:
            # Survived every refresh at 100%: forecast tight against 100%.
            rows.append({"id": trophy["id"], "q": [99.5, 99.9, 99.99, 99.99, 99.99]})
            continue
        cls = trophy_class(trophy)
        mu_log_earners = PRIOR_EARNER_RATE[cls] * stale_h
        mu_refresh = mu_log_earners - mu_log_players
        sd_refresh = math.sqrt(sd_log_players ** 2 + sd_log_earners ** 2)
        shifts = mixture_quantiles(p_refresh, mu_refresh, sd_refresh)
        base = logit(last_rate)
        qs = [round4(inv_logit(base + s)) for s in shifts]
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return rows


# --- ensemble candidates for fit mode ------------------------------------
# Each candidate is a function (states, trophy_index, h_eval, ctx) -> rate
# in percent, where states is a list of distinct states and ctx carries
# precomputed fits for that state list.

TAUS = [2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 350, 600, 1000]


def fit_hyperbola(hs, rs):
    """Best r(h) = R * h / (h + tau) in log space, over a grid of tau.
    Returns (tau, R).  Rates of 0 are skipped (log undefined)."""
    best = None
    pts = [(h, r) for h, r in zip(hs, rs) if r is not None and r > 0 and h > 0]
    if not pts:
        return None
    for tau in TAUS:
        logs = [math.log(r) - math.log(h / (h + tau)) for h, r in pts]
        log_r = sum(logs) / len(logs)
        sse = sum((l - log_r) ** 2 for l in logs)
        if best is None or sse < best[0]:
            best = (sse, tau, math.exp(log_r))
    return best[1], best[2]


def fit_players_loglog(states):
    """log P = a + b*log1p(h) over the given states. Returns (a, b)."""
    xs = [math.log1p(hours_since_release(s["at"])) for s in states]
    a, b, _, _, _, _ = ols(xs, [math.log(max(s["players"], 1)) for s in states])
    if b < 0:
        b = 0.0
        a = sum(math.log(max(s["players"], 1)) for s in states) / len(states)
    return a, b


def fit_earner_loglog(states, idx):
    """log(E + 0.5) = a + b*log1p(h) for one trophy. Returns
    (a, b, resid, xbar, sxx, n) or None when unusable."""
    xs, ys = [], []
    for s in states:
        r = s["rates"][idx]
        if r is None:
            continue
        xs.append(math.log1p(hours_since_release(s["at"])))
        ys.append(math.log(r * s["players"] / 100.0 + 0.5))
    if len(xs) < 2:
        return None
    return ols(xs, ys)


def tier_slopes_for(states, trophies):
    """Median fitted earner slope per tier, from the given states."""
    out = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = []
        for i, t in enumerate(trophies):
            if t["tier"] != tier:
                continue
            f = fit_earner_loglog(states, i)
            if f is not None and f[5] >= 3 and f[2] < 1.0:
                slopes.append(f[1])
        if slopes:
            out[tier] = median(slopes)
    return out


def recent_slope(states, idx):
    """OLS slope of logit(rate) vs log1p(hours) over the last 24 h of
    states for one trophy (0 when too few points).  Negative means the
    trophy's rate is currently falling."""
    from datetime import timedelta
    last_at = states[-1]["at"]
    xs, ys = [], []
    for s in states:
        if s["at"] < last_at - timedelta(hours=24):
            continue
        r = s["rates"][idx]
        if r is not None:
            xs.append(math.log1p(hours_since_release(s["at"])))
            ys.append(logit(r))
    return ols(xs, ys)[1] if len(xs) >= 3 else 0.0


def candidate_rates(states, trophies, h_eval, ctx):
    """All four candidates' rates at h_eval, per trophy.  ctx holds the
    tier slopes for these states.  Trend-like candidates fit the last 24 h
    of states only: the early launch surge makes a whole-history slope
    far too steep to extrapolate 4.5 days out (it is the house curve's
    shape, and the house fits the last 24 h for the same reason)."""
    from datetime import timedelta
    x_eval = math.log1p(max(h_eval, 0.0))
    last_at = states[-1]["at"]
    recent = [s for s in states if s["at"] >= last_at - timedelta(hours=24)]
    if len(recent) < 3:
        recent = states
    a_p, b_p = fit_players_loglog(recent)
    p_pred = math.exp(a_p + b_p * x_eval)
    last_p = states[-1]["players"]
    p_pred = min(max(p_pred, float(last_p)), last_p * 50.0)
    out = []
    for i, trophy in enumerate(trophies):
        hs = [hours_since_release(s["at"]) for s in states]
        rs = [s["rates"][i] for s in states]
        last_rate = next((r for r in reversed(rs) if r is not None), 1.0)
        # Recent logit slope, for gating the rising-only hyperbola.
        r_slope = recent_slope(states, i)
        # hyperbola (saturating progression, full history)
        hyp = fit_hyperbola(hs, rs)
        r_hyp = clamp(hyp[1] * h_eval / (h_eval + hyp[0])) if hyp else last_rate
        if r_slope < 0:
            # Rates are falling: a rising-only curve may not predict above
            # the last rate.
            r_hyp = min(r_hyp, last_rate)
        # logit-linear trend over the last 24 h (house curve shape)
        rxs, rys = [], []
        for s in recent:
            r = s["rates"][i]
            if r is not None:
                rxs.append(math.log1p(hours_since_release(s["at"])))
                rys.append(logit(r))
        if len(rxs) >= 3:
            a_t, b_t, _, _, _, _ = ols(rxs, rys)
            r_trend = inv_logit(a_t + b_t * x_eval)
        else:
            r_trend = last_rate
        # earners / players (log-log, last 24 h for both sides)
        f = fit_earner_loglog(recent, i)
        if f is not None and f[5] >= 3:
            b_tier = ctx["tiers"].get(trophy["tier"], PRIOR_SLOPE[trophy_class(trophy)])
            w = f[5] / (f[5] + 4.0)
            b = w * f[1] + (1.0 - w) * b_tier
            xs_f = [math.log1p(hours_since_release(s["at"])) for s in recent
                    if s["rates"][i] is not None]
            ys_f = [math.log(s["rates"][i] * s["players"] / 100.0 + 0.5) for s in recent
                    if s["rates"][i] is not None]
            a = sum(ys_f) / len(ys_f) - b * (sum(xs_f) / len(xs_f))
            e_pred = math.exp(a + b * x_eval)
            r_loglog = clamp(100.0 * e_pred / p_pred)
        elif f is not None:
            b = PRIOR_SLOPE[trophy_class(trophy)]
            x_last = math.log1p(hours_since_release(states[-1]["at"]))
            e_last = max(last_rate * last_p / 100.0, 0.0)
            a = math.log(e_last + 0.5) - b * x_last
            e_pred = math.exp(a + b * x_eval)
            r_loglog = clamp(100.0 * e_pred / p_pred)
        else:
            r_loglog = last_rate
        # Downside-only decay: logit(rate) linear in HOURS over the last
        # 24 h, used only while the rate is falling.  A fall that is
        # exponential in h (dilution compounding) is invisible to the
        # log1p(h) candidates; this one can see it.  When the rate is not
        # falling it is just persistence, so it can never push a forecast up.
        hxs = [hours_since_release(s["at"]) for s in recent
               if s["rates"][i] is not None]
        hys = [logit(s["rates"][i]) for s in recent if s["rates"][i] is not None]
        if len(hxs) >= 3:
            a_d, b_d, _, _, _, _ = ols(hxs, hys)
            r_decay = inv_logit(a_d + b_d * h_eval) if b_d < 0 else last_rate
        else:
            r_decay = last_rate
        out.append({"hyp": r_hyp, "trend": r_trend, "loglog": r_loglog,
                    "decay": r_decay, "persist": last_rate})
    return out


CANDIDATES = ("hyp", "trend", "loglog", "decay", "persist")


def forecast_fit(case, states):
    """History with real variation: ensemble of four candidates, weighted
    by per-trophy backtest error, with a spread from backtest error,
    candidate disagreement, page volatility and a model floor."""
    target = parse_time(case["target"])
    h_target = hours_since_release(target)

    gaps = [(states[j + 1]["at"] - states[j]["at"]).total_seconds() / 3600.0
            for j in range(len(states) - 1)]
    refresh_h = median(gaps) if gaps else 5.0
    lag_h = min(12.0, max(0.0, refresh_h / 2.0))
    h_eval = max(h_target - lag_h, 0.0)

    # Pooled per-hour logit volatility between distinct states.
    vols = []
    for j in range(len(states) - 1):
        dt = max((states[j + 1]["at"] - states[j]["at"]).total_seconds() / 3600.0, 0.25)
        for i in range(len(case["trophies"])):
            r0, r1 = states[j]["rates"][i], states[j + 1]["rates"][i]
            if r0 is None or r1 is None or r0 <= 0 or r1 <= 0:
                continue
            vols.append(abs(logit(r1) - logit(r0)) / dt)
    vol_h = min(max(median(vols) if vols else 0.05, 0.02), 0.30)
    sigma_lag = vol_h * math.sqrt(lag_h + 1.0)

    # Backtest split: fit on the first part of the history, score on the
    # rest.  70/30 with temperature 0.25 won a sweep over synthetic
    # final-like worlds (see validation notes in the README).
    n = len(states)
    frac = float(os.environ.get("MF_BACKTEST_FRAC", "0.7"))
    temp = float(os.environ.get("MF_BACKTEST_TEMP", "0.25"))
    n_early = max(3, int(frac * n))
    early, held = states[:n_early], states[n_early:]
    ctx_early = {"tiers": tier_slopes_for(early, case["trophies"])}
    # Score each candidate by its prediction at each held-out state's h
    # (candidates are refit-free here: they were fitted on `early`).
    errs = {c: [0.0] * len(case["trophies"]) for c in CANDIDATES}
    blend_err = [0.0] * len(case["trophies"])
    counts = [0] * len(case["trophies"])
    for s in held:
        h_s = hours_since_release(s["at"])
        cand_s = candidate_rates(early, case["trophies"], h_s, ctx_early)
        for i, trophy in enumerate(case["trophies"]):
            actual = s["rates"][i]
            if actual is None or actual <= 0:
                continue
            la = logit(actual)
            for c in CANDIDATES:
                errs[c][i] += abs(logit(cand_s[i][c]) - la)
            counts[i] += 1
    for c in CANDIDATES:
        for i in range(len(case["trophies"])):
            errs[c][i] = errs[c][i] / counts[i] if counts[i] else 0.5

    # Weights from backtest error (softmax in 1/error space via exp).
    weights = []
    for i in range(len(case["trophies"])):
        ws = {c: math.exp(-errs[c][i] / temp) for c in CANDIDATES}
        tot = sum(ws.values())
        weights.append({c: ws[c] / tot for c in CANDIDATES})

    # Blended backtest error (for the spread): weighted prediction error.
    for s in held:
        h_s = hours_since_release(s["at"])
        cand_s = candidate_rates(early, case["trophies"], h_s, ctx_early)
        for i, trophy in enumerate(case["trophies"]):
            actual = s["rates"][i]
            if actual is None or actual <= 0 or counts[i] == 0:
                continue
            la = logit(actual)
            blended = sum(weights[i][c] * logit(cand_s[i][c]) for c in CANDIDATES)
            blend_err[i] += abs(blended - la)
    for i in range(len(case["trophies"])):
        blend_err[i] = blend_err[i] / counts[i] if counts[i] else 0.5

    # Final predictions on all states.
    ctx_full = {"tiers": tier_slopes_for(states, case["trophies"])}
    cand_full = candidate_rates(states, case["trophies"], h_eval, ctx_full)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = states[-1]["rates"][i]
        if last_rate is None:
            last_rate = next((s["rates"][i] for s in reversed(states)
                              if s["rates"][i] is not None), 1.0)
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.5, 99.9, 99.99, 99.99, 99.99]})
            continue
        logits = {c: logit(cand_full[i][c]) for c in CANDIDATES}
        med_logit = sum(weights[i][c] * logits[c] for c in CANDIDATES)
        mean_l = med_logit
        var_dis = sum(weights[i][c] * (logits[c] - mean_l) ** 2 for c in CANDIDATES)
        sigma = math.sqrt(min(max(blend_err[i], 0.25), 1.2) ** 2
                          + var_dis + sigma_lag ** 2 + 0.35 ** 2)
        sigma = min(max(sigma, 0.5), 2.0)
        # Downside tail: when a trophy's rate is falling steeply, the fall
        # can run much further than the rise ever could (rates are bounded
        # below by 0 and dilution compounds), so widen the lower quantiles.
        slope = recent_slope(states, i)
        sigma_down = min(sigma * (1.0 + 1.5 * max(0.0, -slope - 0.5)), 2.0)
        qs = []
        for z in Z:
            qs.append(round4(inv_logit(med_logit + z * (sigma_down if z < 0 else sigma))))
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return rows


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        # No history at all: a flat 1% with a wide band beats crashing
        # (a crash scores the case as last-read rates and voids prizes).
        return {"trophies": [{"id": t["id"], "q": [0.05, 0.3, 1.0, 3.0, 10.0]}
                             for t in case["trophies"]]}
    states = distinct_states(case)
    players_seen = {s["players"] for s in states}
    log_p_range = (math.log(max(players_seen)) - math.log(min(players_seen))) if players_seen else 0.0
    flat = len(states) < 4 or len(players_seen) < 3 or log_p_range < 0.10
    rows = forecast_flat(case, states) if flat else forecast_fit(case, states)
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
