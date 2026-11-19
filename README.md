# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/daviddika/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/daviddika/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

**Dilution Ensemble:** forecast each GTA VI (PS5) trophy's Exophase earn rate
at 2026-11-26T00:00:00Z. A printed rate is a ratio — earners / players
tracked — and the two sides move for different reasons, so the program
models them separately and divides at the end, instead of extrapolating
the ratio itself.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` case in
the input directory, writes one `<id>.out` per case. Deterministic and fast
(~0.1 s on a 358-read case, limit 8 s).

## Model

- **History is a step function.** Exophase only changes the page when it
  refreshes, so the history is collapsed to distinct page states first.
- **Fit mode** (history has real variation, as the final's will): five
  candidates per trophy, weighted by a per-trophy backtest (refit on the
  first 70% of states, scored on the last 30%, softmax temperature 0.25):
  1. **hyperbola** — saturating progression curve `r(h) = R·h/(h+τ)`,
     fitted in log space over a grid of τ (gated: never predicts above the
     last rate while rates are falling);
  2. **trend** — logit-linear in `log(1+h)` over the last 24 h (the house
     curve's shape);
  3. **loglog** — earners / players decomposition: `log(E+0.5)` and
     `log P` fitted against `log(1+h)` over the last 24 h, earner slopes
     shrunk toward their tier's median;
  4. **decay** — downside-only: `logit(rate)` linear in *hours* over the
     last 24 h, used only while the rate is falling (a fall that is
     exponential in h is invisible to the `log(1+h)` candidates);
  5. **persistence** — the last read's rate.
  The median is the weight-averaged logit of the candidates. The spread
  combines the blended backtest error, candidate disagreement, page
  volatility (refresh/staleness, trend read half a typical refresh
  interval early) and a model floor — with a wider lower tail when a
  trophy's rate is falling steeply. The trophy at 100% ("Just Like
  Paradise", which survived a 6× player refresh still at 100%) is
  forecast tight against 100%.
- **Flat mode** (practice cases: every read identical, 638 players): no
  trend signal, so instead of a flat line the program uses a
  refresh-mixture prior. With probability *p* (lognormal refresh-interval
  prior conditioned on the current state's age) a refresh lands before
  the target and every rate is diluted by players growth / earners
  growth; with probability *1−p* the page is still frozen. The five
  quantiles are the quantiles of that two-component mixture.

## Validation

Practice cases (answers are server-side, so they were scored against
simulated refresh scenarios — pure dilution through strong early earner
growth; the house baseline's published ~1.51 practice loss calibrates the
plausible middle of that range):

| scenario | this entry | house baseline |
|---|---|---|
| pure dilution | 0.69 | 1.69 |
| weak earner growth | 0.55 | 1.45 |
| moderate | 0.44 | 1.14 |
| strong early growth | 0.37 | 0.80 |
| early boom | 0.32 | 0.57 |

Synthetic final-like worlds (358 reads, 4.5-day horizon, true players and
earn-rate curves simulated under ten different dynamics — power-law and
saturating player growth, hyperbolic / exponential / logistic rate
curves, rise-then-fall rates, and power-law earners-vs-players worlds
where rates fall or mix): mean pinball loss **0.12–0.14** in eight worlds
and **1.00–1.02** in the two exponential-collapse stress worlds, beating
both the house baseline (0.28–0.37 / 1.64) and the house curve baseline
(0.21–0.36 / 1.10) in **all ten worlds**. 90% coverage 0.96–0.98 in the
eight main worlds. Outputs are valid JSON in the required shape, 55
trophies, quantiles non-decreasing, ≤ 65,536 bytes, byte-identical across
two runs.

Known weak spot: if most rates collapsed exponentially between the
deadline and the target (the two stress worlds), the median would still
sit too high and 90% coverage would drop to ~0.18 there — but the loss
still beats both house entries in that world, and that shape (earners
essentially stopping while players grow 5×) is not what the launch-week
dynamics point to.

---

<sub>Entered on [make.fun](https://make.fun/?ref=2vjd5xyx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
