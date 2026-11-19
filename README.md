# token-radar-01-v4

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/Hoppererin/token-radar-01-v4/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/Hoppererin/token-radar-01-v4/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

A token launch radar for make.fun: it ranks all 226 pre-token protocols in the frozen DefiLlama snapshot by the value their own figures imply, and gives each one a concrete six-step farm plan with chain-specific gas costs. The model is fit at run time from the snapshot's 128 token-bearing protocols — a grouped comps curve (shrunk category-group medians on ln(market cap) vs ln(TVL)) blended 90/10 with a median (L1) regression that also reads seven-day growth, age and chains. The 90/10 blend was chosen by a sweep, not by taste: it has the best five-fold cross-validation (1.266 vs 1.268 for the grouped curve alone and 1.273 for v3's 80/20) and the best blind hold-out (35% of predictions inside ±30%, median error −5.5% in levels, vs 29% and −6.4% for v3). Validation on the same protocol for every number: 5-fold CV 1.27, walk-forward 1.05, blind 20% hold-out median error −5.5% in levels with 35% of predictions inside ±30% (the single global curve it replaces scores 1.29 / 1.13 / −19.5%).

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <snapshot.json>
```

Node 22, standard library only. It reads `/input/snapshot.json` (or `--input`), writes the ranked board to `/output/board.json`, the dashboard data to `/output/data.js`, copies `dashboard/` and `screenshots/` to `/output`, and writes `/output/dossier.json` with the vendored make.fun library.

## The dashboard

A static page, no network: the ranked board (sortable, filterable by category, chain and watchlist, with a search box), a detail view for every protocol showing how its implied value was built — grouped curve, second opinion and blend, with a value build-up bar — and its farm plan, a live-token view of the market values the model learns from, and a method page with the model, the group offsets and the validation numbers.

---

<sub>Entered on [make.fun](https://make.fun/?ref=dhj53q2r&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
