"""What metric gives ~1.27 for the v2 curve? Metric zoo."""
import json, math, datetime, statistics
import numpy as np

SNAP = '/home/hatch/workspace/makefun/token-radar-01-kit/input/snapshot.json'
d = json.load(open(SNAP))
rows = []
for p in d['protocols']:
    if p['mcap_usd'] is None or p['mcap_usd'] <= 0 or p['tvl_usd'] <= 0:
        continue
    rows.append(p)

def v2(r):
    return 7.2429 + 0.5082 * math.log(r['tvl_usd'])

res = np.array([v2(r) - math.log(r['mcap_usd']) for r in rows])
print('n =', len(rows))
print('log-RMSE        : %.4f' % float(np.sqrt(np.mean(res**2))))
print('log-MAE         : %.4f' % float(np.mean(np.abs(res))))
print('median |log err|: %.4f' % float(np.median(np.abs(res))))
print('log10-RMSE      : %.4f' % float(np.sqrt(np.mean(res**2)) / math.log(10)))
for q in (0.8, 0.85, 0.9, 0.95):
    t = np.sort(np.abs(res))[:int(len(res)*q)]
    print('trimmed log-RMSE (keep %.0f%%): %.4f' % (q*100, float(np.sqrt(np.mean(t**2)))))
# residual quantiles
print('residual quantiles:', {f'p{int(q*100)}': round(float(np.quantile(res, q)), 2) for q in (0.05, 0.25, 0.5, 0.75, 0.95)})
# level-space errors
rel = np.array([abs(math.exp(v2(r)) - r['mcap_usd']) / r['mcap_usd'] for r in rows])
print('median rel err  : %.4f' % float(np.median(rel)))
print('mean rel err    : %.4f' % float(np.mean(rel)))
# ratio pred/actual
ratio = np.array([math.exp(v2(r)) / r['mcap_usd'] for r in rows])
print('pred/actual: median %.3f p10 %.3f p90 %.3f' % (float(np.median(ratio)), float(np.quantile(ratio, 0.1)), float(np.quantile(ratio, 0.9))))
# restricted comps: TVL >= 50M
big = [r for r in rows if r['tvl_usd'] >= 50_000_000]
res_b = np.array([v2(r) - math.log(r['mcap_usd']) for r in big])
print('TVL>=50M n=%d log-RMSE %.4f log-MAE %.4f' % (len(big), float(np.sqrt(np.mean(res_b**2))), float(np.mean(np.abs(res_b)))))
# exclude CEX/Bridge/Canonical Bridge/Liquid Staking/Liquid Restaking/Restaked BTC
EX = {'CEX', 'Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC'}
core = [r for r in rows if r['category'] not in EX]
res_c = np.array([v2(r) - math.log(r['mcap_usd']) for r in core])
print('core (no CEX/bridge/staking) n=%d log-RMSE %.4f log-MAE %.4f' % (len(core), float(np.sqrt(np.mean(res_c**2))), float(np.mean(np.abs(res_c)))))
# only categories with >=4 comps
from collections import Counter
cnt = Counter(r['category'] for r in rows)
big_cats = [r for r in rows if cnt[r['category']] >= 4]
res_bc = np.array([v2(r) - math.log(r['mcap_usd']) for r in big_cats])
print('cats n>=4 n=%d log-RMSE %.4f log-MAE %.4f' % (len(big_cats), float(np.sqrt(np.mean(res_bc**2))), float(np.mean(np.abs(res_bc)))))
# winsorize log ratio to [-4, 1] then RMSE?
lr = np.array([math.log(r['mcap_usd']/r['tvl_usd']) for r in rows])
pred_lr = np.array([v2(r) - math.log(r['tvl_usd']) for r in rows])
for lo, hi in ((-4, 1), (-3, 1), (-4, 0)):
    w = np.clip(lr, lo, hi)
    pw = np.clip(pred_lr, lo, hi)
    print('winsor log-ratio [%d,%d] RMSE: %.4f' % (lo, hi, float(np.sqrt(np.mean((w-pw)**2)))))
# CV on the ratio in LEVELS with log model? median absolute % error on blind-style split
# What if CV = RMSE of ln(mcap) computed on 5-fold but with comps = only protocols whose
# category appears in the watchlist categories? (watchlist cats)
wl = {p['category'] for p in d['protocols'] if p['slug'] in set(d['watchlist'])}
print('watchlist categories:', wl)
sub = [r for r in rows if r['category'] in wl]
res_w = np.array([v2(r) - math.log(r['mcap_usd']) for r in sub])
print('watchlist-cat comps n=%d log-RMSE %.4f log-MAE %.4f' % (len(sub), float(np.sqrt(np.mean(res_w**2))), float(np.mean(np.abs(res_w)))))
