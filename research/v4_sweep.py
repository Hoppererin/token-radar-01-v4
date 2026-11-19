"""v4 model sweep. Metric: median |log error| (same protocol as v3).
Tests: blend weights, two-level hierarchy, growth interaction, per-group slopes.
Validation: 5-fold CV x5 seeds + walk-forward + 10 blind splits (mean)."""
import math, statistics
import numpy as np
from validate import load, blind_split

d, rows = load()
print('comps n =', len(rows))

GROUP_OF = {
  'Bridge': 'Bridge', 'Canonical Bridge': 'Bridge', 'Cross Chain Bridge': 'Bridge',
  'Liquid Staking': 'Staking', 'Liquid Restaking': 'Staking', 'Staking Pool': 'Staking',
  'Restaking': 'Staking', 'Yield': 'Yield', 'Yield Aggregator': 'Yield',
  'Leveraged Farming': 'Yield', 'Lending': 'Lending', 'CDP': 'Lending',
  'Collateral Management': 'Lending', 'Dexs': 'Dexs', 'Liquidity Manager': 'Dexs',
}
def group_of(c): return GROUP_OF.get(c, c)
def med(v): return float(np.median(np.abs(np.array(v))))

def fit_global(train):
    X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return lambda r: float(b[0] + b[1] * r['ltvl'])

def fit_grouped(train, k=5.0):
    b = fit_global(train)
    res = [r['lmcap'] - b(r) for r in train]
    gm = float(np.median(res))
    by = {}
    for r, e in zip(train, res): by.setdefault(group_of(r['category']), []).append(e)
    off = {}
    for g, lst in by.items():
        m = float(np.median(lst))
        off[g] = (len(lst)/(len(lst)+k))*m + (k/(len(lst)+k))*gm
    return lambda r: float(b(r) + off.get(group_of(r['category']), gm))

def fit_grouped2(train, k1=5.0, k2=3.0):
    """Two-level: pooled-group offset + per-category offset within group (shrunk)."""
    b = fit_global(train)
    res = [r['lmcap'] - b(r) for r in train]
    gm = float(np.median(res))
    byg, byc = {}, {}
    for r, e in zip(train, res):
        byg.setdefault(group_of(r['category']), []).append(e)
        byc.setdefault(r['category'], []).append(e)
    goff = {}
    for g, lst in byg.items():
        m = float(np.median(lst))
        goff[g] = (len(lst)/(len(lst)+k1))*m + (k1/(len(lst)+k1))*gm
    coff = {}
    for c, lst in byc.items():
        g = group_of(c)
        m = float(np.median(lst))
        coff[c] = (len(lst)/(len(lst)+k2))*m + (k2/(len(lst)+k2))*goff[g]
    return lambda r: float(b(r) + coff.get(r['category'], goff.get(group_of(r['category']), gm)))

def fit_ridge_lad(train, lam=10.0, iters=40):
    cats = sorted({r['category'] for r in train})
    chains = ['Ethereum','Solana','Base','Arbitrum','BNB Chain']
    def feat(r):
        v = [1.0, r['ltvl']]
        v += [1.0 if r['category']==c else 0.0 for c in cats]
        ch = r['change_7d'] if r['change_7d'] is not None else 0.0
        v.append(max(-60.0,min(120.0,ch))/100.0)
        v.append(math.log(max(r['age'],30)) if r['age'] else math.log(365.0))
        v.append(float(len(r['chains'])))
        v += [1.0 if c in r['chains'] else 0.0 for c in chains]
        return v
    X = np.array([feat(r) for r in train]); y = np.array([r['lmcap'] for r in train])
    p = X.shape[1]
    diag = np.full(p, lam); diag[0]=0; diag[1]=0
    beta = np.linalg.solve(X.T@X + np.diag(diag), X.T@y)
    for _ in range(iters):
        res = y - X@beta
        w = 1.0/np.maximum(np.abs(res), 0.05)
        XtW = X.T*w
        beta = np.linalg.solve(XtW@X + np.diag(diag), XtW@y)
    return lambda r: float(np.dot(beta, feat(r)))

def blend(f1, f2, w1):
    return lambda r: float(w1*f1(r) + (1-w1)*f2(r))

def fit_growth_adj(train, k=5.0, gcoef=0.3):
    """Grouped + growth interaction: offset scaled by clipped 7d growth."""
    b = fit_global(train)
    res = [r['lmcap'] - b(r) for r in train]
    gm = float(np.median(res))
    by = {}
    for r, e in zip(train, res): by.setdefault(group_of(r['category']), []).append(e)
    off = {}
    for g, lst in by.items():
        m = float(np.median(lst))
        off[g] = (len(lst)/(len(lst)+k))*m + (k/(len(lst)+k))*gm
    def pred(r):
        g = max(-0.5, min(1.0, (r['change_7d'] or 0.0)/100.0))
        return float(b(r) + off.get(group_of(r['category']), gm) * (1.0 + gcoef*g))
    return pred

def fit_group_slope(train, k=5.0):
    """Grouped offset + per-group slope deviation (shrunk)."""
    b = fit_global(train)
    res = [r['lmcap'] - b(r) for r in train]
    gm = float(np.median(res))
    byg = {}
    for r, e in zip(train, res): byg.setdefault(group_of(r['category']), []).append((r, e))
    off, slope = {}, {}
    for g, lst in byg.items():
        m = float(np.median([e for _, e in lst]))
        off[g] = (len(lst)/(len(lst)+k))*m + (k/(len(lst)+k))*gm
        # slope deviation: regress residual on (ltvl - mean ltvl) within group
        xs = [r['ltvl'] for r, _ in lst]; ys = [e for _, e in lst]
        mx = sum(xs)/len(xs)
        num = sum((x-mx)*(y-m) for x, y in zip(xs, ys))
        den = sum((x-mx)**2 for x in xs)
        ds = num/den if den > 1e-9 else 0.0
        slope[g] = (len(lst)/(len(lst)+10.0))*ds
    def pred(r):
        g = group_of(r['category'])
        return float(b(r) + off.get(g, gm) + slope.get(g, 0.0)*(r['ltvl']-17.0))
    return pred

MODELS = {
  'v2 curve': lambda tr: fit_global(tr),
  'grouped k5': lambda tr: fit_grouped(tr, 5.0),
  'grouped k3': lambda tr: fit_grouped(tr, 3.0),
  'grouped k8': lambda tr: fit_grouped(tr, 8.0),
  'grouped2 k5/k3': lambda tr: fit_grouped2(tr, 5.0, 3.0),
  'grouped2 k5/k8': lambda tr: fit_grouped2(tr, 5.0, 8.0),
  'growth-adj 0.3': lambda tr: fit_growth_adj(tr, 5.0, 0.3),
  'growth-adj 0.6': lambda tr: fit_growth_adj(tr, 5.0, 0.6),
  'group-slope': lambda tr: fit_group_slope(tr, 5.0),
  'blend 1.0/0.0': lambda tr: blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 1.0),
  'blend 0.9/0.1': lambda tr: blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 0.9),
  'blend 0.8/0.2 (v3)': lambda tr: blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 0.8),
  'blend 0.7/0.3': lambda tr: blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 0.7),
  'blend 0.6/0.4': lambda tr: blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 0.6),
  'blend 0.5/0.5': lambda tr: blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 0.5),
}

def cv_wf(fit, data, seeds=(0,1,2,3,4)):
    cv = []
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = np.arange(len(data))
        bycat = {}
        for i, r in enumerate(data): bycat.setdefault(r['category'], []).append(i)
        fold = np.zeros(len(data), dtype=int)
        for c, mem in bycat.items():
            m = np.array(mem); rng.shuffle(m)
            for j, i in enumerate(m): fold[i] = j % 5
        for f in range(5):
            te = [data[i] for i in idx if fold[i]==f]
            tr = [data[i] for i in idx if fold[i]!=f]
            pred = fit(tr)
            cv.append(med([pred(r)-r['lmcap'] for r in te]))
    dated = sorted([r for r in data if r['age'] is not None], key=lambda r: -r['age'])
    n = len(dated); cut = int(n*0.7)
    tr, te = dated[:cut], dated[cut:]
    pred = fit(tr)
    wf = med([pred(r)-r['lmcap'] for r in te])
    return float(np.mean(cv)), wf

def blind10(fit, data):
    ms, pcts, w30 = [], [], []
    for s in range(1000, 1010):
        tr, te = blind_split(data, seed=s, frac=0.20)
        pred = fit(tr)
        p = np.array([pred(r) for r in te]); y = np.array([r['lmcap'] for r in te])
        err = p - y
        ms.append(float(np.median(np.abs(err))))
        pct = (np.exp(p)-np.exp(y))/np.exp(y)*100.0
        pcts.append(float(np.median(pct)))
        w30.append(float(np.mean(np.abs(pct)<=30.0))*100)
    return float(np.mean(ms)), float(np.mean(pcts)), float(np.mean(w30))

print(f"{'model':22} {'CV':>6} {'WF':>6} {'blindMed':>8} {'blindMed%':>9} {'blindW30%':>9}")
results = {}
for name, fit in MODELS.items():
    cv, wf = cv_wf(fit, rows)
    bm, bp, w30 = blind10(fit, rows)
    results[name] = (cv, wf, bm, bp, w30)
    print(f"{name:22} {cv:6.4f} {wf:6.4f} {bm:8.4f} {bp:9.1f} {w30:9.1f}")
