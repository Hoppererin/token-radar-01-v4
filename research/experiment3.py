"""v3 experiments round 3: category merging, bagging, core comps set, ridge-LAD."""
import math, statistics
import numpy as np
from validate import load

d, rows = load()

def med(pred, y):
    return float(np.median(np.abs(np.array(pred) - np.array(y))))

def evaluate(fit, name, data, seeds=(0, 1, 2, 3, 4)):
    cv_med = []
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = np.arange(len(data))
        bycat = {}
        for i, r in enumerate(data):
            bycat.setdefault(r['category'], []).append(i)
        fold_of = np.zeros(len(data), dtype=int)
        for c, members in bycat.items():
            m = np.array(members); rng.shuffle(m)
            for j, i in enumerate(m):
                fold_of[i] = j % 5
        for f in range(5):
            te = [data[i] for i in idx if fold_of[i] == f]
            tr = [data[i] for i in idx if fold_of[i] != f]
            pred = fit(tr)
            p = [pred(r) for r in te]; yy = [r['lmcap'] for r in te]
            cv_med.append(med(p, yy))
    dated = sorted([r for r in data if r['age'] is not None], key=lambda r: -r['age'])
    cut = int(len(dated) * 0.70)
    tr, te = dated[:cut], dated[cut:]
    pred = fit(tr)
    p = [pred(r) for r in te]; yy = [r['lmcap'] for r in te]
    wf = med(p, yy)
    print('%-36s n=%3d CV med %.4f   WF med %.4f' % (name, len(data), statistics.mean(cv_med), wf))
    return statistics.mean(cv_med), wf

def global_ols(train):
    X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(b[0]), float(b[1])

GROUP = {
    'Bridge': 'Bridge', 'Canonical Bridge': 'Bridge', 'Cross Chain Bridge': 'Bridge',
    'Liquid Staking': 'Staking', 'Liquid Restaking': 'Staking', 'Staking Pool': 'Staking', 'Restaking': 'Staking',
    'Yield': 'Yield', 'Yield Aggregator': 'Yield', 'Leveraged Farming': 'Yield',
    'Lending': 'Lending', 'CDP': 'Lending', 'Collateral Management': 'Lending',
    'Dexs': 'Dexs', 'Liquidity Manager': 'Dexs',
}
def group_of(cat):
    return GROUP.get(cat, cat)

def fit_grouped_med(train, k=4.0, use_group=True):
    a, s = global_ols(train)
    key = (lambda r: group_of(r['category'])) if use_group else (lambda r: r['category'])
    res = {r['slug']: r['lmcap'] - a - s * r['ltvl'] for r in train}
    glob = float(np.median(list(res.values())))
    byg = {}
    for r in train:
        byg.setdefault(key(r), []).append(res[r['slug']])
    off = {}
    for c, v in byg.items():
        n = len(v)
        off[c] = (n / (n + k)) * float(np.median(v)) + (k / (n + k)) * glob
    return lambda r: float(a + s * r['ltvl'] + off.get(key(r), glob))

def fit_bagged_med(train, k=4.0, bags=25, seed=7):
    rng = np.random.RandomState(seed)
    preds = []
    n = len(train)
    for _ in range(bags):
        idx = rng.randint(0, n, n)
        sample = [train[i] for i in idx]
        preds.append(fit_grouped_med(sample, k=k, use_group=False))
    return lambda r: float(np.median([p(r) for p in preds]))

def fit_bagged_grouped(train, k=4.0, bags=25, seed=7):
    rng = np.random.RandomState(seed)
    preds = []
    n = len(train)
    for _ in range(bags):
        idx = rng.randint(0, n, n)
        sample = [train[i] for i in idx]
        preds.append(fit_grouped_med(sample, k=k, use_group=True))
    return lambda r: float(np.median([p(r) for p in preds]))

def ridge_lad(train, lam=10.0):
    # features: [1, ltvl, cat dummies (all cats), change/100, log-age, nchains, chain dummies]
    cats = sorted({r['category'] for r in train})
    chains = ['Ethereum', 'Solana', 'Base', 'Arbitrum', 'BNB Chain']
    def feats(r):
        v = [1.0, r['ltvl']]
        v += [1.0 if r['category'] == c else 0.0 for c in cats]
        v.append(max(-60.0, min(120.0, r['change_7d'] if r['change_7d'] is not None else 0.0)) / 100.0)
        v.append(math.log(max(r['age'], 30)) if r['age'] else math.log(365.0))
        v.append(float(len(r['chains'])))
        v += [1.0 if c in r['chains'] else 0.0 for c in chains]
        return v
    X = np.array([feats(r) for r in train]); y = np.array([r['lmcap'] for r in train])
    p = X.shape[1]
    pen = np.eye(p) * lam
    pen[0, 0] = 0.0; pen[1, 1] = 0.0
    b = np.linalg.solve(X.T @ X + pen, X.T @ y)
    for _ in range(40):
        res = y - X @ b
        w = 1.0 / np.maximum(np.abs(res), 0.05)
        XtW = X.T * w
        b = np.linalg.solve(XtW @ X + pen, XtW @ y)
    return lambda r: float(np.dot(b, feats(r)))

EX = {'CEX', 'Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC'}
core = [r for r in rows if r['category'] not in EX]
print('--- full comps (n=128) ---')
evaluate(lambda tr: fit_grouped_med(tr, k=4.0, use_group=True), 'grouped catmed k=4', rows)
evaluate(lambda tr: fit_grouped_med(tr, k=2.0, use_group=True), 'grouped catmed k=2', rows)
evaluate(lambda tr: fit_bagged_med(tr, k=4.0), 'bagged catmed k=4', rows)
evaluate(lambda tr: fit_bagged_grouped(tr, k=4.0), 'bagged grouped k=4', rows)
evaluate(lambda tr: ridge_lad(tr, lam=10.0), 'ridge-LAD lam=10', rows)
evaluate(lambda tr: ridge_lad(tr, lam=30.0), 'ridge-LAD lam=30', rows)
print('--- core comps: no CEX/bridge/staking (n=%d) ---' % len(core))
evaluate(lambda tr: (lambda r: 7.2429 + 0.5082 * r['ltvl']), 'v2 curve (fixed)', core)
evaluate(lambda tr: fit_grouped_med(tr, k=4.0, use_group=False), 'catmed k=4', core)
evaluate(lambda tr: fit_grouped_med(tr, k=4.0, use_group=True), 'grouped catmed k=4', core)
evaluate(lambda tr: fit_bagged_grouped(tr, k=4.0), 'bagged grouped k=4', core)
