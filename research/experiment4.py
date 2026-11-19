"""v3 experiments round 4: LAD-global D, k sweep, ensembles, heteroscedasticity check."""
import math, statistics
import numpy as np
from validate import load
from experiment3 import evaluate, global_ols, group_of

d, rows = load()

# heteroscedasticity: residual sd of the v2 curve by TVL tercile
res = [(r, 7.2429 + 0.5082 * r['ltvl'] - r['lmcap']) for r in rows]
res.sort(key=lambda x: x[0]['tvl'])
n = len(res)
for i, name in enumerate(['small TVL', 'mid TVL', 'big TVL']):
    part = res[i*n//3:(i+1)*n//3]
    vals = [e for _, e in part]
    print('%s n=%d resid sd %.3f med|err| %.3f' % (name, len(part), float(np.std(vals)), float(np.median(np.abs(vals)))))

def lad_global(train, iters=60):
    X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ b
        w = 1.0 / np.maximum(np.abs(r), 0.05)
        XtW = X.T * w
        b = np.linalg.solve(XtW @ X + 1e-6 * np.eye(2), XtW @ y)
    return float(b[0]), float(b[1])

def fit_d(train, k=4.0, use_group=True, lad=False):
    a, s = lad_global(train) if lad else global_ols(train)
    key = (lambda r: group_of(r['category'])) if use_group else (lambda r: r['category'])
    res = {r['slug']: r['lmcap'] - a - s * r['ltvl'] for r in train}
    glob = float(np.median(list(res.values())))
    byg = {}
    for r in train:
        byg.setdefault(key(r), []).append(res[r['slug']])
    off = {}
    for c, v in byg.items():
        nn = len(v)
        off[c] = (nn / (nn + k)) * float(np.median(v)) + (k / (nn + k)) * glob
    return lambda r: float(a + s * r['ltvl'] + off.get(key(r), glob))

def fit_ridge_lad(train, lam=10.0):
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
    pen = np.eye(p) * lam; pen[0, 0] = 0.0; pen[1, 1] = 0.0
    b = np.linalg.solve(X.T @ X + pen, X.T @ y)
    for _ in range(40):
        rr = y - X @ b
        w = 1.0 / np.maximum(np.abs(rr), 0.05)
        XtW = X.T * w
        b = np.linalg.solve(XtW @ X + pen, XtW @ y)
    return lambda r: float(np.dot(b, feats(r)))

print('--- D variants ---')
evaluate(lambda tr: fit_d(tr, k=4.0, use_group=True, lad=False), 'grouped OLS k=4', rows)
evaluate(lambda tr: fit_d(tr, k=4.0, use_group=True, lad=True), 'grouped LAD k=4', rows)
evaluate(lambda tr: fit_d(tr, k=3.0, use_group=True), 'grouped OLS k=3', rows)
evaluate(lambda tr: fit_d(tr, k=5.0, use_group=True), 'grouped OLS k=5', rows)
evaluate(lambda tr: fit_d(tr, k=4.0, use_group=False, lad=True), 'catmed LAD k=4', rows)
print('--- ensembles ---')
def fit_ens1(train):
    f1 = fit_d(train, k=4.0, use_group=True)
    f2 = fit_ridge_lad(train, lam=10.0)
    return lambda r: float(0.5 * f1(r) + 0.5 * f2(r))
def fit_ens2(train):
    f1 = fit_d(train, k=4.0, use_group=True)
    f2 = fit_ridge_lad(train, lam=10.0)
    f3 = fit_d(train, k=4.0, use_group=False)
    return lambda r: float((f1(r) + f2(r) + f3(r)) / 3.0)
evaluate(fit_ens1, 'ens grouped + ridgeLAD 50/50', rows)
evaluate(fit_ens2, 'ens grouped + ridgeLAD + catmed', rows)
