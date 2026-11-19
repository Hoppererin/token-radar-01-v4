"""Final sweep: blend grouped-k5 with ridge-LAD; ridge lambda curve."""
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
    print('%-40s CV med %.4f   WF med %.4f' % (name, statistics.mean(cv_med), wf))
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

def fit_grouped(train, k=5.0):
    a, s = global_ols(train)
    res = {r['slug']: r['lmcap'] - a - s * r['ltvl'] for r in train}
    glob = float(np.median(list(res.values())))
    byg = {}
    for r in train:
        byg.setdefault(group_of(r['category']), []).append(res[r['slug']])
    off = {}
    for c, v in byg.items():
        nn = len(v)
        off[c] = (nn / (nn + k)) * float(np.median(v)) + (k / (nn + k)) * glob
    return lambda r: float(a + s * r['ltvl'] + off.get(group_of(r['category']), glob))

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

def blend(w):
    def fit(train):
        f1 = fit_grouped(train, k=5.0)
        f2 = fit_ridge_lad(train, lam=10.0)
        return lambda r: float(w * f1(r) + (1 - w) * f2(r))
    return fit

if __name__ == '__main__':
    evaluate(lambda tr: fit_ridge_lad(tr, lam=5.0), 'ridge-LAD lam=5', rows)
    evaluate(lambda tr: fit_ridge_lad(tr, lam=20.0), 'ridge-LAD lam=20', rows)
    evaluate(lambda tr: fit_grouped(tr, k=5.0), 'grouped k=5 (final candidate)', rows)
    for w in (0.8, 0.7, 0.6):
        evaluate(blend(w), 'blend grouped k5 w=%.1f + ridgeLAD' % w, rows)
