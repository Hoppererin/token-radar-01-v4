"""v3 model experiments round 2: richer median models.
Base = D: log(mcap) = a + b*lnTVL + shrunk category-median residual offset.
Extensions: growth term, chain offsets, (lnTVL)^2, n_chains, per-category slope.
Also: ridge-LAD with full features.
"""
import math, statistics
import numpy as np
from validate import load

d, rows = load()

def med(pred, y):
    return float(np.median(np.abs(np.array(pred) - np.array(y))))

def evaluate(fit, name, rows, seeds=(0, 1, 2, 3, 4)):
    cv_med = []
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = np.arange(len(rows))
        bycat = {}
        for i, r in enumerate(rows):
            bycat.setdefault(r['category'], []).append(i)
        fold_of = np.zeros(len(rows), dtype=int)
        for c, members in bycat.items():
            m = np.array(members); rng.shuffle(m)
            for j, i in enumerate(m):
                fold_of[i] = j % 5
        for f in range(5):
            te = [rows[i] for i in idx if fold_of[i] == f]
            tr = [rows[i] for i in idx if fold_of[i] != f]
            pred = fit(tr)
            p = [pred(r) for r in te]; yy = [r['lmcap'] for r in te]
            cv_med.append(med(p, yy))
    dated = sorted([r for r in rows if r['age'] is not None], key=lambda r: -r['age'])
    cut = int(len(dated) * 0.70)
    tr, te = dated[:cut], dated[cut:]
    pred = fit(tr)
    p = [pred(r) for r in te]; yy = [r['lmcap'] for r in te]
    wf = med(p, yy)
    print('%-34s CV med %.4f   WF med %.4f' % (name, statistics.mean(cv_med), wf))
    return statistics.mean(cv_med), wf

def global_ols(train):
    X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(b[0]), float(b[1])

CHAINS = ['Ethereum', 'Solana', 'Base', 'Arbitrum', 'BNB Chain']

def base_parts(train, k=4.0):
    a, s = global_ols(train)
    res = {r['slug']: r['lmcap'] - a - s * r['ltvl'] for r in train}
    glob = float(np.median(list(res.values())))
    bycat = {}
    for r in train:
        bycat.setdefault(r['category'], []).append(res[r['slug']])
    off = {}
    for c, v in bycat.items():
        n = len(v)
        off[c] = (n / (n + k)) * float(np.median(v)) + (k / (n + k)) * glob
    return a, s, glob, off, res

def make_catmed(k=4.0, growth=0.0, chains=False, quad=0.0, nchains=0.0, cat_slope=0.0):
    def fit(train):
        a, s, glob, off, res = base_parts(train, k)
        # extra terms fit on residuals (median regression via simple LAD on one feature at a time)
        # growth: residual ~ g * clamp(change_7d)/100
        g_coef = 0.0
        if growth:
            xs = np.array([max(-60.0, min(120.0, r['change_7d'] if r['change_7d'] is not None else 0.0)) / 100.0 for r in train])
            ys = np.array([res[r['slug']] - off.get(r['category'], glob) for r in train])
            g_coef = lad_1d(xs, ys)
        # chain offsets: shrunk median residual per chain
        chain_off = {}
        if chains:
            res2 = {r['slug']: res[r['slug']] - off.get(r['category'], glob) - g_coef * (max(-60.0, min(120.0, r['change_7d'] if r['change_7d'] is not None else 0.0)) / 100.0) for r in train}
            glob2 = float(np.median(list(res2.values())))
            for c in CHAINS:
                v = [res2[r['slug']] for r in train if c in r['chains']]
                n = len(v)
                if n >= 3:
                    chain_off[c] = (n / (n + 6.0)) * float(np.median(v)) + (6.0 / (n + 6.0)) * glob2
                else:
                    chain_off[c] = glob2
            chain_glob = glob2
        else:
            chain_glob = 0.0
        # quadratic in lnTVL (centered)
        lt_mean = float(np.mean([r['ltvl'] for r in train]))
        q_coef = 0.0
        if quad:
            xs = np.array([(r['ltvl'] - lt_mean) ** 2 for r in train])
            ys = np.array([res[r['slug']] - off.get(r['category'], glob) for r in train])
            q_coef = lad_1d(xs, ys)
        nc_coef = 0.0
        if nchains:
            xs = np.array([float(len(r['chains'])) for r in train])
            ys = np.array([res[r['slug']] - off.get(r['category'], glob) for r in train])
            nc_coef = lad_1d(xs, ys)
        # per-category slope: shrunk slope of residual on (ltvl - cat mean ltvl)
        cs = {}
        if cat_slope:
            bycat = {}
            for r in train:
                bycat.setdefault(r['category'], []).append(r)
            for c, members in bycat.items():
                if len(members) >= 6:
                    xs = np.array([r['ltvl'] for r in members]); ys = np.array([res[r['slug']] for r in members])
                    xm = xs.mean()
                    num = float(np.sum((xs - xm) * (ys - ys.mean())))
                    den = float(np.sum((xs - xm) ** 2)) or 1.0
                    raw = num / den
                    n = len(members)
                    cs[c] = (n / (n + 8.0)) * raw
            cat_lt_mean = {c: float(np.mean([r['ltvl'] for r in m])) for c, m in bycat.items()}
        else:
            cat_lt_mean = {}
        def predict(r):
            chg = max(-60.0, min(120.0, r['change_7d'] if r['change_7d'] is not None else 0.0)) / 100.0
            v = a + s * r['ltvl'] + off.get(r['category'], glob)
            v += g_coef * chg
            if chains:
                v += sum(chain_off.get(c, chain_glob) for c in CHAINS if c in r['chains'])
                v -= chain_glob * 0  # offsets already shrunk toward glob2; keep as-is
            if quad:
                v += q_coef * (r['ltvl'] - lt_mean) ** 2
            if nchains:
                v += nc_coef * float(len(r['chains']))
            if cat_slope and r['category'] in cs:
                v += cs[r['category']] * (r['ltvl'] - cat_lt_mean[r['category']])
            return float(v)
        return predict
    return fit

def lad_1d(xs, ys, iters=50):
    """LAD slope for y ~ b*x (no intercept)."""
    b = float(np.sum(xs * ys) / (np.sum(xs * xs) or 1.0))
    for _ in range(iters):
        res = ys - b * xs
        w = 1.0 / np.maximum(np.abs(res), 0.05)
        b = float(np.sum(w * xs * ys) / (np.sum(w * xs * xs) or 1.0))
    return b

if __name__ == '__main__':
    evaluate(make_catmed(k=4.0), 'D catmed k=4 (base)', rows)
    evaluate(make_catmed(k=4.0, growth=1), 'D + growth', rows)
    evaluate(make_catmed(k=4.0, chains=True), 'D + chain offsets', rows)
    evaluate(make_catmed(k=4.0, growth=1, chains=True), 'D + growth + chains', rows)
    evaluate(make_catmed(k=4.0, quad=1), 'D + (lnTVL)^2', rows)
    evaluate(make_catmed(k=4.0, nchains=1), 'D + n_chains', rows)
    evaluate(make_catmed(k=4.0, cat_slope=1), 'D + per-cat slope', rows)
    evaluate(make_catmed(k=4.0, growth=1, chains=True, quad=1, nchains=1, cat_slope=1), 'D + all', rows)
    evaluate(make_catmed(k=2.0, growth=1, chains=True), 'D k=2 + growth + chains', rows)
    evaluate(make_catmed(k=6.0, growth=1, chains=True), 'D k=6 + growth + chains', rows)
