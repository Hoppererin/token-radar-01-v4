"""v3 model experiments. Metric: median |log error| (confirmed family).
Protocol: 5-fold x 5 seeds CV + walk-forward (oldest 70% -> newest 30%).
Blind set (seed 999, 20%) is frozen and only evaluated at the end.
"""
import json, math, datetime, statistics
import numpy as np
from validate import load, blind_split

d, rows = load()
print('comps n =', len(rows))

def med(pred, y):
    return float(np.median(np.abs(np.array(pred) - np.array(y))))

def mae(pred, y):
    return float(np.mean(np.abs(np.array(pred) - np.array(y))))

def rmse(pred, y):
    return float(np.sqrt(np.mean((np.array(pred) - np.array(y)) ** 2)))

# ---------- model fitters: fit(train) -> predict(row) ----------
def fit_global_ols(train):
    X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return lambda r: float(b[0] + b[1] * r['ltvl'])

def fit_global_lad(train, iters=60):
    X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        res = y - X @ b
        w = 1.0 / np.maximum(np.abs(res), 0.05)
        XtW = X.T * w
        b = np.linalg.solve(XtW @ X + 1e-6 * np.eye(2), XtW @ y)
    return lambda r: float(b[0] + b[1] * r['ltvl'])

def fit_cat_shrunk(train, k=6.0, slope=None):
    """log(mcap) = a + b*ln(TVL) + shrunk category offset on log-ratio.
    Category offset = shrunk mean of (lmcap - a - b*ltvl) within the category."""
    if slope is None:
        b = fit_global_ols(train)
        # recover slope/intercept by fitting on train
        X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
        bb = np.linalg.lstsq(X, y, rcond=None)[0]
        a, s = float(bb[0]), float(bb[1])
    else:
        a, s = 0.0, slope
    res = {r['slug']: r['lmcap'] - a - s * r['ltvl'] for r in train}
    bycat = {}
    for r in train:
        bycat.setdefault(r['category'], []).append(res[r['slug']])
    off = {}
    for c, v in bycat.items():
        n = len(v)
        off[c] = (n / (n + k)) * float(np.mean(v))
    return lambda r: float(a + s * r['ltvl'] + off.get(r['category'], 0.0))

def fit_cat_med_shrunk(train, k=6.0):
    """log-ratio median per category, shrunk toward the global median; plus global slope."""
    X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
    bb = np.linalg.lstsq(X, y, rcond=None)[0]
    a, s = float(bb[0]), float(bb[1])
    res = {r['slug']: r['lmcap'] - a - s * r['ltvl'] for r in train}
    glob = float(np.median(list(res.values())))
    bycat = {}
    for r in train:
        bycat.setdefault(r['category'], []).append(res[r['slug']])
    off = {}
    for c, v in bycat.items():
        n = len(v)
        off[c] = (n / (n + k)) * float(np.median(v)) + (k / (n + k)) * glob
    return lambda r: float(a + s * r['ltvl'] + off.get(r['category'], glob))

def fit_trimmed_ols(train, cut=2.5):
    X = np.array([[1.0, r['ltvl']] for r in train]); y = np.array([r['lmcap'] for r in train])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    sd = float(np.std(res)) or 1.0
    keep = np.abs(res) <= cut * sd
    if keep.sum() >= 10:
        b = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    return lambda r: float(b[0] + b[1] * r['ltvl'])

def fit_knn(train, k=8):
    # distance on standardized ln TVL + growth + log age; same-category bonus
    lt = np.array([r['ltvl'] for r in train]); mu, sd = lt.mean(), lt.std() or 1.0
    ch = np.array([r['change_7d'] if r['change_7d'] is not None else 0.0 for r in train])
    cmu, csd = ch.mean(), ch.std() or 1.0
    la = np.array([math.log(max(r['age'], 30)) if r['age'] else math.log(365.0) for r in train])
    lmu, lsd = la.mean(), la.std() or 1.0
    feats = np.column_stack([(lt - mu) / sd, (ch - cmu) / csd, (la - lmu) / lsd])
    y = np.array([r['lmcap'] for r in train])
    cats = [r['category'] for r in train]
    def predict(r):
        age_log = math.log(max(r['age'], 30)) if r['age'] else math.log(365.0)
        chg = r['change_7d'] if r['change_7d'] is not None else 0.0
        f = np.array([(r['ltvl'] - mu) / sd, (chg - cmu) / csd, (age_log - lmu) / lsd])
        dist = np.sqrt(np.sum((feats - f) ** 2, axis=1))
        dist = dist + np.array([0.0 if c == r['category'] else 0.75 for c in cats])
        idx = np.argsort(dist)[:k]
        w = 1.0 / (dist[idx] + 0.1)
        return float(np.sum(w * y[idx]) / np.sum(w))
    return predict

def fit_ensemble(train):
    f1 = fit_global_lad(train)
    f2 = fit_cat_med_shrunk(train, k=6.0)
    f3 = fit_knn(train, k=8)
    return lambda r: float((f1(r) + f2(r) + f3(r)) / 3.0)

def evaluate(fit, name, rows, seeds=(0, 1, 2, 3, 4)):
    cv_med, cv_mae, cv_rm = [], [], []
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
            cv_med.append(med(p, yy)); cv_mae.append(mae(p, yy)); cv_rm.append(rmse(p, yy))
    dated = sorted([r for r in rows if r['age'] is not None], key=lambda r: -r['age'])
    cut = int(len(dated) * 0.70)
    tr, te = dated[:cut], dated[cut:]
    pred = fit(tr)
    p = [pred(r) for r in te]; yy = [r['lmcap'] for r in te]
    wf = med(p, yy)
    print('%-28s CV med %.4f (mae %.4f rmse %.4f)   WF med %.4f' % (name, statistics.mean(cv_med), statistics.mean(cv_mae), statistics.mean(cv_rm), wf))
    return statistics.mean(cv_med), wf

if __name__ == '__main__':
    evaluate(fit_global_ols, 'A global OLS', rows)
    evaluate(fit_global_lad, 'B global LAD', rows)
    evaluate(lambda tr: fit_cat_shrunk(tr, k=4.0), 'C cat-shrunk mean k=4', rows)
    evaluate(lambda tr: fit_cat_shrunk(tr, k=8.0), 'C cat-shrunk mean k=8', rows)
    evaluate(lambda tr: fit_cat_med_shrunk(tr, k=4.0), 'D cat-med-shrunk k=4', rows)
    evaluate(lambda tr: fit_cat_med_shrunk(tr, k=8.0), 'D cat-med-shrunk k=8', rows)
    evaluate(fit_trimmed_ols, 'E trimmed OLS 2.5sd', rows)
    evaluate(lambda tr: fit_knn(tr, k=8), 'F kNN k=8', rows)
    evaluate(lambda tr: fit_knn(tr, k=5), 'F kNN k=5', rows)
    evaluate(fit_ensemble, 'G ensemble LAD+catmed+kNN', rows)
