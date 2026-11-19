"""Pin down the v1/v2 metric: test the house starter formula and global OLS under CV/WF."""
import json, math, datetime, statistics
import numpy as np
from validate import load, cv_score, wf_score, blind_split, ols, rmse

d, rows = load()
print('comps n =', len(rows))

def starter_implied(r):
    # house starter formula (from kit's starter lib/model.ts, read as kit material)
    growth = max(-60.0, min(120.0, r['change_7d'] if r['change_7d'] is not None else 0.0))
    gm = 1 + (growth / 100.0) * 0.5
    age = r['age'] if r['age'] is not None else 365
    mm = 0.6 + (min(max(age, 0), 1095) / 1095.0) * 0.4
    return max(1.0, round(r['tvl'] * 0.25 * gm * mm))

def med_abs_log_err(pred, y):
    return float(np.median(np.abs(np.array(pred) - np.array(y))))

def log_mae(pred, y):
    return float(np.mean(np.abs(np.array(pred) - np.array(y))))

def metrics_for_predictor(predict, name):
    # in-sample
    pred = [predict(r) for r in rows]
    y = [r['lmcap'] for r in rows]
    print(f'{name} in-sample: log-RMSE {rmse(pred,y):.4f} log-MAE {log_mae(pred,y):.4f} med|log| {med_abs_log_err(pred,y):.4f}')
    # CV via generic folds using the same predictor family is hard for fixed formulas;
    # do manual 5-fold with seeds: for fixed formula, CV == in-sample on each fold's test
    rng = np.random.RandomState(0)
    idx = np.arange(len(rows))
    bycat = {}
    for i, r in enumerate(rows):
        bycat.setdefault(r['category'], []).append(i)
    fold_of = np.zeros(len(rows), dtype=int)
    for c, members in bycat.items():
        m = np.array(members); rng.shuffle(m)
        for j, i in enumerate(m):
            fold_of[i] = j % 5
    maes, meds, rmses = [], [], []
    for f in range(5):
        te = [rows[i] for i in idx if fold_of[i] == f]
        p = [predict(r) for r in te]; yy = [r['lmcap'] for r in te]
        maes.append(log_mae(p, yy)); meds.append(med_abs_log_err(p, yy)); rmses.append(rmse(p, yy))
    print(f'{name} 5-fold: log-RMSE {statistics.mean(rmses):.4f} log-MAE {statistics.mean(maes):.4f} med|log| {statistics.mean(meds):.4f}')
    # walk-forward: oldest 70% train (n/a for fixed formula), test newest 30%
    dated = sorted([r for r in rows if r['age'] is not None], key=lambda r: -r['age'])
    cut = int(len(dated) * 0.70)
    te = dated[cut:]
    p = [predict(r) for r in te]; yy = [r['lmcap'] for r in te]
    print(f'{name} WF newest-30%: log-RMSE {rmse(p,yy):.4f} log-MAE {log_mae(p,yy):.4f} med|log| {med_abs_log_err(p,yy):.4f}')

metrics_for_predictor(lambda r: math.log(starter_implied(r)), 'starter formula (v1?)')
metrics_for_predictor(lambda r: 7.2429 + 0.5082 * r['ltvl'], 'v2 curve (brief)')
# global OLS refit per fold: use validate.cv_score machinery but with med/mae too
# quick manual: reuse cv_score? it only returns rmse. Do it inline:
def cv_metrics(build, seeds=(0,1,2,3,4)):
    maes, meds, rmses = [], [], []
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
            cats = sorted({r['category'] for r in tr})
            Xtr = [build(r, cats) for r in tr]; ytr = [r['lmcap'] for r in tr]
            beta = ols(Xtr, ytr)
            p = [float(np.dot(beta, build(r, cats))) for r in te]; yy = [r['lmcap'] for r in te]
            maes.append(log_mae(p, yy)); meds.append(med_abs_log_err(p, yy)); rmses.append(rmse(p, yy))
    return statistics.mean(rmses), statistics.mean(maes), statistics.mean(meds)

r, m, md = cv_metrics(lambda r, cats: [1.0, r['ltvl']])
print('global OLS refit 5-fold x5 seeds: log-RMSE %.4f log-MAE %.4f med|log| %.4f' % (r, m, md))
