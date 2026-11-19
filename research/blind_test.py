"""Blind-set evaluation (frozen split, seed 999, 20% stratified). Run ONCE for the final model.
Final model: 0.8 * grouped-median comps (k=5) + 0.2 * ridge-LAD (lam=10).
Benchmark on the same split: the v2 curve from the brief.
"""
import math, statistics
import numpy as np
from validate import load, blind_split
from experiment5 import fit_grouped, fit_ridge_lad

d, rows = load()
tr, te = blind_split(rows, seed=999, frac=0.20)
print('blind split: train %d test %d' % (len(tr), len(te)))

def final_model(train):
    f1 = fit_grouped(train, k=5.0)
    f2 = fit_ridge_lad(train, lam=10.0)
    return lambda r: float(0.8 * f1(r) + 0.2 * f2(r))

def report(name, pred_fn, test):
    p = np.array([pred_fn(r) for r in test])
    y = np.array([r['lmcap'] for r in test])
    err = p - y
    med_log = float(np.median(np.abs(err)))
    mae_log = float(np.mean(np.abs(err)))
    rmse_log = float(np.sqrt(np.mean(err ** 2)))
    # level-space errors
    pct = (np.exp(p) - np.exp(y)) / np.exp(y) * 100.0
    med_pct = float(np.median(pct))
    med_abs_pct = float(np.median(np.abs(pct)))
    within30 = float(np.mean(np.abs(pct) <= 30.0)) * 100
    within2x = float(np.mean(np.abs(err) <= math.log(2))) * 100
    print('%s:' % name)
    print('  median |log err| %.4f | log-MAE %.4f | log-RMSE %.4f' % (med_log, mae_log, rmse_log))
    print('  level: median %% err %+.1f%% | median |%% err| %.1f%% | within +-30%%: %.0f%% | within 2x: %.0f%%' % (med_pct, med_abs_pct, within30, within2x))
    return med_log

v2 = lambda r: 7.2429 + 0.5082 * r['ltvl']
report('v2 curve (brief)', v2, te)
report('v3 final (blend 0.8/0.2)', final_model(tr), te)
# also the grouped-only finalist for reference
report('v3 grouped k=5 only', fit_grouped(tr, k=5.0), te)
