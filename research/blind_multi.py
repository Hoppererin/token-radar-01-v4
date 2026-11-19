"""Blind-set characterization across 10 splits (model already locked; no selection here).
Reports the distribution of median |log err| and median % err for v2 curve vs v3 final.
"""
import math, statistics
import numpy as np
from validate import load, blind_split
from experiment5 import fit_grouped, fit_ridge_lad

d, rows = load()

def final_model(train):
    f1 = fit_grouped(train, k=5.0)
    f2 = fit_ridge_lad(train, lam=10.0)
    return lambda r: float(0.8 * f1(r) + 0.2 * f2(r))

v2 = lambda r: 7.2429 + 0.5082 * r['ltvl']

for name, mk in [('v2 curve', lambda tr: v2), ('v3 final blend', final_model), ('v3 grouped k=5', lambda tr: fit_grouped(tr, k=5.0))]:
    meds, pcts, w30 = [], [], []
    for seed in [999] + list(range(1000, 1009)):
        tr, te = blind_split(rows, seed=seed, frac=0.20)
        pred = mk(tr)
        p = np.array([pred(r) for r in te]); y = np.array([r['lmcap'] for r in te])
        meds.append(float(np.median(np.abs(p - y))))
        pct = (np.exp(p) - np.exp(y)) / np.exp(y) * 100.0
        pcts.append(float(np.median(pct)))
        w30.append(float(np.mean(np.abs(pct) <= 30.0)) * 100)
    print('%-16s blind med|log|: mean %.4f median %.4f range %.4f-%.4f | med %% err mean %+.1f%% | within30 mean %.0f%%' % (
        name, statistics.mean(meds), statistics.median(meds), min(meds), max(meds), statistics.mean(pcts), statistics.mean(w30)))
