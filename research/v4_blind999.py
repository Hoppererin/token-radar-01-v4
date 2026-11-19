"""Seed-999 blind split (the exact protocol v3 reported) for top candidates."""
import math
import numpy as np
from validate import load, blind_split
from v4_sweep import fit_grouped, fit_ridge_lad, blend

d, rows = load()
tr, te = blind_split(rows, seed=999, frac=0.20)
print('blind split: train', len(tr), 'test', len(te))

def report(name, pred_fn):
    p = np.array([pred_fn(r) for r in te]); y = np.array([r['lmcap'] for r in te])
    err = p - y
    med_log = float(np.median(np.abs(err)))
    pct = (np.exp(p)-np.exp(y))/np.exp(y)*100.0
    med_pct = float(np.median(pct))
    w30 = float(np.mean(np.abs(pct)<=30.0))*100
    print(f"{name:20} med|log| {med_log:.4f}  median% {med_pct:6.1f}  within30 {w30:5.1f}%")

report('v2 curve', __import__('v4_sweep').fit_global(tr))
report('grouped k5', fit_grouped(tr, 5.0))
report('grouped k8', fit_grouped(tr, 8.0))
report('blend 0.9/0.1', blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 0.9))
report('blend 0.8/0.2 (v3)', blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 0.8))
report('blend 0.7/0.3', blend(fit_grouped(tr,5.0), fit_ridge_lad(tr,10.0), 0.7))
