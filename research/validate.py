"""Validation harness for token-radar v3 model research (from scratch).

Metric (log-space RMSE on ln(mcap), the same scale as the v2 numbers in the brief):
  CV            : repeated 5-fold CV (5 seeds), stratified by category bucket
  walk-forward  : train on oldest 70% by listed_at, test on newest 30%
  blind         : one held-out 20% stratified split, fixed seed, never used for tuning
"""
import json, math, datetime, statistics
import numpy as np

SNAP = '/home/hatch/workspace/makefun/token-radar-01-kit/input/snapshot.json'

def load():
    d = json.load(open(SNAP))
    asof = datetime.date.fromisoformat(d['as_of'])
    rows = []
    for p in d['protocols']:
        if p['mcap_usd'] is None or p['mcap_usd'] <= 0 or p['tvl_usd'] <= 0:
            continue
        age = None
        if p['listed_at']:
            age = (asof - datetime.date.fromisoformat(p['listed_at'])).days
        rows.append({
            'slug': p['slug'], 'name': p['name'], 'category': p['category'] or 'Unknown',
            'chains': p['chains'], 'tvl': p['tvl_usd'], 'mcap': p['mcap_usd'],
            'change_7d': p['change_7d'], 'age': age,
            'ltvl': math.log(p['tvl_usd']), 'lmcap': math.log(p['mcap_usd']),
        })
    return d, rows

# ---------------- feature builders ----------------
def feat_global(r):
    return [1.0, r['ltvl']]

def feat_cat(r, cats):
    # category intercept (one-hot over known cats) + global slope
    v = [1.0, r['ltvl']]
    v += [1.0 if r['category'] == c else 0.0 for c in cats]
    return v

def feat_full(r, cats, chains):
    # category intercepts + slope + growth + log-age + n-chains + chain dummies
    v = [1.0, r['ltvl']]
    v += [1.0 if r['category'] == c else 0.0 for c in cats]
    ch = r['change_7d'] if r['change_7d'] is not None else 0.0
    v.append(max(-60.0, min(120.0, ch)) / 100.0)
    v.append(math.log(max(r['age'], 30)) if r['age'] else math.log(365.0))
    v.append(float(len(r['chains'])))
    v += [1.0 if c in r['chains'] else 0.0 for c in chains]
    return v

def ols(X, y, ridge=0.0):
    X = np.array(X, dtype=float); y = np.array(y, dtype=float)
    if ridge > 0:
        A = X.T @ X + ridge * np.eye(X.shape[1])
        A[0, 0] -= ridge  # do not penalise the intercept
        return np.linalg.solve(A, X.T @ y)
    return np.linalg.lstsq(X, y, rcond=None)[0]

def rmse(pred, y):
    return float(np.sqrt(np.mean((np.array(pred) - np.array(y)) ** 2)))

def eval_split(train, test, build, ridge=0.0):
    cats = sorted({r['category'] for r in train})
    chains = ['Ethereum', 'Solana', 'Base', 'Arbitrum', 'BNB Chain']
    Xtr = [build(r, cats, chains) for r in train]
    ytr = [r['lmcap'] for r in train]
    beta = ols(Xtr, ytr, ridge)
    Xte = [build(r, cats, chains) for r in test]
    pred = [float(np.dot(beta, x)) for x in Xte]
    return rmse(pred, [r['lmcap'] for r in test]), beta

def cv_score(rows, build, ridge=0.0, folds=5, seeds=(0, 1, 2, 3, 4)):
    rng_master = np.random.RandomState(12345)
    scores = []
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = np.arange(len(rows))
        # stratify by category bucket: shuffle within each category, then round-robin into folds
        bycat = {}
        for i, r in enumerate(rows):
            bycat.setdefault(r['category'], []).append(i)
        fold_of = np.zeros(len(rows), dtype=int)
        for c, members in bycat.items():
            m = np.array(members); rng.shuffle(m)
            for j, i in enumerate(m):
                fold_of[i] = j % folds
        for f in range(folds):
            te = [rows[i] for i in idx if fold_of[i] == f]
            tr = [rows[i] for i in idx if fold_of[i] != f]
            sc, _ = eval_split(tr, te, build, ridge)
            scores.append(sc)
    return statistics.mean(scores), statistics.pstdev(scores)

def wf_score(rows, build, ridge=0.0, frac=0.70):
    dated = [r for r in rows if r['age'] is not None]
    dated.sort(key=lambda r: r['age'], reverse=True)  # oldest first
    cut = int(len(dated) * frac)
    tr, te = dated[:cut], dated[cut:]
    sc, _ = eval_split(tr, te, build, ridge)
    return sc, len(tr), len(te)

def blind_split(rows, seed=999, frac=0.20):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(rows))
    bycat = {}
    for i, r in enumerate(rows):
        bycat.setdefault(r['category'], []).append(i)
    test_idx = []
    for c, members in bycat.items():
        m = np.array(members); rng.shuffle(m)
        k = max(1, int(round(len(m) * frac))) if len(m) >= 5 else 0
        test_idx.extend(m[:k].tolist())
    te = [rows[i] for i in test_idx]
    tr = [rows[i] for i in idx if i not in set(test_idx)]
    return tr, te

if __name__ == '__main__':
    d, rows = load()
    print('comps n =', len(rows))
    # reference: the exact v2 curve from the brief (benchmark only, not our model)
    def v2_curve(r):
        return 7.2429 + 0.5082 * r['ltvl']
    print('v2 curve in-sample log-RMSE: %.4f' % rmse([v2_curve(r) for r in rows], [r['lmcap'] for r in rows]))
    # v2 curve under CV protocol (refit OLS each fold = same family)
    m, s = cv_score(rows, lambda r, cats, chains: feat_global(r))
    print('global OLS ln(mcap)~ln(tvl)   CV: %.4f +- %.4f' % (m, s))
    wf, ntr, nte = wf_score(rows, lambda r, cats, chains: feat_global(r))
    print('global OLS                    WF: %.4f (train %d test %d)' % (wf, ntr, nte))
    # category intercepts
    m, s = cv_score(rows, lambda r, cats, chains: feat_cat(r, cats))
    print('category intercepts + slope   CV: %.4f +- %.4f' % (m, s))
    wf, ntr, nte = wf_score(rows, lambda r, cats, chains: feat_cat(r, cats))
    print('category intercepts + slope   WF: %.4f' % wf)
    # full features, ridge sweep
    for ridge in (0.0, 1.0, 3.0, 10.0, 30.0):
        m, s = cv_score(rows, lambda r, cats, chains: feat_full(r, cats, chains), ridge=ridge)
        wf, _, _ = wf_score(rows, lambda r, cats, chains: feat_full(r, cats, chains), ridge=ridge)
        print('full feats ridge=%-5s          CV: %.4f +- %.4f  WF: %.4f' % (ridge, m, s, wf))
