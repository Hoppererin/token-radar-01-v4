"""Explore the token-radar snapshot: comps structure for a from-scratch model."""
import json, math, statistics
from collections import Counter, defaultdict

d = json.load(open('/home/hatch/workspace/makefun/token-radar-01-kit/input/snapshot.json'))
ps = d['protocols']
tok = [p for p in ps if p['mcap_usd'] is not None and p['mcap_usd'] > 0 and p['tvl_usd'] > 0]
print('token-bearing usable for comps:', len(tok), 'of', len(ps))
print('mcap==0:', sum(1 for p in ps if p['mcap_usd'] == 0), '| tvl<=0:', sum(1 for p in ps if p['tvl_usd'] <= 0))

# log mcap/TVL by category (only categories with >=4 comps)
bycat = defaultdict(list)
for p in tok:
    bycat[p['category']].append(math.log(p['mcap_usd'] / p['tvl_usd']))
print('\nlog(mcap/TVL) by category (n>=4): mean / sd / n')
for c, v in sorted(bycat.items(), key=lambda kv: -len(kv[1])):
    if len(v) >= 4:
        print(f'  {str(c):28s} n={len(v):3d} mean={statistics.mean(v):+.3f} sd={statistics.pstdev(v):.3f} median={statistics.median(v):+.3f}')
    else:
        print(f'  {str(c):28s} n={len(v):3d} (too few)')

# global
all_lr = [math.log(p['mcap_usd'] / p['tvl_usd']) for p in tok]
print('\nglobal log ratio mean %.3f sd %.3f' % (statistics.mean(all_lr), statistics.pstdev(all_lr)))

# correlation of log mcap with log tvl, log age, change_7d, nchains
import datetime
asof = datetime.date.fromisoformat(d['as_of'])
def age(p):
    if not p['listed_at']:
        return None
    return (asof - datetime.date.fromisoformat(p['listed_at'])).days

def corr(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float('nan')

lm = [math.log(p['mcap_usd']) for p in tok]
lt = [math.log(p['tvl_usd']) for p in tok]
print('\ncorr(log mcap, log tvl) = %.3f' % corr(lt, lm))
la = [math.log(age(p)) for p in tok if age(p) and age(p) > 0]
lma = [math.log(p['mcap_usd']) for p in tok if age(p) and age(p) > 0]
print('corr(log mcap, log age) = %.3f  (n=%d, null age=%d)' % (corr(la, lma), len(la), sum(1 for p in tok if not age(p))))
g = [(p['change_7d'], math.log(p['mcap_usd'])) for p in tok if p['change_7d'] is not None]
print('corr(log mcap, change_7d) = %.3f  (n=%d)' % (corr([x[0] for x in g], [x[1] for x in g]), len(g)))
nc = [len(p['chains']) for p in tok]
print('corr(log mcap, n_chains) = %.3f' % corr(nc, lm))
# log ratio vs log tvl (slope check), vs log age, vs change_7d
print('corr(log ratio, log tvl) = %.3f' % corr(lt, all_lr))
lr_a = [math.log(p['mcap_usd'] / p['tvl_usd']) for p in tok if age(p) and age(p) > 0]
print('corr(log ratio, log age) = %.3f' % corr(la, lr_a))
gr = [(p['change_7d'], math.log(p['mcap_usd'] / p['tvl_usd'])) for p in tok if p['change_7d'] is not None]
print('corr(log ratio, change_7d) = %.3f' % corr([x[0] for x in gr], [x[1] for x in gr]))
# change_7d distribution
ch = [p['change_7d'] for p in tok if p['change_7d'] is not None]
print('\nchange_7d: min %.1f p5 %.1f median %.1f p95 %.1f max %.1f' % (min(ch), sorted(ch)[len(ch)//20], statistics.median(ch), sorted(ch)[19*len(ch)//20], max(ch)))
ages = [age(p) for p in tok if age(p)]
print('age days: min %d median %d max %d' % (min(ages), statistics.median(ages), max(ages)))
# chain dummies: does Ethereum presence matter?
for chain in ['Ethereum', 'Solana', 'Base', 'Arbitrum']:
    on = [math.log(p['mcap_usd']/p['tvl_usd']) for p in tok if chain in p['chains']]
    off = [math.log(p['mcap_usd']/p['tvl_usd']) for p in tok if chain not in p['chains']]
    if len(on) >= 5 and len(off) >= 5:
        print(f'log ratio {chain}: on n={len(on)} mean={statistics.mean(on):+.3f} | off n={len(off)} mean={statistics.mean(off):+.3f}')
