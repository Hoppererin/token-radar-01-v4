"""Gate-style checks for token-radar-01-v3 (mirrors RULES.txt gate rows, minus dossier)."""
import json, os, re, hashlib
from PIL import Image

WORK = '/home/hatch/workspace/makefun/token-radar-01-v3'
OUT = '/tmp/token-radar-v3-out'
snap = json.load(open('/home/hatch/workspace/makefun/token-radar-01-kit/input/snapshot.json'))
board = json.load(open(OUT + '/board.json'))['rows']
slugs = {p['slug'] for p in snap['protocols']}
fails = []

# board_file
print('rows:', len(board))
if not (100 <= len(board) <= 4000): fails.append('row count')
seen = set()
for r in board:
    if r['slug'] not in slugs: fails.append('unknown slug ' + r['slug'])
    if r['slug'] in seen: fails.append('dup slug ' + r['slug'])
    seen.add(r['slug'])
    if not isinstance(r['implied_value_usd'], int) or r['implied_value_usd'] <= 0: fails.append('bad implied ' + r['slug'])
    if not (3 <= len(r['farm_steps']) <= 12): fails.append('farm count ' + r['slug'])
    for s in r['farm_steps']:
        if not (20 <= len(s.strip()) <= 600): fails.append('farm len %d %s' % (len(s), r['slug']))
# sorted_by_value
ranks = sorted(r['rank'] for r in board)
if ranks != list(range(1, len(board) + 1)): fails.append('ranks not 1..N')
vals = [r['implied_value_usd'] for r in board]
if any(vals[i] > vals[i-1] for i in range(1, len(vals))): fails.append('implied rises with rank')
if len(set(vals)) == 1: fails.append('all implied equal')
# coverage
missing = [s for s in snap['watchlist'] if s not in seen]
if missing: fails.append('missing watchlist: %s' % missing)
print('watchlist coverage: %d/%d' % (len(snap['watchlist']) - len(missing), len(snap['watchlist'])))
print('implied range: $%s .. $%s' % (f"{vals[-1]:,}", f"{vals[0]:,}"))
print('farm step length range: %d..%d chars' % (
    min(len(s) for r in board for s in r['farm_steps']),
    max(len(s) for r in board for s in r['farm_steps'])))

# screenshots
for f in sorted(os.listdir(WORK + '/screenshots')):
    p = WORK + '/screenshots/' + f
    im = Image.open(p); im.load()
    sz = os.path.getsize(p)
    ok = im.format == 'PNG' and sz <= 524288 and max(im.size) <= 1440 and min(im.size) >= 800
    print(f, im.size, im.format, sz, 'bytes', 'OK' if ok else 'FAIL')
    if not ok: fails.append('screenshot ' + f)

# dashboard source size (dossier text cap 65536, incl. block overhead ~600B)
total = 0
for root, _, files in os.walk(WORK + '/dashboard'):
    for f in files:
        fp = os.path.join(root, f)
        total += os.path.getsize(fp)
        txt = open(fp, encoding='utf-8').read()
        if re.search(r'https?://', txt):
            # visible text/comments may carry URLs; attributes/imports may not. Flag for review.
            for i, line in enumerate(txt.split('\n'), 1):
                if re.search(r'https?://', line):
                    print('  URL mention %s:%d: %s' % (f, i, line.strip()[:100]))
print('dashboard source total: %d bytes (cap 65536)' % total)
if total > 64000: fails.append('dashboard too big for dossier text')

# data.js sanity + board/data agreement (judge criterion 1: screenshots and code agree)
data = open(OUT + '/data.js', encoding='utf-8').read()
assert data.startswith('window.RADAR=')
d = json.loads(data[len('window.RADAR='):-2])
print('data.js rows:', len(d['rows']), '| live:', len(d['live']), '| top implied matches board:', d['rows'][0]['implied_value_usd'] == board[0]['implied_value_usd'])
print('data.js bytes:', len(data))
# no http(s) in dashboard attributes: check index.html for href/src with http
html = open(WORK + '/dashboard/index.html', encoding='utf-8').read()
if re.search(r'(href|src)\s*=\s*["\']https?://', html): fails.append('http in html attr')
js = open(WORK + '/dashboard/app.js', encoding='utf-8').read()
if re.search(r'(import|require|from)\s*\(?\s*["\']https?://', js): fails.append('http import in js')
css = open(WORK + '/dashboard/app.css', encoding='utf-8').read()
if 'http' in css: fails.append('http in css')

print('\nFAILS:', fails if fails else 'none — all gate rows I can check pass')
