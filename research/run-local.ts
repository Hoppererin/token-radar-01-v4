// Local test driver: runs buildRadar WITHOUT the vendor dossier step, so the model
// and the board can be verified while vendor/makefun-board.ts is quarantined.
// Usage: node run-local.ts --input <snapshot.json> --output <out dir>
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { boardIssue, buildRadar, dataScript, parseSnapshot } from './lib/model.ts';

function arg(name: string, fallback: string): string {
  const at = process.argv.indexOf(`--${name}`);
  if (at >= 0 && at + 1 < process.argv.length) return process.argv[at + 1]!;
  const withEquals = process.argv.find((a) => a.startsWith(`--${name}=`));
  return withEquals === undefined ? fallback : withEquals.slice(name.length + 3);
}

const inputFile = arg('input', '/home/hatch/workspace/makefun/token-radar-01-kit/input/snapshot.json');
const out = arg('output', '/tmp/token-radar-v3-out');
const snapshot = parseSnapshot(readFileSync(inputFile, 'utf8'));
const { board, radar } = buildRadar(snapshot);
const issue = boardIssue(board, snapshot);
console.log('board issue:', issue ?? 'none');
console.log('rows:', board.length, '| watchlist:', radar.watchlist_size, '| early:', radar.early_size, '| live:', radar.live_size);
console.log('model: comps', radar.model.comps, '| intercept', radar.model.intercept.toFixed(4), '| slope', radar.model.slope.toFixed(4));
console.log('global median residual:', radar.model.global_median_residual.toFixed(4));
console.log('ridge: intercept', radar.model.ridge_intercept.toFixed(4), '| slope', radar.model.ridge_slope.toFixed(4), '| growth', radar.model.ridge_growth.toFixed(4), '| log-age', radar.model.ridge_log_age.toFixed(4), '| n-chains', radar.model.ridge_n_chains.toFixed(4));
console.log('top 5:');
for (const r of board.slice(0, 5)) {
  console.log(`  #${r.rank} ${r.name} implied $${r.implied_value_usd.toLocaleString()} (grouped $${r.model.grouped_value_usd.toLocaleString()}, ridge $${r.model.ridge_value_usd.toLocaleString()}, ${r.model.group} ${r.model.group_offset}) TVL $${r.tvl_usd.toLocaleString()}`);
}
console.log('bottom 3:');
for (const r of board.slice(-3)) {
  console.log(`  #${r.rank} ${r.name} implied $${r.implied_value_usd.toLocaleString()} TVL $${r.tvl_usd.toLocaleString()}`);
}
// determinism: build twice, compare bytes
const again = buildRadar(snapshot);
const b1 = JSON.stringify({ rows: board });
const b2 = JSON.stringify({ rows: again.board });
console.log('deterministic:', b1 === b2);
mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'board.json'), `${JSON.stringify({ rows: board })}\n`);
writeFileSync(join(out, 'data.js'), dataScript(radar));
console.log('wrote', join(out, 'board.json'), 'and', join(out, 'data.js'));
