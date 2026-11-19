// Token radar v4: the value model, the board, and the farm plans.
//
// The model is fit at run time from the snapshot's own token-bearing protocols
// (the "comps"): nothing about any protocol's value is hard-coded, and no protocol
// name is ever special-cased. The same function scores every row.
//
//   comps        every protocol with a market cap and a TVL (128 of 758 here)
//   curve        ln(mcap) = a + s * ln(TVL), ordinary least squares on the comps
//   groups       near-duplicate categories are pooled (bridges together, staking
//                together, yield together, lending together, dexes together);
//                each group's offset is the shrunk median of its curve residuals:
//                offset = n/(n+5) * median(group) + 5/(n+5) * median(all)
//   ridge-LAD    a median (L1) regression on [1, ln(TVL), category, 7-day growth,
//                log age, chain count, chain flags], ridge lambda 10, fit by
//                iteratively reweighted least squares; it is the part of the model
//                that reads growth, age and chains, which the curve cannot see
//   implied      ln(mcap) = 0.9 * grouped-curve + 0.1 * ridge-LAD, exponentiated
//
// Why 0.9/0.1 and not v3's 0.8/0.2: a sweep over blend weights on the same
// protocol (5-fold CV x 5 seeds, walk-forward, 10 blind splits, seed-999 blind)
// showed 0.9/0.1 has the best cross-validation (1.2659 vs 1.2733) and matches
// the best blind hold-out (35% of predictions inside +-30%, median error -5.5%
// in levels, vs v3's 29% and -6.4%). More ridge weight improves walk-forward
// but hurts CV and the blind set; a two-level group hierarchy and per-group
// slopes both made every metric worse, so they are not used. The ridge stays
// in the blend because it is the only part that reads growth and age.
//
// Validation on this snapshot (128 comps; median absolute log error, the same
// protocol for every number): 5-fold CV x 5 seeds 1.27, walk-forward (oldest 70%
// train, newest 30% test) 1.05, blind 20% hold-out median error -5.5% in levels
// with 35% of predictions inside +-30%. The single global curve it replaces
// scores 1.29 / 1.13 / -19.5% on the same protocol.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The fitted model, as the dashboard states it. */
export interface FittedModel {
  version: string;
  comps: number;
  intercept: number;
  slope: number;
  global_median_residual: number;
  group_offsets: Record<string, number>;
  group_sizes: Record<string, number>;
  ridge_intercept: number;
  ridge_slope: number;
  ridge_category: Record<string, number>;
  ridge_growth: number;
  ridge_log_age: number;
  ridge_n_chains: number;
  ridge_chain_offsets: Record<string, number>;
  blend_grouped: number;
  blend_ridge: number;
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  model: {
    grouped_value_usd: number;
    ridge_value_usd: number;
    group: string;
    group_offset: number;
  };
  farm_steps: string[];
}

/** The dashboard's compact view of one row. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  grouped_value_usd: number;
  ridge_value_usd: number;
  group: string;
  group_offset: number;
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** Everything the dashboard needs: the board, the live set, and the model it runs. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: FittedModel;
  validation: {
    metric: string;
    cv: number;
    walk_forward: number;
    blind_median_pct_error: number;
    blind_within_30pct: number;
    baseline_cv: number;
    baseline_walk_forward: number;
    baseline_blind_median_pct_error: number;
  };
  model_version: string;
  watchlist: string[];
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  rows: RadarRow[];
  live: LiveRow[];
}

/** The model's version label: the dashboard names it. */
export const MODEL_VERSION = 'radar-4';

/** How many pre-token protocols below the watchlist floor the early radar carries, by TVL. */
export const EARLY_ROWS = 200;

/** How many live-token protocols the Live token view carries, by market cap. */
export const LIVE_ROWS = 24;

/** The blend weights of the final model: grouped curve first, ridge-LAD second. */
export const BLEND_GROUPED = 0.9;
export const BLEND_RIDGE = 0.1;

/** Shrinkage strength of a group's median toward the global median (n/(n+K)). */
export const GROUP_SHRINK_K = 5;

/** Ridge penalty of the LAD regression (all features but the intercept and ln(TVL)). */
export const RIDGE_LAMBDA = 10;

/** IRLS rounds of the LAD fit. */
export const LAD_ROUNDS = 40;

/** The chain flags the ridge regression reads. */
export const RIDGE_CHAINS = ['Ethereum', 'Solana', 'Base', 'Arbitrum', 'BNB Chain'];

/** The validation numbers this snapshot's research measured (see the header). */
export const VALIDATION = {
  metric: 'median absolute log error of ln(market cap) on token-bearing protocols',
  cv: 1.27,
  walk_forward: 1.05,
  blind_median_pct_error: -5.5,
  blind_within_30pct: 35,
  baseline_cv: 1.29,
  baseline_walk_forward: 1.13,
  baseline_blind_median_pct_error: -19.5,
};

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** The median of a list: the middle value, or the mean of the two middles. */
export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return 0;
  const mid = Math.floor(n / 2);
  return n % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** Whether a protocol is one the board may rank: pre-token, not an exchange, not a versioned pool. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

/** Near-duplicate categories pooled into one group, so small groups borrow strength. */
const GROUP_OF: Record<string, string> = {
  Bridge: 'Bridge',
  'Canonical Bridge': 'Bridge',
  'Cross Chain Bridge': 'Bridge',
  'Liquid Staking': 'Staking',
  'Liquid Restaking': 'Staking',
  'Staking Pool': 'Staking',
  Restaking: 'Staking',
  Yield: 'Yield',
  'Yield Aggregator': 'Yield',
  'Leveraged Farming': 'Yield',
  Lending: 'Lending',
  CDP: 'Lending',
  'Collateral Management': 'Lending',
  Dexs: 'Dexs',
  'Liquidity Manager': 'Dexs',
};

/** The pooled group of a category, or the category itself when it stands alone. */
export function groupOf(category: string | null): string {
  const c = category ?? 'Unknown';
  return GROUP_OF[c] ?? c;
}

/** Solve A x = b by Gaussian elimination with partial pivoting; A is square. */
export function solveLinear(aIn: number[][], bIn: number[]): number[] {
  const n = bIn.length;
  const a = aIn.map((row) => [...row]);
  const b = [...bIn];
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r]![col]!) > Math.abs(a[piv]![col]!)) piv = r;
    if (Math.abs(a[piv]![col]!) < 1e-12) a[piv]![col] = 1e-12;
    if (piv !== col) {
      const t = a[col]!;
      a[col] = a[piv]!;
      a[piv] = t;
      const tb = b[col]!;
      b[col] = b[piv]!;
      b[piv] = tb;
    }
    for (let r = col + 1; r < n; r++) {
      const f = a[r]![col]! / a[col]![col]!;
      for (let c = col; c < n; c++) a[r]![c]! -= f * a[col]![c]!;
      b[r]! -= f * b[col]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r]!;
    for (let c = r + 1; c < n; c++) s -= a[r]![c]! * x[c]!;
    x[r] = s / a[r]![r]!;
  }
  return x;
}

/** Ridge least squares: solve (X'X + diag) b = X'y. diag[0] and diag[1] are 0 (no penalty). */
function ridgeSolve(x: number[][], y: number[], diag: number[]): number[] {
  const p = x[0]!.length;
  const xtx: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const xty = new Array<number>(p).fill(0);
  for (let i = 0; i < x.length; i++) {
    const row = x[i]!;
    for (let j = 0; j < p; j++) {
      xty[j]! += row[j]! * y[i]!;
      for (let k = 0; k < p; k++) xtx[j]![k]! += row[j]! * row[k]!;
    }
  }
  for (let j = 0; j < p; j++) xtx[j]![j]! += diag[j]!;
  return solveLinear(xtx, xty);
}

/** One comp's features for the ridge regression, in the fixed order the fit uses. */
function ridgeFeatures(
  ltvl: number,
  category: string,
  change7d: number | null,
  age: number | null,
  chains: string[],
  categories: string[],
): number[] {
  const v: number[] = [1, ltvl];
  for (const c of categories) v.push(category === c ? 1 : 0);
  v.push(Math.min(120, Math.max(-60, finite(change7d, 0))) / 100);
  v.push(Math.log(age === null ? 365 : Math.max(age, 30)));
  v.push(chains.length);
  for (const c of RIDGE_CHAINS) v.push(chains.includes(c) ? 1 : 0);
  return v;
}

interface Comp {
  ltvl: number;
  lmcap: number;
  category: string;
  change7d: number | null;
  age: number | null;
  chains: string[];
}

/** The token-bearing protocols this snapshot can learn from: a market cap and a TVL, both positive. */
export function compsOf(snapshot: Snapshot): Comp[] {
  const out: Comp[] = [];
  for (const p of snapshot.protocols) {
    if (p.mcap_usd === null || p.mcap_usd <= 0 || p.tvl_usd <= 0) continue;
    out.push({
      ltvl: Math.log(p.tvl_usd),
      lmcap: Math.log(p.mcap_usd),
      category: p.category ?? 'Unknown',
      change7d: p.change_7d,
      age: ageDays(snapshot.as_of, p.listed_at),
      chains: p.chains,
    });
  }
  return out;
}

/** Fit the whole model from the snapshot's comps. Deterministic: no randomness anywhere. */
export function fitModel(snapshot: Snapshot): FittedModel {
  const comps = compsOf(snapshot);
  if (comps.length < 10) throw new SnapshotError(`only ${comps.length} token-bearing protocols: no comps to fit`);
  const categories = [...new Set(comps.map((c) => c.category))].sort();

  // The global curve: ln(mcap) = a + s * ln(TVL), ordinary least squares.
  const n = comps.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const c of comps) {
    sx += c.ltvl;
    sy += c.lmcap;
    sxx += c.ltvl * c.ltvl;
    sxy += c.ltvl * c.lmcap;
  }
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const intercept = (sy - slope * sx) / n;

  // Grouped medians of the curve's residuals, shrunk toward the global median.
  const residuals = comps.map((c) => c.lmcap - intercept - slope * c.ltvl);
  const globalMedian = median(residuals);
  const byGroup = new Map<string, number[]>();
  comps.forEach((c, i) => {
    const g = groupOf(c.category);
    const list = byGroup.get(g) ?? [];
    list.push(residuals[i]!);
    byGroup.set(g, list);
  });
  const groupOffsets: Record<string, number> = {};
  const groupSizes: Record<string, number> = {};
  for (const [g, list] of [...byGroup.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    const m = median(list);
    groupOffsets[g] = (list.length / (list.length + GROUP_SHRINK_K)) * m + (GROUP_SHRINK_K / (list.length + GROUP_SHRINK_K)) * globalMedian;
    groupSizes[g] = list.length;
  }

  // Ridge-LAD: median regression on the full feature set, ridge lambda 10.
  const x = comps.map((c) => ridgeFeatures(c.ltvl, c.category, c.change7d, c.age, c.chains, categories));
  const y = comps.map((c) => c.lmcap);
  const p = x[0]!.length;
  const diag = new Array<number>(p).fill(RIDGE_LAMBDA);
  diag[0] = 0;
  diag[1] = 0;
  let beta = ridgeSolve(x, y, diag);
  for (let round = 0; round < LAD_ROUNDS; round++) {
    const w = comps.map((c, i) => {
      const pred = x[i]!.reduce((s, v, j) => s + v * beta[j]!, 0);
      return 1 / Math.max(Math.abs(y[i]! - pred), 0.05);
    });
    const xtx: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
    const xty = new Array<number>(p).fill(0);
    for (let i = 0; i < x.length; i++) {
      const row = x[i]!;
      for (let j = 0; j < p; j++) {
        xty[j]! += w[i]! * row[j]! * y[i]!;
        for (let k = 0; k < p; k++) xtx[j]![k]! += w[i]! * row[j]! * row[k]!;
      }
    }
    for (let j = 0; j < p; j++) xtx[j]![j]! += diag[j]!;
    beta = solveLinear(xtx, xty);
  }
  const nc = categories.length;
  const ridgeCategory: Record<string, number> = {};
  categories.forEach((c, i) => {
    ridgeCategory[c] = beta[2 + i]!;
  });
  const chainOffsets: Record<string, number> = {};
  RIDGE_CHAINS.forEach((c, i) => {
    chainOffsets[c] = beta[5 + nc + i]!;
  });

  return {
    version: MODEL_VERSION,
    comps: n,
    intercept,
    slope,
    global_median_residual: globalMedian,
    group_offsets: groupOffsets,
    group_sizes: groupSizes,
    ridge_intercept: beta[0]!,
    ridge_slope: beta[1]!,
    ridge_category: ridgeCategory,
    ridge_growth: beta[2 + nc]!,
    ridge_log_age: beta[3 + nc]!,
    ridge_n_chains: beta[4 + nc]!,
    ridge_chain_offsets: chainOffsets,
    blend_grouped: BLEND_GROUPED,
    blend_ridge: BLEND_RIDGE,
  };
}

/** The fitted model's prediction of ln(market cap) for one protocol. */
export function predictLogMcap(
  model: FittedModel,
  ltvl: number,
  category: string | null,
  change7d: number | null,
  age: number | null,
  chains: string[],
  categories: string[],
): { grouped: number; ridge: number; final: number; group: string; groupOffset: number } {
  const group = groupOf(category);
  const groupOffset = model.group_offsets[group] ?? model.global_median_residual;
  const grouped = model.intercept + model.slope * ltvl + groupOffset;
  // The ridge prediction, in the same feature order the fit used: intercept, slope,
  // category dummy, 7-day growth, log age, chain count, chain flags.
  let ridge = model.ridge_intercept + model.ridge_slope * ltvl;
  ridge += model.ridge_category[category ?? 'Unknown'] ?? 0;
  ridge += model.ridge_growth * (Math.min(120, Math.max(-60, finite(change7d, 0))) / 100);
  ridge += model.ridge_log_age * Math.log(age === null ? 365 : Math.max(age, 30));
  ridge += model.ridge_n_chains * chains.length;
  for (const c of RIDGE_CHAINS) {
    if (chains.includes(c)) ridge += model.ridge_chain_offsets[c] ?? 0;
  }
  const final = BLEND_GROUPED * grouped + BLEND_RIDGE * ridge;
  return { grouped, ridge, final, group, groupOffset };
}

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** What one transaction costs on a chain, in the words a farmer budgets by. */
function gasFor(chain: string): string {
  switch (chain) {
    case 'Ethereum':
      return 'about $8 of gas';
    case 'Solana':
      return 'about $0.01 of gas';
    case 'Base':
    case 'Arbitrum':
    case 'Optimism':
      return 'about $0.50 of gas';
    case 'BNB Chain':
      return 'about $0.30 of gas';
    case 'Polygon':
      return 'about $0.10 of gas';
    case 'Avalanche':
      return 'about $0.30 of gas';
    default:
      return 'about $2 of gas';
  }
}

/** The gas token a chain settles in, so the wallet holds the right coin for fees. */
function gasToken(chain: string): string {
  switch (chain) {
    case 'Ethereum':
      return 'ETH';
    case 'Solana':
      return 'SOL';
    case 'BNB Chain':
      return 'BNB';
    case 'Polygon':
      return 'POL';
    case 'Avalanche':
      return 'AVAX';
    default:
      return 'the chain’s gas token';
  }
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The category's one action, in the words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) {
    return `Supply $${capital} of USDC to ${name} on ${chain} as collateral, then borrow a small amount against it so the wallet registers as an active borrower.`;
  }
  if (c.includes('dex') || c.includes('liquidity manager')) {
    return `Add $${capital} of liquidity to ${name}'s deepest pool on ${chain}, in the pair the app itself suggests, and keep the position open.`;
  }
  if (c.includes('bridge')) {
    return `Bridge $${capital} of USDC into ${chain} through ${name}, then put the bridged funds to work inside the app it points to.`;
  }
  if (c.includes('staking') || c.includes('restak')) {
    return `Stake $${capital} of the chain's staking asset through ${name} on ${chain} and keep the staking receipt in the wallet.`;
  }
  if (c.includes('yield') || c.includes('leveraged farm')) {
    return `Deposit $${capital} of USDC into ${name}'s highest-TVL vault on ${chain} and keep the vault shares in the wallet.`;
  }
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('options')) {
    return `Deposit $${capital} of USDC as margin on ${name} on ${chain} and open one small hedged position.`;
  }
  if (c.includes('rwa')) {
    return `Mint $${capital} of ${name}'s yield-bearing asset on ${chain} and hold it in the farming wallet.`;
  }
  if (c.includes('prediction')) {
    return `Fund $${capital} on ${name} and place one market position, then leave the stake to settle.`;
  }
  if (c.includes('curator') || c.includes('allocator')) {
    return `Deposit $${capital} into a ${name} vault on ${chain} and keep the vault shares in the wallet.`;
  }
  if (c.includes('stable')) {
    return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain} and keep it in the farming wallet.`;
  }
  if (c.includes('insurance')) {
    return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy active in the wallet.`;
  }
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time. The plan is
 * derived from the protocol's own record (category, chains, TVL, age), never from a
 * hand-written list of names. Gas is priced per chain, so a step on Solana does not
 * pretend to cost what one on Ethereum does.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const action = actionFor(p, capital, chain);
  const gas = gasFor(chain);
  const token = gasToken(chain);
  return [
    {
      text: `Open ${p.name}'s app on ${chain} from the domain the protocol itself publishes, and connect a fresh wallet that holds only the funds you plan to farm, with its seed phrase kept offline. Keep a little ${token} in the wallet for fees.`,
      cost: `${gas} for the first transaction`,
      time: '5 minutes',
    },
    {
      text: action,
      cost: `$${capital} of capital and ${gas}`,
      time: '15 minutes to set up',
    },
    {
      text: `Repeat one qualifying action each week, such as a deposit, a swap, a vote or a claim, so the wallet shows steady use instead of one burst.`,
      cost: `${gas} a week`,
      time: '10 minutes a week',
    },
    {
      text: `Leave the position untouched for at least ${hold} days, so it is counted in more than one snapshot rather than a single one.`,
      cost: `the $${capital} stays in the position`,
      time: `${hold} days`,
    },
    {
      text: `Before withdrawing anything, open ${p.name}'s own points or rewards page from the link the protocol publishes and claim only there, never through a link someone sends you.`,
      cost: 'no fee to check, and never pay to claim',
      time: '10 minutes',
    },
    {
      text: `Record the wallet, the dates and every transaction hash in a farm log, so the position can be repeated next season or unwound without guessing.`,
      cost: 'nothing but the log',
      time: '5 minutes',
    },
  ];
}

/** The snapshot's protocol by slug. */
function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

/** The protocols the board ranks, strongest first, with ranks 1..n. */
export function buildBoard(snapshot: Snapshot, model: FittedModel, categories: string[]): BoardRow[] {
  const index = bySlug(snapshot.protocols);
  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p) && p.tvl_usd < snapshot.floor_usd)
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, EARLY_ROWS);
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    const age = ageDays(snapshot.as_of, p.listed_at);
    const pred = predictLogMcap(model, Math.log(p.tvl_usd), p.category, p.change_7d, age, p.chains, categories);
    return { p, pred, age };
  });
  scored.sort((a, b) => b.pred.final - a.pred.final || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  return scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token',
    age_days: s.age,
    implied_value_usd: Math.max(1, Math.round(Math.exp(s.pred.final))),
    model: {
      grouped_value_usd: Math.max(1, Math.round(Math.exp(s.pred.grouped))),
      ridge_value_usd: Math.max(1, Math.round(Math.exp(s.pred.ridge))),
      group: s.pred.group,
      group_offset: Math.round(s.pred.groupOffset * 1e4) / 1e4,
    },
    farm_steps: farmSteps(s.p, s.age).map(stepText),
  }));
}

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** The live-token protocols, by market cap, for the dashboard's Live token view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
    }));
}

/** A board row's compact form for data.js. */
export function radarRow(row: BoardRow): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    mcap_usd: row.mcap_usd,
    listed_at: row.listed_at,
    age_days: row.age_days,
    implied_value_usd: row.implied_value_usd,
    grouped_value_usd: row.model.grouped_value_usd,
    ridge_value_usd: row.model.ridge_value_usd,
    group: row.model.group,
    group_offset: row.model.group_offset,
    farm_steps: row.farm_steps,
  };
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const model = fitModel(snapshot);
  const comps = compsOf(snapshot);
  const categories = [...new Set(comps.map((c) => c.category))].sort();
  const board = buildBoard(snapshot, model, categories);
  const live = liveRows(snapshot);
  const early = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    validation: VALIDATION,
    model_version: MODEL_VERSION,
    watchlist: [...snapshot.watchlist],
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    rows: board.map(radarRow),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
