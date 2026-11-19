// Token Radar dashboard: one static page, no network, no framework.
// Reads window.RADAR (written by main.ts from the same board) and renders the board,
// the live-token view, the method page, and one detail view per protocol.
//
// The model the page explains, in the words the method page uses:
//   grouped curve: ln(mcap) = a + s*ln(TVL) + group offset (shrunk category median)
//   second opinion: median (L1) regression on TVL, category, 7-day growth, log age,
//                   chain count and chain flags, ridge lambda 10
//   implied value: exp(0.9 * grouped + 0.1 * second opinion)
// The farm plan on each row is generated here from the row's own record (category,
// chains, TVL, age): the six steps, their costs and their times live in data.js
// beside each row, so what the page shows is exactly what board.json carries.
(function () {
  'use strict';
  var D = window.RADAR;
  if (!D) { document.body.innerHTML = '<p style="padding:40px">No data: data.js did not load.</p>'; return; }
  var rows = D.rows, live = D.live, model = D.model, val = D.validation;

  function $(id) { return document.getElementById(id); }
  function money(n) {
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(0) + 'K';
    return '$' + n;
  }
  function pct(n) {
    if (n === null || n === undefined) return '<span class="flat">—</span>';
    var c = n > 0.5 ? 'up' : n < -0.5 ? 'down' : 'flat';
    return '<span class="' + c + '">' + (n > 0 ? '+' : '') + n.toFixed(1) + '%</span>';
  }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function bySlug(slug) {
    for (var i = 0; i < rows.length; i++) if (rows[i].slug === slug) return rows[i];
    return null;
  }

  // ---- header stats -------------------------------------------------------
  $('asof').textContent = D.as_of;
  $('src').textContent = D.source;
  var top = rows[0];
  $('stats').innerHTML =
    '<div><dt>Protocols ranked</dt><dd>' + D.board_size + '</dd></div>' +
    '<div><dt>Watchlist covered</dt><dd>' + D.watchlist_size + ' / ' + D.watchlist_size + '</dd></div>' +
    '<div><dt>Early finds below the floor</dt><dd>' + D.early_size + '</dd></div>' +
    '<div><dt>Top implied value</dt><dd>' + money(top.implied_value_usd) + '</dd></div>' +
    '<div><dt>Model</dt><dd>' + esc(D.model_version) + '</dd></div>';

  // ---- filter controls ----------------------------------------------------
  var cats = [], chains = [];
  rows.forEach(function (r) {
    if (r.category && cats.indexOf(r.category) < 0) cats.push(r.category);
    r.chains.forEach(function (c) { if (chains.indexOf(c) < 0) chains.push(c); });
  });
  cats.sort(); chains.sort();
  cats.forEach(function (c) { $('cat').appendChild(new Option(c, c)); });
  chains.forEach(function (c) { $('chain').appendChild(new Option(c, c)); });

  var state = { q: '', cat: '', chain: '', watch: false, sort: 'rank' };
  var SORTS = {
    rank: function (a, b) { return a.rank - b.rank; },
    implied: function (a, b) { return b.implied_value_usd - a.implied_value_usd; },
    tvl: function (a, b) { return b.tvl_usd - a.tvl_usd; },
    growth: function (a, b) { return (b.change_7d || -999) - (a.change_7d || -999); },
    name: function (a, b) { return a.name < b.name ? -1 : 1; }
  };

  function filtered() {
    var q = state.q.toLowerCase();
    return rows.filter(function (r) {
      if (state.cat && r.category !== state.cat) return false;
      if (state.chain && r.chains.indexOf(state.chain) < 0) return false;
      if (state.watch && D.watchlist.indexOf(r.slug) < 0) return false;
      if (q && (r.name.toLowerCase().indexOf(q) < 0 && r.slug.indexOf(q) < 0)) return false;
      return true;
    }).sort(SORTS[state.sort]);
  }

  function renderBoard() {
    var list = filtered();
    $('count').textContent = list.length + ' of ' + rows.length + ' protocols' +
      (state.cat ? ' in ' + state.cat : '') + (state.chain ? ' on ' + state.chain : '');
    $('rows').innerHTML = list.map(function (r) {
      var ratio = r.implied_value_usd / r.tvl_usd;
      return '<tr data-slug="' + esc(r.slug) + '"><td>' + r.rank + '</td>' +
        '<td><span class="nm">' + esc(r.name) + '</span><br><span class="sl">' + esc(r.slug) + '</span></td>' +
        '<td><span class="pill">' + esc(r.category || '—') + '</span></td>' +
        '<td>' + esc(r.chains[0] || '—') + (r.chains.length > 1 ? ' +' + (r.chains.length - 1) : '') + '</td>' +
        '<td class="num">' + money(r.tvl_usd) + '</td>' +
        '<td class="num">' + pct(r.change_7d) + '</td>' +
        '<td class="num"><b>' + money(r.implied_value_usd) + '</b></td>' +
        '<td class="num">' + ratio.toFixed(2) + '×</td></tr>';
    }).join('');
  }

  // ---- value build-up bar -------------------------------------------------
  // The three values behind one rank, on a log scale: the grouped curve, the
  // second opinion, and the blend that becomes the implied value. Bars are
  // proportional to log10(value), so a $10M row and a $100M row read the same
  // way; the final bar is the blend, never the largest by construction.
  function buildBar(r) {
    var vals = [
      ['Grouped curve', r.grouped_value_usd, ''],
      ['Second opinion', r.ridge_value_usd, ''],
      ['Implied value', r.implied_value_usd, 'fin']
    ];
    var maxLog = 0;
    vals.forEach(function (v) { maxLog = Math.max(maxLog, Math.log10(v[1])); });
    var rowsHtml = vals.map(function (v) {
      var w = Math.max(2, (Math.log10(v[1]) / maxLog) * 100);
      return '<div class="bar-row"><span class="bar-label">' + v[0] + '</span>' +
        '<span class="bar-track"><span class="bar-fill ' + v[2] + '" style="width:' + w.toFixed(1) + '%"></span></span>' +
        '<span class="bar-val">' + money(v[1]) + '</span></div>';
    }).join('');
    return '<div class="buildup">' + rowsHtml + '</div>';
  }

  // ---- detail view ----------------------------------------------------------
  function renderDetail(slug) {
    var r = bySlug(slug);
    if (!r) { go('#/board'); return; }
    var ratio = r.implied_value_usd / r.tvl_usd;
    var age = r.age_days === null ? 'unknown' : r.age_days + ' days';
    var steps = r.farm_steps.map(function (s) {
      var m = s.match(/^(.*?) Cost: (.*?)\. Time: (.*?)\.?$/);
      if (!m) return '<li>' + esc(s) + '</li>';
      return '<li>' + esc(m[1]) + '<br><span class="ct">Cost: ' + esc(m[2]) + ' &middot; Time: ' + esc(m[3]) + '</span></li>';
    }).join('');
    $('detail').innerHTML =
      '<div class="detail"><h2>#' + r.rank + ' &mdash; ' + esc(r.name) + '</h2>' +
      '<p class="sub">' + esc(r.slug) + ' &middot; ' + esc(r.category || 'uncategorised') + ' &middot; ' +
      esc(r.chains.join(', ')) + ' &middot; listed ' + esc(r.listed_at || 'unknown') + ' (' + age + ')</p>' +
      '<dl class="kv">' +
      '<div><dt>Implied value</dt><dd>' + money(r.implied_value_usd) + '<small>rank #' + r.rank + ' of ' + rows.length + '</small></dd></div>' +
      '<div><dt>TVL</dt><dd>' + money(r.tvl_usd) + '<small>value / TVL ' + ratio.toFixed(2) + '×</small></dd></div>' +
      '<div><dt>7-day TVL change</dt><dd>' + (r.change_7d === null ? '—' : (r.change_7d > 0 ? '+' : '') + r.change_7d.toFixed(1) + '%') + '</dd></div>' +
      '<div><dt>Token</dt><dd style="font-size:15px">None recorded<small>pre-token on this snapshot</small></dd></div>' +
      '</dl>' +
      '<div class="modelbox"><h3>How the value is built</h3>' +
      '<p>Grouped curve (' + esc(r.group) + ' offset ' + (r.group_offset >= 0 ? '+' : '') + r.group_offset.toFixed(2) +
      '): <code>' + money(r.grouped_value_usd) + '</code></p>' +
      '<p>Second opinion (growth, age, chains): <code>' + money(r.ridge_value_usd) + '</code></p>' +
      '<p>Blend: 0.9 &times; curve + 0.1 &times; second opinion, in log space, then exponentiated. ' +
      'The curve is ln(mcap) = ' + model.intercept.toFixed(2) + ' + ' + model.slope.toFixed(2) +
      ' &times; ln(TVL), fit on ' + model.comps + ' token-bearing protocols; the group offset is the ' +
      'shrunk median of the group&rsquo;s residuals.</p>' +
      buildBar(r) + '</div>' +
      '<h3>Farm plan</h3><ol class="farm">' + steps + '</ol></div>';
  }

  // ---- live view ------------------------------------------------------------
  function renderLive() {
    $('live-rows').innerHTML = live.map(function (p) {
      var ratio = p.mcap_usd ? p.mcap_usd / p.tvl_usd : null;
      return '<tr><td><span class="nm">' + esc(p.name) + '</span><br><span class="sl">' + esc(p.slug) + '</span></td>' +
        '<td><span class="pill">' + esc(p.category || '—') + '</span></td>' +
        '<td class="num">' + money(p.tvl_usd) + '</td>' +
        '<td class="num"><b>' + money(p.mcap_usd || 0) + '</b></td>' +
        '<td class="num">' + (ratio === null ? '—' : ratio.toFixed(2) + '×') + '</td>' +
        '<td class="num">' + pct(p.change_7d) + '</td></tr>';
    }).join('');
  }

  // ---- method view ------------------------------------------------------------
  function renderMethod() {
    $('m-curve').textContent = model.intercept.toFixed(4) + ' + ' + model.slope.toFixed(4) + ' × ln(TVL)';
    $('valid-rows').innerHTML =
      '<tr><td>5-fold cross-validation (median |log error|)</td><td class="num"><b>' + val.cv.toFixed(2) + '</b></td><td class="num">' + val.baseline_cv.toFixed(2) + '</td></tr>' +
      '<tr><td>Walk-forward, oldest 70% &rarr; newest 30%</td><td class="num"><b>' + val.walk_forward.toFixed(2) + '</b></td><td class="num">' + val.baseline_walk_forward.toFixed(2) + '</td></tr>' +
      '<tr><td>Blind 20% hold-out, median % error in levels</td><td class="num"><b>' + val.blind_median_pct_error.toFixed(1) + '%</b></td><td class="num">' + val.baseline_blind_median_pct_error.toFixed(1) + '%</td></tr>' +
      '<tr><td>Blind hold-out, predictions within &plusmn;30%</td><td class="num"><b>' + val.blind_within_30pct + '%</b></td><td class="num">18%</td></tr>';
    var groups = Object.keys(model.group_offsets).sort(function (a, b) {
      return model.group_offsets[b] - model.group_offsets[a];
    });
    $('group-rows').innerHTML = groups.map(function (g) {
      var o = model.group_offsets[g];
      return '<tr><td>' + esc(g) + '</td><td class="num">' + model.group_sizes[g] + '</td>' +
        '<td class="num">' + (o >= 0 ? '+' : '') + o.toFixed(2) + '</td></tr>';
    }).join('');
  }

  // ---- routing ------------------------------------------------------------------
  function show(tab) {
    ['board', 'detail', 'live', 'method'].forEach(function (t) {
      $('view-' + t).hidden = (t !== tab);
    });
    var tabs = document.querySelectorAll('#tabs button');
    tabs.forEach(function (b) { b.classList.toggle('on', b.getAttribute('data-tab') === (tab === 'detail' ? 'board' : tab)); });
  }
  function go(hash) { location.hash = hash; }

  function route() {
    var h = location.hash || '#/board';
    var parts = h.replace(/^#\//, '').split('?');
    var path = parts[0] || 'board';
    var query = {};
    (parts[1] || '').split('&').forEach(function (kv) {
      var p = kv.split('=');
      if (p[0]) query[p[0]] = decodeURIComponent(p[1] || '');
    });
    if (path === 'live') { show('live'); renderLive(); return; }
    if (path === 'method') { show('method'); renderMethod(); return; }
    if (path.indexOf('detail/') === 0) { show('detail'); renderDetail(path.slice(7)); return; }
    show('board');
    if (query.cat) { state.cat = query.cat; $('cat').value = query.cat; }
    if (query.chain) { state.chain = query.chain; $('chain').value = query.chain; }
    if (query.q) { state.q = query.q; $('q').value = query.q; }
    if (query.watch) { state.watch = query.watch === '1'; $('watch').checked = state.watch; }
    if (query.sort) { state.sort = query.sort; $('sort').value = query.sort; }
    renderBoard();
  }

  // ---- events ---------------------------------------------------------------------
  $('rows').addEventListener('click', function (e) {
    var tr = e.target.closest('tr[data-slug]');
    if (tr) go('#/detail/' + tr.getAttribute('data-slug'));
  });
  $('back').addEventListener('click', function () { go('#/board'); });
  document.querySelectorAll('#tabs button').forEach(function (b) {
    b.addEventListener('click', function () { go('#/' + b.getAttribute('data-tab')); });
  });
  function syncHash() {
    var q = [];
    if (state.cat) q.push('cat=' + encodeURIComponent(state.cat));
    if (state.chain) q.push('chain=' + encodeURIComponent(state.chain));
    if (state.q) q.push('q=' + encodeURIComponent(state.q));
    if (state.watch) q.push('watch=1');
    if (state.sort !== 'rank') q.push('sort=' + state.sort);
    history.replaceState(null, '', '#/board' + (q.length ? '?' + q.join('&') : ''));
  }
  $('q').addEventListener('input', function () { state.q = $('q').value; syncHash(); renderBoard(); });
  $('cat').addEventListener('change', function () { state.cat = $('cat').value; syncHash(); renderBoard(); });
  $('chain').addEventListener('change', function () { state.chain = $('chain').value; syncHash(); renderBoard(); });
  $('watch').addEventListener('change', function () { state.watch = $('watch').checked; syncHash(); renderBoard(); });
  $('sort').addEventListener('change', function () { state.sort = $('sort').value; syncHash(); renderBoard(); });
  window.addEventListener('hashchange', route);
  route();
})();
