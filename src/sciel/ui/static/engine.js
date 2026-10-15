// The Engine tab: charts of the engine's own machinery. Uses el, color, isl, OPS, DECISIONS and section from the page.
const SEG = ["#0A84FF", "#3DAE7C", "#E8A33D", "#A66BE0", "#2BB3C0", "#D67AB1", "#5A5A5E"];
const OUTCOME = {better: ["#3DAE7C", "better than its parent"], same: ["#5A5A5E", "same score"], worse: ["#E8A33D", "worse"], failed: ["#E5484D", "failed"], measured: ["#98989D", "measured"]};
function svgBox(w, h, inner, label) { const d = el("div", {class: "chart"}); d.innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${inner}</svg>`; return d; }
const tx = (x, y, s, o = {}) => `<text x="${x}" y="${y}" fill="${o.c || "#98989D"}" font-size="${o.size || 11}" text-anchor="${o.a || "start"}">${String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")}</text>`;
function legend(items) { return el("div", {class: "legend"}, items.map(([c, t, dash]) => el("span", {}, el("i", {style: dash ? `background:repeating-linear-gradient(90deg,${c} 0 4px,transparent 4px 7px)` : `background:${c};height:${dash === false ? 8 : 2}px;width:${dash === false ? 8 : 14}px;border-radius:${dash === false ? "50%" : "0"}`}), t))); }

function islandTimeline(s, e) {
  const att = s.attempts || [], N = Math.max(1, att.length), kept = new Set(((e.schedule || {}).survivors || []).map(String));
  const islands = [...new Set(att.map(a => a.island))].sort((a, b) => a - b);
  if (!islands.length) return el("p", {class: "empty"}, "No attempts yet.");
  const W = 1000, L = 44, R = 170, row = 26, T = 8, H = T + islands.length * row + 34, X = n => L + (n - 1) / Math.max(1, N - 1) * (W - L - R);
  let g = "";
  islands.forEach((isl, k) => {
    const y = T + k * row + row / 2, own = att.filter(a => a.island === isl), first = own.length ? own[0].n : 1;
    g += tx(L - 10, y + 4, "I" + isl, {a: "end", c: color(isl)});
    g += `<line x1="${X(first)}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#2C2C2E"/>`;
    own.forEach(a => { const o = OUTCOME[a.status] || OUTCOME.measured; g += a.status === "failed"
      ? `<path d="M${X(a.n) - 4},${y - 4} L${X(a.n) + 4},${y + 4} M${X(a.n) + 4},${y - 4} L${X(a.n) - 4},${y + 4}" stroke="${o[0]}" stroke-width="1.6"/>`
      : `<circle cx="${X(a.n)}" cy="${y}" r="${a.status === "better" ? 5 : 3.5}" fill="${o[0]}"><title>#${a.n} ${a.child || ""} ${a.status}</title></circle>`; });
    const better = own.filter(a => a.status === "better").length;
    g += tx(W - R + 10, y + 4, own.length + " steps · " + better + "▲" + (kept.has(String(isl)) ? " · kept" : ""), {c: kept.has(String(isl)) ? "#F5F5F7" : "#98989D"});
  });
  for (let k = 0; k < 6; k++) { const n = Math.round(1 + (N - 1) * k / 5); g += tx(X(n), H - 14, n, {a: "middle"}); }
  g += tx((L + W - R) / 2, H - 1, "attempt", {a: "middle"});
  return el("div", {}, svgBox(W, H, g, "Each island's attempts over the run, coloured by outcome"),
    legend(Object.values(OUTCOME).slice(0, 4).map(([c, t]) => [c, t, false])));
}

function decisionBars(e) {
  const rows = (e.decisions || []).filter(d => d.options > 1 || d.randomized);
  const fixed = (e.decisions || []).filter(d => !(d.options > 1 || d.randomized));
  const box = el("div", {style: "display:grid;grid-template-columns:150px minmax(0,1fr) 120px;gap:14px 20px;align-items:center"});
  box.append(el("span", {class: "lab", style: "margin:0"}, "Decision"), el("span", {class: "lab", style: "margin:0"}, "Share of each choice so far"), el("span", {class: "lab", style: "margin:0"}, "How random"));
  rows.forEach(d => {
    const top = d.choices.slice(0, 6).map(([c, f]) => [d.decision === "island" ? "I" + c : c, f]), rest = 1 - top.reduce((a, [, f]) => a + f, 0);
    const bar = el("div", {style: "display:flex;height:22px;border-radius:5px;overflow:hidden;background:#1C1C1E"});
    top.forEach(([c, f], i) => bar.append(el("div", {title: c + " " + Math.round(f * 100) + "%", style: `width:${f * 100}%;background:${SEG[i % SEG.length]};opacity:.85;display:flex;align-items:center;padding-left:6px;overflow:hidden;white-space:nowrap;font-size:11px;color:#000`}, f > 0.09 ? c + " " + Math.round(f * 100) + "%" : "")));
    if (rest > 0.005) bar.append(el("div", {title: "other " + Math.round(rest * 100) + "%", style: `width:${rest * 100}%;background:#3A3A3C`}));
    const p = d.mean_propensity ?? 1, randomness = Math.max(0, Math.min(1, 1 - p));
    box.append(el("div", {}, el("b", {}, d.decision), el("div", {class: "muted", style: "font-size:11.5px"}, DECISIONS[d.decision] || "")), bar,
      el("div", {title: "mean probability of the choice taken: " + p.toFixed(2)}, el("div", {style: "height:6px;background:#1C1C1E;border-radius:3px;overflow:hidden"}, el("div", {style: `height:6px;width:${randomness * 100}%;background:var(--accent)`})),
        el("div", {class: "muted num", style: "font-size:11px;margin-top:4px"}, "p ≈ " + p.toFixed(2))));
  });
  return el("div", {}, box, fixed.length ? el("p", {class: "note", style: "margin-top:16px"}, "Not randomized in this run: " + fixed.map(d => d.decision + " = " + d.choices[0][0]).join(" · ")) : null);
}

function banditCharts(s, e) {
  const att = (s.attempts || []).filter(a => a.op), ops = [...new Set(att.map(a => a.op))], W = 640, H = 200, L = 36, R = 10, T = 8, B = 30;
  const win = Math.max(5, Math.round(att.length / 12)), buckets = [];
  for (let i = 0; i < att.length; i += win) { const part = att.slice(i, i + win), c = {}; part.forEach(a => c[a.op] = (c[a.op] || 0) + 1); buckets.push({n: i + part.length, c, t: part.length}); }
  let g = "";
  if (buckets.length) {
    const X = k => L + (buckets.length === 1 ? 0.5 : k / (buckets.length - 1)) * (W - L - R), Y = v => T + (1 - v) * (H - T - B);
    let base = buckets.map(() => 0);
    ops.forEach((op, j) => { const top = buckets.map((b, k) => base[k] + (b.c[op] || 0) / b.t);
      const pts = top.map((v, k) => `${X(k)},${Y(v)}`).concat(base.map((v, k) => `${X(k)},${Y(v)}`).reverse()).join(" ");
      g += `<polygon points="${pts}" fill="${SEG[j % SEG.length]}" fill-opacity=".8"/>`; base = top; });
    [0, 0.5, 1].forEach(v => g += tx(L - 6, Y(v) + 4, Math.round(v * 100) + "%", {a: "end"}));
    g += tx(L, H - 10, "start") + tx(W - R, H - 10, "attempt " + att.length, {a: "end"}) + tx((L + W - R) / 2, H - 10, "operator share, windows of " + win + " attempts", {a: "middle"});
  }
  const bandit = e.bandit || {}, arms = Object.entries(bandit), BW = 340, BH = 30 + arms.length * 34;
  let h = "";
  arms.forEach(([arm, v], k) => { const y = 14 + k * 34, w = (v.mean ?? 0) * (BW - 150);
    h += tx(0, y + 12, OPS[arm] || arm, {c: "#F5F5F7", size: 12}) + `<rect x="90" y="${y + 2}" width="${BW - 150}" height="14" rx="3" fill="#1C1C1E"/>`;
    h += v.mean === null ? tx(96, y + 13, "untried") : `<rect x="90" y="${y + 2}" width="${w}" height="14" rx="3" fill="${SEG[ops.indexOf(arm) >= 0 ? ops.indexOf(arm) % SEG.length : 6]}"/>` + tx(BW - 52, y + 13, v.mean.toFixed(2) + " · n " + v.n); });
  return el("div", {style: "display:grid;grid-template-columns:minmax(0,1.8fr) minmax(0,1fr);gap:0 40px;align-items:start"},
    el("div", {}, buckets.length ? svgBox(W, H, g, "Share of each operator over the run") : el("p", {class: "empty"}, "No operators used yet."), legend(ops.map((op, j) => [SEG[j % SEG.length], OPS[op] || op, false]))),
    el("div", {}, el("p", {class: "lab"}, "Mean recent reward"), svgBox(BW, BH, h, "Mean recent reward per operator")));
}

function modeTimeline(e) {
  const hist = e.mode_history || [];
  if (!hist.length) return el("p", {class: "empty"}, "No controller decisions yet.");
  const islands = [...new Set(hist.map(h => h[1]))].sort((a, b) => a - b), maxI = Math.max(1, ...hist.map(h => h[0] || 0));
  const W = 1000, L = 44, R = 10, row = 22, T = 6, H = T + islands.length * row + 30, X = i => L + (i / maxI) * (W - L - R), cw = Math.max(3, (W - L - R) / (maxI + 1) - 1);
  let g = "";
  islands.forEach((isl, k) => { const y = T + k * row; g += tx(L - 10, y + 14, "I" + isl, {a: "end", c: color(isl)});
    hist.filter(h => h[1] === isl).forEach(([i, , mode, plateau]) => {
      g += `<rect x="${X(i)}" y="${y + 3}" width="${cw}" height="${row - 8}" rx="2" fill="${mode === "exploit" ? "#0A84FF" : "#3A3A3C"}"/>`;
      if (plateau) g += `<rect x="${X(i)}" y="${y + row - 4}" width="${cw}" height="3" fill="#E8A33D"/>`; }); });
  g += tx(L, H - 6, "iteration 0") + tx(W - R, H - 6, maxI, {a: "end"});
  return el("div", {}, svgBox(W, H, g, "Prompt mode chosen per island over iterations"), legend([["#3A3A3C", "plain prompt", false], ["#0A84FF", "exploit prompt", false], ["#E8A33D", "island on a plateau"]]));
}

function swmChart(e) {
  const f = e.forecasts || {}, series = (f.series || []).filter(p => typeof p[1] === "number"), w = e.swm || {};
  if (!series.length) return el("p", {class: "empty"}, "No forecast has been scored yet.");
  const win = 10, roll = series.map((p, k) => { const part = series.slice(Math.max(0, k - win + 1), k + 1); return [p[0], part.reduce((a, q) => a + q[1], 0) / part.length]; });
  const W = 1000, H = 200, L = 50, R = 10, T = 8, B = 30, maxI = Math.max(1, ...series.map(p => p[0] || 0)), X = i => L + i / maxI * (W - L - R), Y = v => T + (1 - Math.min(1, v / 0.5)) * (H - T - B);
  let g = "";
  [0, 0.25, 0.5].forEach(v => g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#2C2C2E"/>` + tx(L - 6, Y(v) + 4, v, {a: "end"}));
  g += `<line x1="${L}" x2="${W - R}" y1="${Y(0.25)}" y2="${Y(0.25)}" stroke="#98989D" stroke-dasharray="4 4"/>` + tx(L + 8, Y(0.25) - 6, "coin flip (0.25)");
  series.forEach(p => g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.2" fill="${p[2] ? "#3DAE7C" : "#E5484D"}" fill-opacity=".7"/>`);
  g += `<polyline points="${roll.map(p => X(p[0]) + "," + Y(p[1])).join(" ")}" fill="none" stroke="#F5F5F7" stroke-width="2"/>`;
  if (w.trained_at !== null && w.trained_at !== undefined) { const xt = X(w.trained_at), right = xt > W * 0.75;
    g += `<line x1="${xt}" x2="${xt}" y1="${T}" y2="${H - B}" stroke="#0A84FF"/>` + tx(right ? xt - 6 : xt + 6, T + 12, "world model trained", {c: "#0A84FF", a: right ? "end" : "start"}); }
  g += tx(L, H - 8, "iteration 0") + tx(W - R, H - 8, maxI, {a: "end"}) + `<text x="10" y="${(T + H - B) / 2}" fill="#98989D" font-size="11" text-anchor="middle" transform="rotate(-90 10 ${(T + H - B) / 2})">Brier</text>`;
  return el("div", {}, svgBox(W, H, g, "Forecast error over iterations"),
    legend([["#F5F5F7", "Brier score, rolling over " + win + " forecasts (lower is better)"], ["#3DAE7C", "outcome inside the forecast interval", false], ["#E5484D", "outside it", false]]));
}

function nichesChart(e) {
  const n = e.niches || [];
  if (!n.length) return el("p", {class: "empty"}, "Niches have not been built yet.");
  const W = 1000, H = 170, L = 44, R = 10, T = 10, B = 26, maxI = Math.max(1, ...n.map(x => x.iteration || 0)), maxK = Math.max(...n.map(x => x.niches || 0), 1);
  const X = i => L + i / maxI * (W - L - R), Y = k => T + (1 - k / maxK) * (H - T - B);
  let d = `M${X(n[0].iteration)},${Y(n[0].niches)}`; n.slice(1).forEach((x, k) => d += ` L${X(x.iteration)},${Y(n[k].niches)} L${X(x.iteration)},${Y(x.niches)}`);
  let g = `<path d="${d}" fill="none" stroke="#0A84FF" stroke-width="2"/>` + n.map(x => `<circle cx="${X(x.iteration)}" cy="${Y(x.niches)}" r="3" fill="#0A84FF"/>`).join("");
  g += tx(L - 6, Y(maxK) + 4, maxK, {a: "end"}) + tx(L - 6, Y(0) + 4, 0, {a: "end"}) + tx(L, H - 6, "iteration 0") + tx(W - R, H - 6, maxI, {a: "end"});
  return el("div", {}, svgBox(W, H, g, "Number of niches over iterations"), el("p", {class: "note", style: "margin-top:8px"}, "niches in " + n[n.length - 1].space + " space · latent archive " + ((e.archive || {}).cells || 0) + " cells"));
}

function experimentCards(s, e) {
  const exp = e.experiments || [], it = s.run.iteration || 0;
  if (!exp.length) return el("p", {class: "empty"}, "No experiment yet.");
  return el("div", {style: "display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px"}, exp.slice().reverse().map(x => {
    const progress = x.outcome === "running" && x.length ? Math.min(1, (it - x.iteration) / x.length) : 1;
    const tone = x.outcome === "running" ? "busy" : x.outcome === "undecided" ? "muted" : "up";
    return el("div", {style: "border:1px solid var(--rule);border-radius:10px;padding:14px 16px"}, el("div", {class: tone, style: "font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase"}, x.outcome),
      el("p", {style: "margin:6px 0 10px"}, x.hypothesis), el("div", {style: "height:5px;background:#1C1C1E;border-radius:3px;overflow:hidden"}, el("div", {style: `height:5px;width:${progress * 100}%;background:${x.outcome === "running" ? "var(--accent)" : "#3A3A3C"}`})),
      el("div", {class: "muted", style: "font-size:11px;margin-top:6px"}, "started at iteration " + x.iteration + (x.length ? " · block of " + x.length + " steps" : "") + (x.required ? " · needs " + x.required + " rows" : "")));
  }));
}

function startStrip(e) {
  const st = e.start || {}, spawns = e.spawns || [];
  const chip = (title, sub, tone) => el("div", {style: `border:1px solid ${tone || "var(--rule)"};border-radius:10px;padding:12px 16px;min-width:150px`}, el("b", {}, title), el("div", {class: "muted", style: "font-size:12px"}, sub));
  const arrow = () => el("span", {class: "muted", style: "font-size:18px"}, "→");
  return el("div", {style: "display:flex;flex-wrap:wrap;align-items:center;gap:12px"},
    chip(st.cold ? "Cold start" : "Warm start", st.cold ? "from the task's seed" : "from " + (st.warm_start || st.inherit), st.cold ? null : "var(--accent)"), arrow(),
    chip(st.seed_programs + " seed measurements", "the baseline every island starts from"), arrow(),
    chip(st.portfolio ? st.portfolio_designs + " first designs" : "no portfolio", st.portfolio ? "one deliberately different design per island" : "islands start from the seed"), arrow(),
    chip(spawns.length + " islands opened later", spawns.length ? spawns.map(x => "I" + x.island).join(", ") + " after stalls of " + [...new Set(spawns.map(x => x.stall))].join("/") : "none so far"),
    chip("ε " + (st.epsilon ?? "—") + " · " + (st.inherit ? "inherited culture" : "no inherited culture"), "share of choices kept random"));
}

function rlCard(e) {
  const pol = e.policy || {}, fits = e.policy_fits || [], moved = Object.entries(pol.moved || {});
  const planner = (e.decisions || []).find(d => d.decision === "planner");
  const share = planner ? planner.choices.map(([c, f]) => c + " " + Math.round(f * 100) + "%").join(" · ") : "—";
  if (!moved.length) return el("div", {style: "border:1px solid var(--rule);border-radius:10px;padding:16px 18px"}, el("b", {}, fits.length ? "Fitted, not yet steering" : "Not active in this run"),
    el("p", {class: "note", style: "margin:6px 0 0"}, (fits.length ? fits.length + " fits; the last on " + fits[fits.length - 1].steps + " steps" + (fits[fits.length - 1].reason ? " (" + fits[fits.length - 1].reason + ")" : "") + ". " : "The policy has not been fitted yet. ") + "Who steered the steps: " + share + ". " + (e.stand_downs ? e.stand_downs + " stand-downs." : "")));
  const W = 520, H = 20 + moved.length * 28, max = Math.max(...moved.map(([, v]) => Math.abs(v)), 0.01);
  let g = ""; moved.forEach(([d, v], k) => { const y = 10 + k * 28; g += tx(0, y + 12, d, {c: "#F5F5F7"}) + `<rect x="150" y="${y + 2}" width="${Math.abs(v) / max * (W - 220)}" height="14" rx="3" fill="#0A84FF"/>` + tx(W - 60, y + 13, v.toFixed(2)); });
  return el("div", {}, svgBox(W, H, g, "How far the RL policy moved each decision"), el("p", {class: "note"}, "Distance of the RL policy from the base distribution, per decision. Who steered: " + share + "."));
}

function engineTab(s) {
  const e = s.mechanics || {}, box = document.getElementById("engine-view");
  box.replaceChildren(
    section("How the run started", "Where the search began, and how the islands were seeded.", startStrip(e)),
    section("Islands over the run", "Every attempt of every island, in order. A larger green dot is an improvement on its parent. Kept islands survived successive halving.", islandTimeline(s, e)),
    section("Randomized decisions", "Each step draws these choices from logged distributions, which makes the run an experiment on itself. The bar is how often each choice was taken; the blue gauge is how random the choices were (1 − mean probability of the choice taken).", decisionBars(e)),
    section("Operator bandit", "The bandit samples operators by the recent reward of their children. Left: does the mix move toward operators that work? Right: the reward it currently sees.", banditCharts(s, e)),
    section("Mode controller", "Per island, whether the step used the plain or the exploit prompt; the controller shifts toward exploit on a plateau.", modeTimeline(e),
      el("p", {class: "note", style: "margin-top:10px"}, (e.intensify || 0) + " refine intensifications · " + (e.plateaus || 0) + " plateaus detected")),
    section("RL step policy", "A policy learned from lineage returns that may tilt the randomized decisions when the planner hands it the step.", rlCard(e)),
    section("Search world model", "Before each child is scored the search forecasts whether it will improve. Lower Brier is better; 0.25 is what a coin flip scores.", swmChart(e)),
    section("Niches", "Programs are grouped into vector-quantized niches so islands keep diverse programs instead of near-copies.", nichesChart(e)),
    section("Theory experiments", "Hypotheses about the search's own decisions, tested by randomizing the condition in blocks.", experimentCards(s, e)));
}
