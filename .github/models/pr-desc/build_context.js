// .github/models/pr-desc/build_context.js
// This script assembles the context that is fed to the PR summarization model.
// Every section below adds a different type of hint (diff, commit messages,
// module docs, etc.) so the model can understand which parts of the repo the
// PR is touching.
const { execSync } = require('child_process');
const { readFileSync, readdirSync, existsSync } = require('fs');
const fs = require('fs');
const { join } = require('path');

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return (i > -1 && process.argv[i + 1]) ? process.argv[i + 1] : def;
}
const base = arg('--base', 'origin/main');
const head = arg('--head', 'HEAD');

function sh(cmd) {
  try { return execSync(cmd, { encoding: 'utf8' }).trim(); }
  catch { return ''; }
}
function clip(s, max) {
  if (!s) return '';
  return s.length <= max ? s : s.slice(0, max) + `\n\n[truncated ${s.length - max} chars]`;
}

// ---------- 0) Rule Loading ----------
const rulesPath = '.github/models/pr-desc/rules.json';
let rules = { modules: [], fallbackModule: 'Misc' };
if (existsSync(rulesPath)) {
  try { rules = JSON.parse(readFileSync(rulesPath, 'utf8')); }
  catch { /* ignore parse error; keep defaults */ }
}
const compiledPatterns = rules.modules.map(m => ({
  name: m.name,
  weight: Number(m.weight || 1),
  regs: (m.patterns || []).map(p => new RegExp(p))
}));

function classifyModule(filepath) {
  for (const m of compiledPatterns) {
    if (m.regs.some(r => r.test(filepath))) return { module: m.name, weight: m.weight };
  }
  return { module: rules.fallbackModule || 'Misc', weight: 1 };
}

// Utility helpers for normalising and scoring keywords that describe
// "interesting" parts of the change. These are later used to selectively load
// module documentation.
function normalizeToken(token) {
  return token ? token.toLowerCase().replace(/[^a-z0-9]/g, '') : '';
}
const interestTokens = new Map();
function bumpToken(token, weight = 1) {
  const norm = normalizeToken(token);
  if (!norm) return;
  interestTokens.set(norm, (interestTokens.get(norm) || 0) + weight);
}

// ---------- 1) Overview / Modules Source Documents ----------
const ctxRoot = '.github/models/pr-desc/context';
let overview = '';
const overviewPath = join(ctxRoot, 'overview.md');
if (existsSync(overviewPath)) {
  overview = readFileSync(overviewPath, 'utf8');
}
overview = clip(overview, 8000);

// ---------- 2) Git diff/numstat ----------
const nameStatusRaw = sh(`git diff --name-status -M -C ${base}...${head}`);
const statRaw = sh(`git diff --stat ${base}...${head}`);
const numstatRaw = sh(`git diff --numstat -M -C ${base}...${head}`);
const commitSubjects = sh(`git log --pretty=%s ${base}..${head}`).split('\n').filter(Boolean);
const commitBodiesRaw = sh(`git log --pretty=%B ${base}..${head}`);
const commitBodies = commitBodiesRaw.split('\n\n').map(s => s.trim()).filter(Boolean).slice(0, 10).map(s => clip(s, 800));

// name-status parsing (status, path[, path2])
// M A D R100 old -> new format is separated by tabs
const changedFiles = [];
if (nameStatusRaw) {
  for (const line of nameStatusRaw.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    const status = parts[0]; // e.g., 'M', 'A', 'D', 'R100'
    const from = parts[1];
    const to = parts[2] || parts[1];
    changedFiles.push({ status, from, to, path: to });
    // Directory / filename tokens tell us which doc hints are relevant.
    const segments = to.split('/').slice(0, 3); // favour the upper path for module hints
    segments.forEach((seg, idx) => bumpToken(seg, Math.max(1, 3 - idx)));
    const baseName = to.split('/').pop();
    if (baseName) {
      const [stem] = baseName.split('.');
      bumpToken(stem, 1);
    }
  }
}

// numstat parsing (added removed path[, path2])
const churnMap = new Map(); // key: path(to), value: {added, removed}
if (numstatRaw) {
  for (const line of numstatRaw.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    if (parts.length < 3) continue;
    const added = parts[0] === '-' ? 0 : parseInt(parts[0], 10) || 0;
    const removed = parts[1] === '-' ? 0 : parseInt(parts[1], 10) || 0;
    const pth = parts[2].includes('\t') ? parts[3] : parts[2]; // for rename path\tpath2
    const path = parts[3] || parts[2];
    churnMap.set(path, { added, removed });
  }
}

// === A. build changedFiles FIRST ===
// parse name-status/numstat/etc. (existing code that fills changedFiles)
// ensure you have something like:
// const changedFiles = []; // declare before pushing to it
// ... push { path, status, additions, deletions, ... } into changedFiles

// === B. relevance-driven module docs (SAFE: changedFiles is ready) ===
const modulesDir = join(ctxRoot, 'modules');
let modulesDoc = '';
if (existsSync(modulesDir)) {
  // Step 1: module classifier votes. We bump interest tokens with the module
  // names so docs such as compiler.md or layers.md get a high score if a file
  // belonging to that module changed.
  for (const f of changedFiles) {
    const cls = classifyModule(f.path);
    if (cls && cls.module) {
      bumpToken(cls.module, 4 * (cls.weight || 1));
    }
  }

  // Step 2: commit messages occasionally mention the component (“dataset”,
  // “optimizers”). We extract keywords to guide doc selection when diff paths
  // alone are not conclusive.
  for (const subj of commitSubjects) {
    const words = subj.toLowerCase().match(/[a-z0-9]{4,}/g) || [];
    for (const w of words) bumpToken(w, 0.5);
  }

  const allMd = readdirSync(modulesDir).filter(f => f.endsWith('.md'));
  const docScores = allMd.map(file => {
    const slug = normalizeToken(file.replace(/\.md$/, ''));
    let score = 0;
    if (interestTokens.has(slug)) score += interestTokens.get(slug) * 2;
    for (const [token, val] of interestTokens.entries()) {
      if (token && slug.includes(token) && token !== slug) score += val;
    }
    return { file, score };
  }).sort((a, b) => b.score - a.score);

  const pickedDocs = docScores.filter(d => d.score > 0).map(d => d.file).slice(0, 8);
  const fallbackDocs = pickedDocs.length ? pickedDocs : docScores.slice(0, 3).map(d => d.file);

  for (const f of fallbackDocs) {
    const body = readFileSync(join(modulesDir, f), 'utf8');
    modulesDoc += `\n\n## ${f}\n` + clip(body, 6000);
  }
}
modulesDoc = clip(modulesDoc, 24000);



function statusWeight(status) {
  // Rxxx, Cxxx, etc. are considered renames/copies
  if (status.startsWith('R') || status.startsWith('C')) return 2.0;
  if (status === 'D') return 2.5;
  if (status === 'A') return 1.5;
  // Default M
  return 1.0;
}

// ---------- 3) Module Grouping + Impact Calculation ----------
const modulesAgg = new Map(); // name -> { files:[], score:0, weight, counts, lines }
const unmatched = [];
for (const f of changedFiles) {
  const cls = classifyModule(f.path);
  if (!cls || !cls.module) { unmatched.push(f.path); continue; }
  const key = cls.module;
  if (!modulesAgg.has(key)) {
    modulesAgg.set(key, {
      files: [],
      score: 0,
      baseWeight: cls.weight,
      count: 0,
      adds: 0,
      dels: 0,
      hasRename: false,
      hasDelete: false
    });
  }
  const agg = modulesAgg.get(key);
  agg.files.push({ path: f.path, status: f.status });
  agg.count += 1;

  const churn = churnMap.get(f.path) || { added: 0, removed: 0 };
  agg.adds += churn.added;
  agg.dels += churn.removed;
  if (f.status.startsWith('R') || f.status.startsWith('C')) agg.hasRename = true;
  if (f.status === 'D') agg.hasDelete = true;

  // score = (module weight) * (status weight) * (size weight)
  const sizeFactor = Math.log10(1 + churn.added + churn.removed + 1); // 0~approximately 4
  agg.score += cls.weight * statusWeight(f.status) * (1 + sizeFactor);
}

// Final impact level determination by module
function levelFromScore(s) {
  if (s >= 30) return 'High';
  if (s >= 12) return 'Medium';
  return 'Low';
}
function bumpForSignals(agg) {
  let bonus = 0;
  if (agg.hasDelete) bonus += 2.0;
  if (agg.hasRename) bonus += 1.0;
  if (agg.count >= 10) bonus += 1.5;
  const churn = agg.adds + agg.dels;
  if (churn >= 500) bonus += 2.0;
  else if (churn >= 200) bonus += 1.0;
  return bonus;
}
const moduleImpact = {};
for (const [name, agg] of modulesAgg.entries()) {
  const score = agg.score + bumpForSignals(agg);
  moduleImpact[name] = {
    impact: levelFromScore(score),
    score: Math.round(score * 10) / 10,
    files: agg.files,
    stats: { files: agg.count, added: agg.adds, removed: agg.dels,
             rename: agg.hasRename, delete: agg.hasDelete }
  };
}
// extra reviewer signals (place AFTER changedFiles built)
function headerOrConfig(p){
 return /\.(h|hpp|hh|hxx|inc)$/.test(p) ||
 /(^|\/)(CMakeLists\.txt|configure|.*\.cmake|.*\.bazel|build\.gradle|settings\.gradle|package\.json)$/.test(p);
}
const apiSurfaceChanges = changedFiles.filter(f => headerOrConfig(f.path)).map(f => f.path);
const testFiles = changedFiles.filter(f => /(^|\/)(test|tests|testing|spec)\b|_test\.(cc|cpp|c|py|js|ts)$/.test(f.path)).map(f => f.path);
const concurrencySensitive= changedFiles.filter(f => /(thread|mutex|atomic|lock|concurrent|parallel)/i.test(f.path)).map(f => f.path);

// ---------- 4) Diff/Commits Text ----------
const diff = clip(`### name-status\n${nameStatusRaw}\n\n### stat\n${statRaw}`, 8000);

const buckets = { feat:0, fix:0, refactor:0, test:0, docs:0, chore:0, other:0 };
for (const s of commitSubjects) {
  const t = s.toLowerCase();
  if (t.startsWith('feat')) buckets.feat++;
  else if (t.startsWith('fix')) buckets.fix++;
  else if (t.startsWith('refactor')) buckets.refactor++;
  else if (t.startsWith('test')) buckets.test++;
  else if (t.startsWith('docs')) buckets.docs++;
  else if (t.startsWith('chore')) buckets.chore++;
  else buckets.other++;
}
const commits =
 `Total commits: ${commitSubjects.length}\n` +
  Object.entries(buckets).map(([k,v]) => `- ${k}: ${v}`).join('\n') +
 (commitSubjects.length ? `\n\nSamples:\n- ${commitSubjects.slice(0,5).join('\n- ')}` : '') +
 (commitBodies.length ? `\n\nCommit bodies (top, clipped):\n- ${commitBodies.join('\n- ')}` : '');

// ---------- 5) Module Impact Summary Text (Model Hint) ----------
let moduleImpactSummary = '';
if (Object.keys(moduleImpact).length) {
  // Top 5 modules sorted by highest impact
  const top = Object.entries(moduleImpact)
    .sort((a,b) => b[1].score - a[1].score)
    .slice(0,5);
  moduleImpactSummary = top.map(([name, m]) =>
    `- ${name}: impact=${m.impact} (score=${m.score}, files=${m.stats.files}, +${m.stats.added}/-${m.stats.removed}${m.stats.rename?', rename':''}${m.stats.delete?', delete':''})`
  ).join('\n');
}

// ---------- 6) Output ----------
const out = {
  overview: clip(overview, 8000),
  modules: modulesDoc,
  diff,
  commits,
  // New fields for enhanced context
  moduleImpact, // Per-module details (machine-readable)
  moduleImpactSummary: moduleImpactSummary || '(no module impact detected)',
 reviewerSignals: {
 apiSurfaceChanges,
 testFiles,
 concurrencySensitive
 }
};

process.stdout.write(JSON.stringify(out, null, 2));
