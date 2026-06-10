import fs from 'fs';
import {
  createTerm, createFraction,
  differentiateFraction, integrateFraction,
  definiteIntegrateFraction, evaluateFraction
} from '../../SLaNg/slang-basic.js';
import {
  productRuleDifferentiate,
  quotientRuleDifferentiate,
  gradient
} from '../../SLaNg/slang-advanced.js';

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randCoeff() {
  const choices = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5];
  return randChoice(choices);
}

function randomTerms(variable, maxTerms = 3) {
  const count = randInt(1, maxTerms);
  const terms = [];
  const usedExps = new Set();
  for (let i = 0; i < count; i++) {
    let exp = randInt(0, 4);
    while (usedExps.has(exp)) exp = randInt(0, 4);
    usedExps.add(exp);
    if (exp === 0) terms.push(createTerm(randCoeff()));
    else terms.push(createTerm(randCoeff(), { [variable]: exp }));
  }
  return terms;
}

function randomFraction(variable = 'x') {
  const numi = randomTerms(variable, 3);
  if (Math.random() < 0.6) {
    return createFraction(numi, [createTerm(1)]);
  }
  return createFraction(numi, randomTerms(variable, 2));
}

function randomMultivarFraction(vars = ['x', 'y']) {
  const coeff1 = randCoeff();
  const coeff2 = randCoeff();
  const p1 = {}, p2 = {};
  for (const v of vars) { p1[v] = randInt(0, 2); p2[v] = randInt(0, 2); }
  return createFraction(
    [createTerm(coeff1, p1), createTerm(coeff2, p2)],
    [createTerm(1)]
  );
}

function classifyRule(expr, variable) {
  const numiTerms = expr.numi?.terms || [];
  const denoTerms = expr.deno?.terms || [];
  const trivialDeno = denoTerms.length === 1 && !denoTerms[0].var?.[variable];
  if (!trivialDeno) return { rule: 'quotient_rule', desc: 'Apply quotient rule: d/dx[u/v] = (v*u\' - u*v\')/v^2' };
  if (numiTerms.length > 1) return { rule: 'sum_rule', desc: 'Differentiate each term separately (sum rule)' };
  const exp = numiTerms[0]?.var?.[variable] || 0;
  if (exp === 0) return { rule: 'constant_rule', desc: 'Derivative of a constant is 0' };
  return { rule: 'power_rule', desc: `Apply power rule: d/dx[x^${exp}] = ${exp}x^${exp-1}` };
}

function numericallyEqual(exprA, exprB, variables, tol = 1e-9, points = 50) {
  let validCount = 0;
  for (let i = 0; i < points; i++) {
    const pt = {};
    for (const v of variables) pt[v] = (Math.random() * 9.6 - 4.8) + 0.01;
    try {
      const a = evaluateFraction(exprA, pt);
      const b = evaluateFraction(exprB, pt);
      if (!isFinite(a) || !isFinite(b)) continue;
      
      validCount++;
      
      if (Math.abs(a - b) > tol) return false;
    } catch { continue; }
  }
  
  if (validCount === 0) return false;
  return true;
}

function genDiff() {
  const expr = randomFraction('x');
  const output = differentiateFraction(expr, 'x');
  const { rule, desc } = classifyRule(expr, 'x');
  return {
    input: { op: 'diff', var: 'x', expr },
    output: { expr: output, rule, description: desc, steps: [{ rule, description: desc }] },
    verify: () => numericallyEqual(output, differentiateFraction(expr, 'x'), ['x'])
  };
}

function genIntegrate() {
  const expr = createFraction(randomTerms('x', 3), [createTerm(1)]);
  const output = integrateFraction(expr, 'x');
  const rule = 'power_rule_integral';
  const desc = 'Apply reverse power rule: x^n dx = x^(n+1)/(n+1)';
  return {
    input: { op: 'integrate', var: 'x', expr },
    output: { expr: output, rule, description: desc, steps: [{ rule, description: desc }] },
    verify: () => numericallyEqual(output, integrateFraction(expr, 'x'), ['x'])
  };
}

function genProductRule() {
  const u = createFraction(randomTerms('x', 2), [createTerm(1)]);
  const v = createFraction(randomTerms('x', 2), [createTerm(1)]);
  const output = productRuleDifferentiate([u, v], 'x');
  const rule = 'product_rule';
  const desc = 'Apply product rule: d/dx[u*v] = u\'*v + u*v\'';
  return {
    input: { op: 'product_rule', var: 'x', u, v },
    output: { expr: output, rule, description: desc, steps: [{ rule, description: desc }] },
    verify: () => numericallyEqual(output, productRuleDifferentiate([u, v], 'x'), ['x'])
  };
}

function genQuotientRule() {
  const u = createFraction(randomTerms('x', 2), [createTerm(1)]);
  const v = createFraction(randomTerms('x', 2), [createTerm(1)]);
  const output = quotientRuleDifferentiate(u, v, 'x');
  const rule = 'quotient_rule';
  const desc = 'Apply quotient rule d/dx[u/v] = (v*u\' - u*v\')/v^2';
  return {
    input: { op: 'quotient_rule', var: 'x', u, v },
    output: { expr: output, rule, description: desc, steps: [{ rule, description: desc }] },
    verify: () => numericallyEqual(output, quotientRuleDifferentiate(u, v, 'x'), ['x'])
  };
}

function genGradient() {
  const expr = randomMultivarFraction(['x', 'y']);
  const grad = gradient(expr, ['x', 'y']);
  const rule = 'partial_derivative';
  const desc = 'Compute df/dx and df/dy, treating other variables as constants';
  return {
    input: { op: 'gradient', vars: ['x', 'y'], expr },
    output: { expr: grad, rule, description: desc, steps: [{ rule, description: desc }] },
    verify: () => {
      const realGrad = gradient(expr, ['x', 'y']);
      return numericallyEqual(grad.x, realGrad.x, ['x', 'y']) &&
             numericallyEqual(grad.y, realGrad.y, ['x', 'y']);
    }
  };
}

const GENERATORS = [
  { fn: genDiff, weight: 4 },
  { fn: genIntegrate, weight: 2 },
  { fn: genProductRule, weight: 2 },
  { fn: genQuotientRule, weight: 1 },
  { fn: genGradient, weight: 1 }
];

function pickGen() {
  const total = GENERATORS.reduce((s, g) => s + g.weight, 0);
  let cur = Math.random() * total;
  for (const g of GENERATORS) {
    cur -= g.weight;
    if (cur <= 0) return g.fn;
  }
  return GENERATORS[0].fn;
}

async function run(count, outputPath) {
  console.log(`Generating ${count} verified SLaNg training pairs -> ${outputPath}\n`);
  const stream = fs.createWriteStream(outputPath, { flags: 'w' });
  let saved = 0, skipped = 0;
  while (saved < count) {
    let pair;
    try { pair = pickGen()(); } catch { skipped++; continue; }
    let ok = false;
    try { ok = pair.verify(); } catch {}
    if (!ok) { skipped++; continue; }
    stream.write(JSON.stringify({ input: pair.input, output: pair.output, verified: true }) + '\n');
    saved++;
    if (saved % 1000 === 0) {
      process.stdout.write(`${saved}/${count} saved (${skipped} skipped)\r`);
    }
  }
  stream.end();
  console.log(`\nDone. ${saved} saved, ${skipped} skipped.\nOutput: ${outputPath}`);
}

const args = process.argv.slice(2);
const count = parseInt(args[args.indexOf('--count') + 1] || '1000', 10);
const outPath = args[args.indexOf('--out') + 1] || 'data/slang_dataset.jsonl';

if (process.argv[1].endsWith('generate_slang_data.js')) {
  run(count, outPath).catch(err => { console.error(err); process.exit(1); });
}