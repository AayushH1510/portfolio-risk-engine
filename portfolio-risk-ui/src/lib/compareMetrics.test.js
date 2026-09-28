// Regression tests for compareMetrics.js's winner() — the single source of
// truth for "which side wins" a head-to-head metric on the Compare tab.
//
// Covers the exact bug this file fixes: Drawdown/VaR 95%/CVaR 95% are
// stored as negative numbers (a loss), and a naive `lowerBetter` flag
// applied to the raw signed value used to treat "more negative" (a bigger
// loss) as "better" — the opposite of correct. Each loss-metric case below
// uses the reported bug's own numbers where practical.
//
// Uses Node's built-in test runner (node:test) — no test framework was
// already a dependency of this project (see package.json), and this ships
// with Node itself, same "stdlib is enough" call as test_finnhub_reliability
// .py's on the backend.
//
// Run: node --test src/lib/compareMetrics.test.js

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { winner, METRIC_DIRECTIONS } from './compareMetrics.js'

// ── Naturally-signed metrics: higher raw value = better ────────────────────

test('return: higher raw value wins, positive values', () => {
  assert.equal(winner(0.12, 0.08, 'return'), 'A')
  assert.equal(winner(0.08, 0.12, 'return'), 'B')
})

test('return: higher raw value wins, negative values (a smaller loss beats a bigger loss)', () => {
  assert.equal(winner(-0.02, -0.10, 'return'), 'A')
  assert.equal(winner(-0.10, -0.02, 'return'), 'B')
})

test('return: tie', () => {
  assert.equal(winner(0.10, 0.10, 'return'), null)
})

test('sharpe: higher raw value wins', () => {
  assert.equal(winner(1.8, 1.2, 'sharpe'), 'A')
  assert.equal(winner(1.2, 1.8, 'sharpe'), 'B')
  assert.equal(winner(-0.3, -0.9, 'sharpe'), 'A')
  assert.equal(winner(1.5, 1.5, 'sharpe'), null)
})

test('sortino: higher raw value wins', () => {
  assert.equal(winner(2.1, 1.4, 'sortino'), 'A')
  assert.equal(winner(1.4, 2.1, 'sortino'), 'B')
  assert.equal(winner(0.9, 0.9, 'sortino'), null)
})

test('alpha: higher raw value wins, including negative-vs-positive', () => {
  assert.equal(winner(0.03, -0.01, 'alpha'), 'A')
  assert.equal(winner(-0.01, 0.03, 'alpha'), 'B')
  assert.equal(winner(0.0, 0.0, 'alpha'), null)
})

// ── Positive-magnitude metrics: lower raw value = better ────────────────────

test('volatility: lower raw value wins', () => {
  assert.equal(winner(0.14, 0.22, 'volatility'), 'A')
  assert.equal(winner(0.22, 0.14, 'volatility'), 'B')
  assert.equal(winner(0.18, 0.18, 'volatility'), null)
})

test('beta: lower raw value wins', () => {
  assert.equal(winner(0.9, 1.4, 'beta'), 'A')
  assert.equal(winner(1.4, 0.9, 'beta'), 'B')
  assert.equal(winner(1.1, 1.1, 'beta'), null)
})

// ── Loss metrics expressed as negatives: smaller loss (less negative /
//    higher raw value) wins — the bug this file exists to fix ──────────────

test('drawdown: smaller loss wins (reported bug case: -6.6% vs -4.7%)', () => {
  assert.equal(winner(-0.066, -0.047, 'drawdown'), 'B')
  assert.equal(winner(-0.047, -0.066, 'drawdown'), 'A')
})

test('drawdown: tie', () => {
  assert.equal(winner(-0.05, -0.05, 'drawdown'), null)
})

test('var95: smaller loss wins (reported bug case: -1.5% vs -1.4%)', () => {
  assert.equal(winner(-0.015, -0.014, 'var95'), 'B')
  assert.equal(winner(-0.014, -0.015, 'var95'), 'A')
})

test('var95: tie', () => {
  assert.equal(winner(-0.02, -0.02, 'var95'), null)
})

test('cvar95: smaller loss wins (reported bug case: -2.2% vs -1.6%)', () => {
  assert.equal(winner(-0.022, -0.016, 'cvar95'), 'B')
  assert.equal(winner(-0.016, -0.022, 'cvar95'), 'A')
})

test('cvar95: tie', () => {
  assert.equal(winner(-0.03, -0.03, 'cvar95'), null)
})

// ── Missing / non-numeric inputs never crash and never declare a winner ────

test('null or missing values return null rather than a winner', () => {
  assert.equal(winner(null, -0.05, 'drawdown'), null)
  assert.equal(winner(-0.05, null, 'drawdown'), null)
  assert.equal(winner(undefined, undefined, 'return'), null)
})

// ── Every metric consumed by Comparison.jsx must have a direction defined ──

test('every metric direction table entry is well-formed', () => {
  for (const [key, dir] of Object.entries(METRIC_DIRECTIONS)) {
    assert.equal(typeof dir.label, 'string', `${key}.label should be a string`)
    assert.equal(typeof dir.score, 'function', `${key}.score should be a function`)
  }
})

test('unknown metric key throws rather than silently defaulting a direction', () => {
  assert.throws(() => winner(1, 2, 'not_a_real_metric'))
})
