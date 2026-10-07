// Regression tests for rateLimit.js — which backend responses become the
// friendly busy state. Everything not matched here must keep its raw message.
// Uses Node's built-in test runner, same as compareMetrics.test.js.

import test from 'node:test'
import assert from 'node:assert/strict'
import { isRateLimited, isRateLimitError } from './rateLimit.js'

test('503 with the provider rate-limit message is rate limited', () => {
  assert.equal(isRateLimited(503, "The market data provider's rate limit may have been reached. Please try again in a minute."), true)
  assert.equal(isRateLimited(503, "Unable to fetch data for: AAPL. The market data provider's rate limit may have been reached. Please try again in a minute."), true)
  assert.equal(isRateLimited(503, 'Finnhub rate limit reached (60 calls/minute). Please try again in a minute.'), true)
})

test('429 from the per-IP limiter is rate limited', () => {
  assert.equal(isRateLimited(429, 'Too many requests (20 per 1 minute). Please wait a moment and try again.'), true)
})

test('other 503s keep their own message', () => {
  assert.equal(isRateLimited(503, 'Something went wrong processing your request. Please try again.'), false)
})

test('other statuses and details are not rate limited', () => {
  assert.equal(isRateLimited(404, 'Ticker not found: ZZZZ'), false)
  assert.equal(isRateLimited(500, 'rate limit'), false)
  assert.equal(isRateLimited(503, undefined), false)
  assert.equal(isRateLimited(503, [{ msg: 'rate limit' }]), false)
  assert.equal(isRateLimited(undefined, undefined), false)
})

test('isRateLimitError reads an axios-shaped error', () => {
  const err = { response: { status: 503, data: { detail: 'The market data provider\'s rate limit may have been reached.' } } }
  assert.equal(isRateLimitError(err), true)
  assert.equal(isRateLimitError({ response: { status: 503, data: { detail: 'Something went wrong.' } } }), false)
  assert.equal(isRateLimitError({}), false)
})
