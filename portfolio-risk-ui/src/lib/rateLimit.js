// The market-data provider's free allowance running out: the backend answers
// 503 with a message that says so (api.py's analyse/stress-test paths, and
// stock_detail_route.py's Finnhub path). The app's own per-IP limiter answers
// 429. These are the only responses shown as a busy state; every other error
// keeps the message the backend sent.
export function isRateLimited(status, detail) {
  if (status === 429) return true
  return status === 503 && typeof detail === 'string' && /rate limit/i.test(detail)
}

export function isRateLimitError(err) {
  return isRateLimited(err.response?.status, err.response?.data?.detail)
}
