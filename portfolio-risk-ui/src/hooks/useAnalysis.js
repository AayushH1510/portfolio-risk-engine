import { useState, useCallback, useRef } from 'react'
import axios from 'axios'
import { format, subMonths, subYears } from 'date-fns'
import { errorMessage } from '../lib/errorMessage'
import { isRateLimitError } from '../lib/rateLimit'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const periodMap = {
  '1M':  () => subMonths(new Date(), 1),
  '3M':  () => subMonths(new Date(), 3),
  '6M':  () => subMonths(new Date(), 6),
  '1Y':  () => subYears(new Date(), 1),
  '3Y':  () => subYears(new Date(), 3),
  '5Y':  () => subYears(new Date(), 5),
  'Max': () => subYears(new Date(), 10),
}

const fmt = (d) => format(d, 'yyyy-MM-dd')

export function useAnalysis() {
  const [tickers, setTickers]           = useState(['AAPL', 'MSFT', 'GOOGL'])
  const [weights, setWeights]           = useState([0.34, 0.33, 0.33])
  const [period, setPeriod]             = useState('1Y')
  const [portfolioValue, setPortfolioValue] = useState(10000)
  const [benchmark, setBenchmark]           = useState('SPY')   // ticker, or null for "None"
  const [rollingWindow, setRollingWindow]   = useState(30)

  const [data, setData]         = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [hasRun, setHasRun]     = useState(false)

  // The heavy tier (Monte Carlo, Efficient Frontier, backtest) fetches in
  // the background right after the fast summary resolves — hasRun/loading
  // above track the fast tier only, so the Dashboard renders the moment
  // /api/analyse-summary comes back instead of waiting on the full
  // simulation too. These track that second, slower call separately so
  // Monte Carlo/Frontier/Backtest can show their own scoped loading state
  // instead of blocking the whole app.
  const [heavyLoading, setHeavyLoading] = useState(false)
  const [heavyError, setHeavyError]     = useState(null)

  // True when the error above is the market-data rate limit (see lib/rateLimit.js),
  // so the UI can show a busy state with a retry instead of the raw message.
  const [rateLimited, setRateLimited]           = useState(false)
  const [heavyRateLimited, setHeavyRateLimited] = useState(false)

  // The last run's arguments and request body, so "Try again" re-sends exactly
  // the same request without re-reading (or resetting) any input.
  const lastRunArgs = useRef({ customDates: null, onSuccess: undefined })
  const lastPayload = useRef(null)

  const runAnalysis = useCallback(async (customDates = null, onSuccess) => {
    setLoading(true)
    setError(null)
    setRateLimited(false)
    setHeavyError(null)
    setHeavyRateLimited(false)
    lastRunArgs.current = { customDates, onSuccess }

    const startDate = customDates ? customDates.start : fmt(periodMap[period]())
    const endDate   = customDates ? customDates.end   : fmt(new Date())

    const payload = {
      tickers,
      weights,
      start_date:      startDate,
      end_date:        endDate,
      portfolio_value: portfolioValue,
      show_benchmark:  benchmark !== null,
      benchmark:       benchmark || 'SPY',
      rolling_window:  rollingWindow,
    }
    lastPayload.current = payload

    try {
      const res = await axios.post(`${API}/api/analyse-summary`, payload)
      setData(res.data)
      setHasRun(true)
      onSuccess?.()

      // Not awaited — the summary render above already happened. Fires
      // immediately, not on-demand per tab click, so Monte Carlo/Frontier/
      // Backtest are usually already loaded (or loading) by the time
      // someone navigates there. /api/analyse-full's response is a
      // superset of the summary (identical fast-tier numbers, see api.py's
      // shared _serialize_fast_tier), so replacing data wholesale here is
      // safe — nothing already on screen changes, it only gains fields.
      setHeavyLoading(true)
      axios.post(`${API}/api/analyse-full`, payload)
        .then(fullRes => setData(fullRes.data))
        .catch(err => {
          setHeavyRateLimited(isRateLimitError(err))
          setHeavyError(errorMessage(err, 'Something went wrong loading the full simulation. Please try again in a moment.'))
        })
        .finally(() => setHeavyLoading(false))

    } catch (err) {
      // errorMessage() surfaces the backend's actual detail whenever the
      // server responded at all (invalid-ticker 404, rate-limit 503, or our
      // own generic 500 all carry their own accurate message already — see
      // api.py). This fallback only fires when there was no response to
      // read a detail from at all — a real network failure, a timeout, or
      // the request never reaching the server — so it must not imply the
      // user did anything wrong; the tickers may be completely fine.
      setRateLimited(isRateLimitError(err))
      setError(errorMessage(err, 'Something went wrong loading data. Please try again in a moment.'))
    } finally {
      setLoading(false)
    }
  }, [tickers, weights, period, portfolioValue, benchmark, rollingWindow])

  // Re-runs the last Run Analysis exactly as it was called.
  const retryRun = useCallback(() => {
    const { customDates, onSuccess } = lastRunArgs.current
    return runAnalysis(customDates, onSuccess)
  }, [runAnalysis])

  // Re-sends only the heavy-tier request (/api/analyse-full) with the last
  // run's body. The fast summary already on screen is left as it is.
  const retryHeavy = useCallback(() => {
    const payload = lastPayload.current
    if (!payload) return
    setHeavyLoading(true)
    setHeavyError(null)
    setHeavyRateLimited(false)
    axios.post(`${API}/api/analyse-full`, payload)
      .then(fullRes => setData(fullRes.data))
      .catch(err => {
        setHeavyRateLimited(isRateLimitError(err))
        setHeavyError(errorMessage(err, 'Something went wrong loading the full simulation. Please try again in a moment.'))
      })
      .finally(() => setHeavyLoading(false))
  }, [])

  const updateWeight = useCallback((index, value) => {
    setWeights(prev => {
      const next = [...prev]
      next[index] = value
      return next
    })
  }, [])

  const updateTickers = useCallback((newTickers) => {
    setTickers(newTickers)
    const n   = newTickers.length
    const eq  = Math.floor(100 / n)           // e.g. 33 for 3 tickers
    const rem = 100 - eq * (n - 1)            // last ticker gets the remainder: 34
    setWeights(newTickers.map((_, i) =>
      (i === n - 1 ? rem : eq) / 100          // always sums to exactly 1.0
    ))
  }, [])

  const setWeightsAll = useCallback((newWeights) => {
    setWeights(newWeights)
  }, [])

  return {
    tickers, weights, period, portfolioValue, benchmark, rollingWindow,
    data, loading, error, hasRun, heavyLoading, heavyError,
    rateLimited, heavyRateLimited, retryRun, retryHeavy,
    setTickers: updateTickers,
    updateWeight,
    setWeightsAll,
    setPeriod,
    setPortfolioValue,
    setBenchmark,
    setRollingWindow,
    runAnalysis,
  }
}