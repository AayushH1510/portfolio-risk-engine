import { useState, useCallback } from 'react'
import axios from 'axios'
import { format, subMonths, subYears } from 'date-fns'
import { errorMessage } from '../lib/errorMessage'

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
  const [showBenchmark, setShowBenchmark]   = useState(true)
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

  const runAnalysis = useCallback(async (customDates = null) => {
    setLoading(true)
    setError(null)
    setHeavyError(null)

    const startDate = customDates ? customDates.start : fmt(periodMap[period]())
    const endDate   = customDates ? customDates.end   : fmt(new Date())

    const payload = {
      tickers,
      weights,
      start_date:      startDate,
      end_date:        endDate,
      portfolio_value: portfolioValue,
      show_benchmark:  showBenchmark,
      rolling_window:  rollingWindow,
    }

    try {
      const res = await axios.post(`${API}/api/analyse-summary`, payload)
      setData(res.data)
      setHasRun(true)

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
        .catch(err => setHeavyError(errorMessage(err, 'Something went wrong loading the full simulation. Please try again in a moment.')))
        .finally(() => setHeavyLoading(false))

    } catch (err) {
      // errorMessage() surfaces the backend's actual detail whenever the
      // server responded at all (invalid-ticker 404, rate-limit 503, or our
      // own generic 500 all carry their own accurate message already — see
      // api.py). This fallback only fires when there was no response to
      // read a detail from at all — a real network failure, a timeout, or
      // the request never reaching the server — so it must not imply the
      // user did anything wrong; the tickers may be completely fine.
      setError(errorMessage(err, 'Something went wrong loading data. Please try again in a moment.'))
    } finally {
      setLoading(false)
    }
  }, [tickers, weights, period, portfolioValue, showBenchmark, rollingWindow])

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
    tickers, weights, period, portfolioValue, showBenchmark, rollingWindow,
    data, loading, error, hasRun, heavyLoading, heavyError,
    setTickers: updateTickers,
    updateWeight,
    setWeightsAll,
    setPeriod,
    setPortfolioValue,
    setShowBenchmark,
    setRollingWindow,
    runAnalysis,
  }
}