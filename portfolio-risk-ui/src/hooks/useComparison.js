import { useState, useCallback } from 'react'
import axios from 'axios'
import { format, subMonths, subYears } from 'date-fns'

const periodMap = {
  '1M':  () => subMonths(new Date(), 1),
  '3M':  () => subMonths(new Date(), 3),
  '6M':  () => subMonths(new Date(), 6),
  '1Y':  () => subYears(new Date(), 1),
  '3Y':  () => subYears(new Date(), 3),
  '5Y':  () => subYears(new Date(), 5),
  'Max': () => subYears(new Date(), 10),
}
const fmt = d => format(d, 'yyyy-MM-dd')

export function useComparison() {
  const [tickers, setTickersRaw]   = useState(['SPY', 'QQQ', 'GLD'])
  const [weights, setWeights]      = useState([0.34, 0.33, 0.33])
  const [period, setPeriod]        = useState('1Y')
  const [portfolioValue, setPortfolioValue] = useState(10000)

  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [hasRun, setHasRun]   = useState(false)

  const setTickers = useCallback((newTickers) => {
    setTickersRaw(newTickers)
    const n    = newTickers.length
    const base = Math.floor(100 / n)
    const rem  = 100 - base * n
    setWeights(newTickers.map((_, i) => (base + (i < rem ? 1 : 0)) / 100))
  }, [])

  const setWeightsAll = useCallback((w) => setWeights(w), [])

  const runComparison = useCallback(async (customDates = null) => {
    setLoading(true)
    setError(null)
    const startDate = customDates ? customDates.start : fmt(periodMap[period]())
    const endDate   = customDates ? customDates.end   : fmt(new Date())
    try {
      // Normalise weights to exactly 1.0 before sending
      const total = weights.reduce((a, b) => a + b, 0)
      const normWeights = weights.map(w => w / total)

      const res = await axios.post('/api/analyse', {
        tickers,
        weights: normWeights,
        start_date:      startDate,
        end_date:        endDate,
        portfolio_value: portfolioValue,
        show_benchmark:  false,
      })
      setData(res.data)
      setHasRun(true)
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Check your tickers.')
    } finally {
      setLoading(false)
    }
  }, [tickers, weights, period, portfolioValue])

  return {
    tickers, weights, period, portfolioValue,
    data, loading, error, hasRun,
    setTickers, setWeightsAll, setPeriod, setPortfolioValue,
    runComparison,
  }
}