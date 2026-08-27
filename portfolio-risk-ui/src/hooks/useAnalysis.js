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

  const runAnalysis = useCallback(async (customDates = null) => {
    setLoading(true)
    setError(null)

    const startDate = customDates ? customDates.start : fmt(periodMap[period]())
    const endDate   = customDates ? customDates.end   : fmt(new Date())

    try {
      const res = await axios.post(`${API}/api/analyse`, {
        tickers,
        weights,
        start_date:      startDate,
        end_date:        endDate,
        portfolio_value: portfolioValue,
        show_benchmark:  showBenchmark,
        rolling_window:  rollingWindow,
      })
      setData(res.data)
      setHasRun(true)
    } catch (err) {
      setError(errorMessage(err, 'Something went wrong. Check your tickers and try again.'))
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
    data, loading, error, hasRun,
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