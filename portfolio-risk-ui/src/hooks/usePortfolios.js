import { useState, useEffect, useCallback } from 'react'
import { supabase } from '../lib/supabase'

export function usePortfolios(user) {
  const [portfolios, setPortfolios] = useState([])
  const [loading, setLoading]       = useState(false)

  // Fetch all portfolios for the current user
  const fetchPortfolios = useCallback(async () => {
    if (!user) { setPortfolios([]); return }
    setLoading(true)
    const { data, error } = await supabase
      .from('portfolios')
      .select('*')
      .order('created_at', { ascending: false })
    if (!error) setPortfolios(data || [])
    setLoading(false)
  }, [user?.id])

  useEffect(() => { fetchPortfolios() }, [fetchPortfolios])

  const savePortfolio = async ({ name, tickers, weights, period, portfolioValue }) => {
    if (!user) { console.error('Not signed in'); return null }
    console.log('Saving portfolio for user:', user.id)
    const { data, error } = await supabase
      .from('portfolios')
      .insert({
        user_id:         user.id,
        name:            name.trim() || 'My Portfolio',
        tickers,
        weights,
        period,
        portfolio_value: portfolioValue,
      })
      .select()
      .single()
    if (error) {
      console.error('Supabase save error:', error)
      alert(`Save failed: ${error.message}`)
      return null
    }
    console.log('Saved:', data)
    setPortfolios(prev => [data, ...prev])
    return data.id
  }

  const deletePortfolio = async (id) => {
    if (!user) return
    const { error } = await supabase
      .from('portfolios')
      .delete()
      .eq('id', id)
    if (!error) setPortfolios(prev => prev.filter(p => p.id !== id))
  }

  const renamePortfolio = async (id, name) => {
    if (!user) return
    const { error } = await supabase
      .from('portfolios')
      .update({ name })
      .eq('id', id)
    if (!error) setPortfolios(prev => prev.map(p => p.id === id ? { ...p, name } : p))
  }

  return { portfolios, loading, savePortfolio, deletePortfolio, renamePortfolio, refetch: fetchPortfolios }
}