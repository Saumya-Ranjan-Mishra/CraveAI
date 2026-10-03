import { FormEvent, useState } from 'react'
import { MapPin, Search, Star, Utensils } from 'lucide-react'

type SearchItem = {
  item_id: string
  item_name: string
  restaurant_name: string
  menu_price: string | number | null
  item_rating: string | number | null
  restaurant_address?: string
  city?: string
  rerank_score?: number
}

const API_URL = import.meta.env.VITE_SEARCH_API_URL || '/api/search'

function displayPrice(value: SearchItem['menu_price']) {
  if (value === null || value === undefined || value === '') return 'Price unavailable'
  const text = String(value).trim()
  return text.startsWith('$') ? text : `$${text}`
}

function displayRating(value: SearchItem['item_rating']) {
  const rating = Number.parseFloat(String(value ?? ''))
  return Number.isFinite(rating) && rating > 0 ? rating.toFixed(1) : 'New'
}

export default function App() {
  const [query, setQuery] = useState('craving for pizza')
  const [topK, setTopK] = useState(10)
  const [items, setItems] = useState<SearchItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [searched, setSearched] = useState(false)

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError('')
    setSearched(true)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), top_k: topK }),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `Search failed (${response.status})`)
      }

      const payload: unknown = await response.json()
      if (!Array.isArray(payload)) throw new Error('The search API returned an unexpected response.')
      setItems(payload as SearchItem[])
    } catch (caught) {
      setItems([])
      setError(caught instanceof Error ? caught.message : 'Unable to search right now.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="CraveAI home">
          <span className="brand-mark"><Utensils size={18} strokeWidth={2.2} /></span>
          <span>crave<span className="brand-light">ai</span></span>
        </a>
        <span className="topbar-label">MENU SEARCH</span>
      </header>

      <section className="search-section" aria-labelledby="search-title">
        <div className="section-kicker"><span /> HOUSTON MENU</div>
        <h1 id="search-title">Find your next <em>favorite bite.</em></h1>

        <form className="search-form" onSubmit={handleSearch}>
          <Search className="search-icon" size={20} aria-hidden="true" />
          <input
            aria-label="Search menu items"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="What are you craving?"
            maxLength={500}
          />
          <label className="top-k-control">
            <span>SHOW</span>
            <select aria-label="Number of results" value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              {[5, 10, 20, 50].map((count) => <option key={count} value={count}>{count}</option>)}
            </select>
          </label>
          <button type="submit" disabled={loading || !query.trim()} aria-label="Search">
            {loading ? <span className="spinner" /> : <Search size={18} />}
          </button>
        </form>
      </section>

      <section className="results-section" aria-live="polite">
        <div className="results-heading">
          <div>
            <p className="section-kicker">YOUR RESULTS</p>
            <h2>{loading ? 'Searching menus…' : searched ? `Matches for “${query.trim()}”` : 'Popular around you'}</h2>
          </div>
          {items.length > 0 && <span className="result-count">{items.length} ITEMS</span>}
        </div>

        {error && <div className="state-message error-state">{error}</div>}
        {!loading && !error && searched && items.length === 0 && (
          <div className="state-message">No matches came back. Try a different dish or ingredient.</div>
        )}
        {!searched && (
          <div className="start-state"><span className="start-icon"><Search size={22} /></span><span>Search for a dish, ingredient, or craving.</span></div>
        )}

        {items.length > 0 && (
          <div className="results-grid">
            {items.map((item, index) => (
              <article className="result-tile" key={`${item.item_id}-${index}`}>
                <div className="tile-topline">
                  <span className="tile-number">{String(index + 1).padStart(2, '0')}</span>
                  <span className="rating"><Star size={14} fill="currentColor" strokeWidth={1.5} /> {displayRating(item.item_rating)}</span>
                </div>
                <h3>{item.item_name || 'Menu item'}</h3>
                <p className="restaurant"><Utensils size={14} /> {item.restaurant_name || 'Restaurant'}</p>
                <div className="tile-bottomline">
                  <span className="price">{displayPrice(item.menu_price)}</span>
                  {item.city && <span className="location"><MapPin size={13} /> {item.city}</span>}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
      <footer className="footer"><span>CRAVEAI</span><span>LOCAL MENU DISCOVERY</span></footer>
    </main>
  )
}