import { StrictMode, lazy, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
// react, not /next — this is Vite, not Next.js. The Vercel dashboard's
// setup instructions default to the Next.js import; that one silently
// does nothing here (no Next.js router to hook into).
import { Analytics } from '@vercel/analytics/react'
// Generated from these same @fontsource packages by
// scripts/fix-font-display.mjs (prebuild/predev) — identical rules, except
// font-display: swap rewritten to font-display: optional, which is the fix
// for the font-swap-driven layout shift Lighthouse's CLS audit flagged on
// first paint of "/". See that script's own comment for the full reasoning.
import './landing-fonts.css'
import './index.css'
import Landing from './pages/Landing.jsx'
import LandingPro from './pages/LandingPro.jsx'
import Methodology from './pages/Methodology.jsx'
import Privacy from './pages/Privacy.jsx'
import Terms from './pages/Terms.jsx'
import StylePreview from './pages/StylePreview.jsx'

// The app shell (tab components, Recharts, the Supabase client, everything
// under App.jsx) is the vast majority of the production bundle — a plain
// static `import App from './App.jsx'` put all of that in the same chunk
// every landing-page visitor downloaded before first paint, whether or not
// they ever click through to /app. React.lazy + Suspense below moves it to
// its own chunk, fetched only once a visitor actually navigates to /app.
const App = lazy(() => import('./App.jsx'))

// Same spinner markup/tokens as every other loading state already in the
// app (Valuation.jsx, StockDrawer.jsx, SectorChart.jsx, etc.) — deliberately
// not a new visual, just that same affordance filling the viewport instead
// of a card, since there's no app shell mounted yet for it to sit inside.
function AppLoading() {
  return (
    <div style={{
      height: '100dvh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      gap: 12, background: 'var(--surface-canvas)', color: 'var(--text-muted)',
      fontFamily: 'var(--font-primary)', fontSize: 13,
    }}>
      <svg className="spin" width="20" height="20" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="var(--signal-positive)" strokeWidth="3" strokeDasharray="40 20" />
      </svg>
      Loading…
    </div>
  )
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Suspense fallback={<AppLoading />}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/pro" element={<LandingPro />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/app" element={<App />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="/terms" element={<Terms />} />
          {/* Internal design-system reference page — dev-only, not for
              production. import.meta.env.DEV is a build-time constant Vite
              replaces with a literal false in production builds, so this
              whole conditional (route element included) is dead code Rollup
              eliminates from the bundle, not just an unreachable path left
              live in the shipped JS. */}
          {import.meta.env.DEV && <Route path="/style-preview" element={<StylePreview />} />}
        </Routes>
      </Suspense>
    </BrowserRouter>
    {/* Mounted once here, not per-route — one script for every route this
        app has, landing pages and /app alike. Tracks page views on actual
        URL changes only (BrowserRouter navigations); switching tabs inside
        /app is React state, not a route change, so it won't register as a
        page view — that's expected, not a gap, per the request. */}
    <Analytics />
  </StrictMode>,
)