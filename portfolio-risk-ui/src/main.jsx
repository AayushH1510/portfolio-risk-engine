import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import '@fontsource/newsreader/200.css'
import '@fontsource/newsreader/200-italic.css'
import '@fontsource/newsreader/300.css'
import '@fontsource/lora/400.css'
import '@fontsource/ibm-plex-sans/300.css'
import '@fontsource/ibm-plex-sans/400.css'
import '@fontsource/ibm-plex-sans/500.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import './index.css'
import App from './App.jsx'
import Landing from './pages/Landing.jsx'
import LandingPro from './pages/LandingPro.jsx'
import Methodology from './pages/Methodology.jsx'
import Privacy from './pages/Privacy.jsx'
import Terms from './pages/Terms.jsx'
import StylePreview from './pages/StylePreview.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
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
    </BrowserRouter>
  </StrictMode>,
)