import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App.jsx'
import LandingRetail from './pages/LandingRetail.jsx'
import LandingPro from './pages/LandingPro.jsx'
import Privacy from './pages/Privacy.jsx'
import Terms from './pages/Terms.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingRetail />} />
        <Route path="/pro" element={<LandingPro />} />
        <Route path="/app" element={<App />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/terms" element={<Terms />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)