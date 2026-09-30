import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /overview is a static file (public/overview/index.html), not a React
// Router route. Vite's own dev-server SPA fallback intercepts any
// extension-less navigation request before it resolves to a public/ file —
// confirmed directly: /overview and /overview/ both returned the React
// app's shell, only /overview/index.html (the literal filename) served the
// deck. This plugin rewrites the two directory-style URLs to the literal
// path before Vite's fallback middleware ever sees them, so local `npm run
// dev` behaves the same way the vercel.json rewrite (see that file's own
// comment) makes production behave.
function overviewStaticRoute() {
  return {
    name: 'overview-static-route',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url === '/overview' || req.url === '/overview/') req.url = '/overview/index.html'
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), overviewStaticRoute()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})