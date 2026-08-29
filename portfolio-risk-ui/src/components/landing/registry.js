import Hero from './Hero'
import ProductPlate from './ProductPlate'
import StatsBand from './StatsBand'
import Prose from './Prose'
import Cards from './Cards'
import Steps from './Steps'
import Features from './Features'
import Pillars from './Pillars'
import Pricing from './Pricing'
import Faq from './Faq'
import Cta from './Cta'

// This is .jsx, not .ts, so there's no `satisfies Record<Block['type'], ...>`
// to fail the build on an unhandled variant. Landing.jsx checks every block
// against this map at render time instead, and throws in dev if one is
// missing — see README > Definition of done > "A missing block renderer
// fails loudly, not silently."
export const BLOCK_RENDERERS = {
  hero: Hero,
  productPlate: ProductPlate,
  statsBand: StatsBand,
  prose: Prose,
  cards: Cards,
  steps: Steps,
  features: Features,
  pillars: Pillars,
  pricing: Pricing,
  faq: Faq,
  cta: Cta,
}
