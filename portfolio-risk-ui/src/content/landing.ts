/**
 * content.ts — the landing page as data.
 *
 * THE RULE: no copy, no metric, no price, and no section ordering lives inside
 * a component. Components render a typed block; this file (or a CMS payload
 * shaped like it) decides what blocks exist and in what order.
 *
 * Adding a section later = add a variant to `Block`, add one entry to the
 * renderer registry, append to `landingPage.blocks`. No existing file changes.
 *
 * If you move this to a CMS (Contentlayer, Sanity, Payload, MDX frontmatter),
 * keep these types as the contract and validate the payload against them —
 * e.g. mirror them as a zod schema and parse at the fetch boundary.
 */

/* ------------------------------------------------------------------ atoms */

export type Tone = 'positive' | 'negative' | 'warning' | 'neutral';

export interface Link {
  label: string;
  href: string;
  /** 'primary' = mint fill, 'secondary' = outlined, 'quiet' = text only */
  variant?: 'primary' | 'secondary' | 'quiet';
  external?: boolean;
}

export interface Metric {
  label: string;
  value: string;
  tone?: Tone;
  /** Small trailing unit rendered dimmer, e.g. the "/100" in "86/100". */
  suffix?: string;
  note?: string;
}

/* ----------------------------------------------------------------- blocks */

export interface HeroBlock {
  type: 'hero';
  eyebrow: string[];
  /** Segments are concatenated; `emphasis: true` renders display-italic. */
  headline: Array<{ text: string; emphasis?: boolean; break?: boolean }>;
  lead: string;
  actions: Link[];
  reassurance?: string;
  metrics: Metric[];
}

export interface ProductPlateBlock {
  type: 'productPlate';
  /** 'mock' renders the static replica; 'live' embeds the real dashboard in a
   *  non-interactive shell. Start with 'mock', switch when the app component
   *  can render from a fixture without auth. See README > Roadmap. */
  source: 'mock' | 'live';
  fixture?: string;
  tabs: string[];
  activeTab: string;
}

export interface StatsBandBlock {
  type: 'statsBand';
  stats: Array<{ figure: string; unit?: string; caption: string }>;
}

export interface ProseBlock {
  type: 'prose';
  index: string;
  label: string;
  heading: string;
  paragraphs: string[];
}

export interface CardsBlock {
  type: 'cards';
  index: string;
  label: string;
  heading: string;
  cards: Array<{ ordinal: string; tone: Tone; title: string; body: string }>;
}

export interface StepsBlock {
  type: 'steps';
  index: string;
  label: string;
  heading: string;
  steps: Array<{
    ordinal: string;
    title: string;
    body: string;
    /** Mono side-panel. Lines are rendered verbatim; last line is accented. */
    annotation: string[];
  }>;
}

export interface FeaturesBlock {
  type: 'features';
  index: string;
  label: string;
  features: Array<{
    kicker: string;
    kickerTone: Tone;
    heading: string;
    body: string;
    tags?: string[];
    /** Which visual to pair with the copy. Registry key, not a file path. */
    visual: 'distribution' | 'simulationPaths' | 'stressBars' | 'frontier';
    /** Side the visual sits on at desktop widths. */
    visualSide: 'left' | 'right';
  }>;
}

export interface PillarsBlock {
  type: 'pillars';
  index: string;
  label: string;
  heading: string;
  pillars: Array<{ title: string; body: string }>;
  /** Optional continuation link below the pillar grid — e.g. the full
   *  technical methodology page. A natural "keep reading" for this section,
   *  not a nav item. */
  moreLink?: Link;
}

export interface PricingBlock {
  type: 'pricing';
  index: string;
  label: string;
  heading: string;
  /** Single centered message in place of the tier grid — e.g. "free during
   *  beta." Set this OR `tiers`, not both; `notice` takes priority so real
   *  pricing can be restored later by just adding `tiers` back. */
  notice?: {
    lead: string;
    action: Link;
    note?: string;
  };
  tiers?: Array<{
    name: string;
    price: string;
    period?: string;
    summary: string;
    features: string[];
    action: Link;
    featured?: boolean;
  }>;
}

export interface FaqBlock {
  type: 'faq';
  index: string;
  label: string;
  items: Array<{ question: string; answer: string }>;
}

export interface CtaBlock {
  type: 'cta';
  heading: string;
  lead: string;
  actions: Link[];
}

export type Block =
  | HeroBlock
  | ProductPlateBlock
  | StatsBandBlock
  | ProseBlock
  | CardsBlock
  | StepsBlock
  | FeaturesBlock
  | PillarsBlock
  | PricingBlock
  | FaqBlock
  | CtaBlock;

export interface LandingPage {
  brand: { name: string; version: string; appUrl: string };
  nav: Link[];
  navAction: Link;
  /** Anchor ids are derived from this list, so nav links stay in sync. */
  blocks: Array<Block & { id?: string }>;
  footer: {
    disclaimer: string;
    columns: Array<{ heading: string; links: Link[] }>;
    copyright: string;
    tagline: string;
  };
  /** 0–100. Global parallax damper, exposed so it can be tuned without a deploy. */
  motionIntensity: number;
}

/* ------------------------------------------------------------- the payload */

// Vite: only VITE_-prefixed vars reach the client. Add VITE_APP_URL to .env.example.
const APP_URL = import.meta.env?.VITE_APP_URL ?? '/app';

export const landingPage: LandingPage = {
  brand: { name: 'Varense', version: 'v1.2', appUrl: APP_URL },

  nav: [
    { label: 'What', href: '#what' },
    { label: 'Why', href: '#why' },
    { label: 'How', href: '#how' },
    { label: 'Method', href: '#method' },
    { label: 'Pricing', href: '#pricing' },
  ],
  navAction: { label: 'Launch the app →', href: APP_URL, variant: 'secondary' },

  motionIntensity: 35,

  blocks: [
    {
      id: 'top',
      type: 'hero',
      eyebrow: ['Portfolio risk engine', 'v1.2'],
      headline: [
        { text: 'Know how your portfolio behaves' },
        { text: 'before', emphasis: true, break: true },
        { text: ' the market tells you.' },
      ],
      lead: 'Varense runs correlated Monte Carlo, historical crisis replay, and full risk decomposition across your holdings - the arithmetic an institutional desk runs on a book of billions, applied to a portfolio of three tickers.',
      actions: [
        { label: 'Launch the app', href: APP_URL, variant: 'primary' },
        { label: 'Read the methodology', href: '#method', variant: 'secondary' },
      ],
      reassurance: 'No account. No brokerage link.',
      metrics: [
        { label: 'VaR 95%', value: '−$180', tone: 'negative' },
        { label: 'CVaR 95%', value: '−$245', tone: 'negative' },
        { label: 'Sharpe', value: '1.48', tone: 'positive' },
        { label: 'Max drawdown', value: '−17.6%', tone: 'warning' },
        { label: 'Diversification', value: '86', suffix: '/100', tone: 'positive' },
      ],
    },

    {
      type: 'productPlate',
      source: 'mock',
      fixture: 'aapl-msft-googl-equal',
      tabs: ['Dashboard', 'Risk analysis', 'Monte Carlo', 'Efficient frontier', 'Backtest'],
      activeTab: 'Dashboard',
    },

    {
      type: 'statsBand',
      stats: [
        { figure: '1,000', caption: 'simulated futures per run' },
        { figure: '3', caption: 'crisis regimes replayed' },
        { figure: '8', caption: 'risk-adjusted ratios' },
        { figure: '252', unit: 'd', caption: 'rolling analysis window' },
      ],
    },

    {
      id: 'what',
      type: 'prose',
      index: '01',
      label: 'What it is',
      heading: 'A risk engine. Not a stock picker.',
      paragraphs: [
        'Varense takes the portfolio you already hold and answers a single question in a dozen ways: **what can this lose, and how often?**',
        'Enter your tickers and weights. The engine pulls real fundamentals and price history, builds the covariance structure between your holdings, and runs the full risk stack - value at risk, tail expectation, correlated Monte Carlo, crisis replay, frontier optimization - in a single pass. No recommendations. No signals. Just the distribution you are actually exposed to.',
      ],
    },

    {
      id: 'why',
      type: 'cards',
      index: '02',
      label: 'Why it matters',
      heading: 'Every tool shows you the return. Almost none show you the risk you took to get it.',
      cards: [
        {
          ordinal: '01',
          tone: 'negative',
          title: 'Returns are the part you can see',
          body: 'A green line tells you what happened once. It says nothing about the ninety-nine other years that could have happened instead - including the ones that would have taken you out.',
        },
        {
          ordinal: '02',
          tone: 'warning',
          title: 'Correlation is a hidden position',
          body: "Three excellent companies in one sector is one position wearing three names. Diversification you can't measure is diversification you don't have.",
        },
        {
          ordinal: '03',
          tone: 'positive',
          title: 'The tail is where accounts end',
          body: 'Average outcomes are comfortable and useless. What matters is the worst 5% - its depth, its duration, and whether you could sit through it.',
        },
      ],
    },

    {
      id: 'how',
      type: 'steps',
      index: '03',
      label: 'How it works',
      heading: 'Three inputs. One pass. Every number.',
      steps: [
        {
          ordinal: '01',
          title: 'Describe the book',
          body: 'Up to five tickers, a portfolio value, and weights by percentage or dollar amount. Nothing to connect, nothing to upload.',
          annotation: ['AAPL  34%', 'MSFT  33%', 'GOOGL 33%', '100% allocated'],
        },
        {
          ordinal: '02',
          title: 'Run the engine',
          body: 'Varense fetches price history and fundamentals, estimates the covariance matrix, and factors it with a Cholesky decomposition so simulated paths move together the way your holdings actually do.',
          annotation: ['Σ → LLᵀ', '1,000 paths', '252 trading days', 'bear · base · bull'],
        },
        {
          ordinal: '03',
          title: 'Read the verdict',
          body: 'Eight views of the same portfolio - risk, simulation, frontier, valuation, comparison, backtest - each written in plain language underneath the chart. Export to PDF or CSV.',
          annotation: ['VaR · CVaR · β · α', 'Sharpe · Sortino', 'Treynor · Info ratio', 'PDF / CSV export'],
        },
      ],
    },

    {
      id: 'features',
      type: 'features',
      index: '04',
      label: 'The engine',
      features: [
        {
          kicker: 'Value at risk · CVaR',
          kickerTone: 'negative',
          heading: 'The size of a bad day, and the size of a worse one.',
          body: 'VaR at 95% and 99% marks the loss threshold you should expect to breach one day in twenty and one in a hundred. CVaR answers the question VaR refuses to: when you do breach it, how far past it do you go?',
          tags: ['95% & 99% confidence', 'Rolling 30d / 90d windows'],
          visual: 'distribution',
          visualSide: 'right',
        },
        {
          kicker: 'Monte Carlo · Cholesky-correlated',
          kickerTone: 'positive',
          heading: 'A thousand versions of next year.',
          body: 'Uncorrelated simulation is a lie that flatters your portfolio. Varense factors the covariance matrix so every simulated path preserves the real relationships between your holdings, then runs the year a thousand times under bear, base, and bull drift.',
          tags: ['Chance of profit', 'Chance of −10%', 'Percentile bands'],
          visual: 'simulationPaths',
          visualSide: 'left',
        },
        {
          kicker: 'Historical stress testing',
          kickerTone: 'warning',
          heading: 'Crises you have already lived through, replayed against what you hold now.',
          body: "Simulation assumes tomorrow resembles the distribution. History doesn't. Varense replays 2008, the COVID crash, and the 2022 drawdown across your current book to show the loss you would have carried, and how long it took to come back.",
          visual: 'stressBars',
          visualSide: 'right',
        },
        {
          kicker: 'Efficient frontier · Backtest',
          kickerTone: 'positive',
          heading: 'Where you sit, and where you could have sat.',
          body: 'The frontier plots every weighting of your holdings and marks the one with the best return per unit of risk. Then the backtest settles the argument: your allocation against equal-weight and against the S&P 500, over the same window, net of the same volatility.',
          tags: ['Sharpe · Sortino', 'Treynor · Information'],
          visual: 'frontier',
          visualSide: 'left',
        },
      ],
    },

    {
      id: 'method',
      type: 'pillars',
      index: '05',
      label: 'Method & trust',
      heading: 'Nothing hidden, because nothing needs to be.',
      pillars: [
        {
          title: 'No brokerage connection',
          body: "Varense never asks for account credentials or holdings access. You type tickers and weights; that's the entire attack surface.",
        },
        {
          title: 'Published math',
          body: 'Every metric follows a standard textbook definition - historical VaR and CVaR, Cholesky-correlated Monte Carlo, Markowitz-style frontier simulation. The formulas are documented, not proprietary - down to the exact constants.',
        },
        {
          title: 'Real market data',
          body: 'Historical prices come from Twelve Data; real-time quotes and fundamentals come from Finnhub. Both refresh on every run. No synthetic series, no stale caches.',
        },
        {
          title: 'Educational, by design',
          body: 'Varense is an analysis tool, not an advisor. It produces no recommendations, holds no assets, and takes no position on what you should do next.',
        },
      ],
      moreLink: { label: 'Read the full methodology →', href: '/methodology', variant: 'quiet' },
    },

    {
      id: 'pricing',
      type: 'pricing',
      index: '06',
      label: 'Pricing',
      heading: 'Free while we’re in beta.',
      notice: {
        lead: 'Every early user gets full access to the entire engine - Monte Carlo, stress testing, the efficient frontier, backtesting, all of it - at no cost. Paid plans will follow once the beta ends, but nothing you run today will be billed retroactively.',
        action: { label: 'Start using Varense', href: APP_URL, variant: 'primary' },
        note: 'Pricing coming soon. Beta users keep full access at no charge.',
      },
    },

    {
      id: 'faq',
      type: 'faq',
      index: '07',
      label: 'Questions',
      items: [
        {
          question: 'Is this financial advice?',
          answer: 'No. Varense is an educational analysis tool. It computes standard risk statistics on a portfolio you describe and never recommends buying, selling, or holding anything. Past performance does not guarantee future results.',
        },
        {
          question: 'Do I have to connect my brokerage account?',
          answer: 'Never. You enter tickers and weights by hand. Varense holds no credentials, no account links, and no record of what you own unless you choose to save a portfolio.',
        },
        {
          question: 'Why is there a limit on how many tickers I can add?',
          answer: 'Correlated simulation cost grows with the square of the position count, and market data is metered per symbol, so the engine currently caps a portfolio at five tickers (it works best with two or three). That limit applies to everyone during the beta - there is no paid tier to unlock more today.',
        },
        {
          question: 'How accurate is a Monte Carlo projection?',
          answer: 'It is a model, and every model is wrong in a specific way. Geometric Brownian motion understates fat tails and assumes correlations hold. That is precisely why Varense pairs it with historical crisis replay - the two disagree, and the disagreement is informative.',
        },
        {
          question: 'Where does the market data come from?',
          answer: 'Two sources, split by purpose. Finnhub provides real-time quotes and company fundamentals, powering the stock detail view and the valuation screen. Twelve Data provides the historical price series behind portfolio analysis, backtesting, and Monte Carlo simulation. Both are pulled live on each run, over the window you select, up to the maximum available history.',
        },
      ],
    },

    {
      type: 'cta',
      heading: 'Run your portfolio through it.',
      lead: 'Three tickers, thirty seconds, and a distribution you can actually look at.',
      actions: [
        { label: 'Launch the app', href: APP_URL, variant: 'primary' },
        { label: 'See what it computes', href: '#features', variant: 'secondary' },
      ],
    },
  ],

  footer: {
    disclaimer:
      'Varense is an educational tool. It is not financial advice, and past performance does not guarantee future results.',
    columns: [
      {
        heading: 'Product',
        links: [
          { label: 'What it is', href: '#what' },
          { label: 'The engine', href: '#features' },
          { label: 'Pricing', href: '#pricing' },
        ],
      },
      {
        heading: 'Method',
        links: [
          { label: 'Methodology', href: '#method' },
          { label: 'FAQ', href: '#faq' },
          { label: 'Learn', href: APP_URL },
        ],
      },
      {
        heading: 'Legal',
        links: [
          { label: 'Privacy', href: '/privacy' },
          { label: 'Terms', href: '/terms' },
          { label: 'Disclosures', href: '/terms' },
        ],
      },
    ],
    copyright: `© ${new Date().getFullYear()} Varense`,
    tagline: 'Built for people who read the footnotes.',
  },
};
