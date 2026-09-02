import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useWindowScroll } from '../hooks/useWindowScroll'
import { typeStyle } from '../components/landing/tokens'
import SectionFrame from '../components/landing/SectionFrame'
import Footer from '../components/landing/Footer'
import Logo from '../components/Logo'
import { landingPage } from '../content/landing'
import { constants, divergencesCallout, dataSourcing, metrics, APP_URL } from '../content/methodology'

// Same **bold** convention as Prose.jsx, extended with `mono` backtick spans
// for formula fragments, constants, and numbers inline in prose — this page's
// one departure from the landing page's own components, since a methodology
// page quotes numbers and code far more densely than marketing copy does.
function renderInline(text) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <span key={i} style={{ color: 'var(--color-text-body)' }}>{part.slice(2, -2)}</span>
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.92em', color: 'var(--color-accent-mint)', background: 'var(--color-bg-inset)', padding: '1px 5px', borderRadius: 'var(--radius-xs)', overflowWrap: 'anywhere' }}>
          {part.slice(1, -1)}
        </code>
      )
    }
    return <span key={i}>{part}</span>
  })
}

function FormulaBlock({ children, tone = 'default' }) {
  return (
    <pre
      style={{
        margin: 0,
        padding: '16px 18px',
        background: tone === 'code' ? 'var(--color-bg-inset)' : 'var(--color-bg-panel)',
        border: `1px solid ${tone === 'code' ? 'var(--color-line-default)' : 'var(--color-accent-mintBorder)'}`,
        borderRadius: 'var(--radius-sm)',
        fontFamily: 'var(--font-mono)',
        fontSize: tone === 'code' ? '13px' : '14.5px',
        lineHeight: 1.7,
        color: tone === 'code' ? 'var(--color-text-secondary)' : 'var(--color-accent-mint)',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        overflowX: 'auto',
        minWidth: 0,
      }}
    >
      {children}
    </pre>
  )
}

function MethodologyNav() {
  const [hovered, setHovered] = useState(false)
  return (
    <nav
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 50,
        background: 'var(--color-alpha-scrim)',
        backdropFilter: 'var(--motion-navTransition-blur)',
        WebkitBackdropFilter: 'var(--motion-navTransition-blur)',
        borderBottom: '1px solid var(--color-line-subtle)',
      }}
    >
      <div
        style={{
          maxWidth: 'var(--layout-maxWidth)',
          margin: '0 auto',
          padding: '0 var(--layout-gutter)',
          minHeight: 'var(--layout-navHeight)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '16px',
          flexWrap: 'wrap',
        }}
      >
        <Link
          to="/"
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', color: 'var(--color-text-body)', textDecoration: 'none' }}
        >
          <Logo variant="mark" size={26} ink="var(--color-text-body)" />
          <span style={{ ...typeStyle('monoAction'), fontSize: '15px', letterSpacing: '0.02em', color: hovered ? 'var(--color-accent-mint)' : 'var(--color-text-body)', transition: 'color 200ms ease' }}>
            Varense
          </span>
          <span style={{ ...typeStyle('monoCaption'), color: 'var(--color-text-ghost)' }}>/ Methodology</span>
        </Link>

        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <Link to="/#method" style={{ ...typeStyle('monoNav'), color: 'var(--color-text-faint)', textDecoration: 'none' }}>
            ← Back to overview
          </Link>
          <a
            href={APP_URL}
            style={{
              ...typeStyle('monoNav'), letterSpacing: '0.06em', padding: '10px 20px',
              border: '1px solid var(--color-accent-mint)', borderRadius: 'var(--radius-xs)',
              color: 'var(--color-accent-mint)', whiteSpace: 'nowrap', textDecoration: 'none',
            }}
          >
            Launch the app
          </a>
        </div>
      </div>
    </nav>
  )
}

function MetricSection({ metric }) {
  return (
    <section id={metric.id} style={{ padding: '70px var(--layout-gutter)', borderTop: '1px solid var(--color-line-subtle)' }}>
      <SectionFrame index={metric.index} label={metric.label}>
        <h2 style={{ ...typeStyle('displayS'), margin: '0 0 var(--space-8)', color: 'var(--color-text-primary)' }}>
          {metric.heading}
        </h2>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', minWidth: 0 }}>
          <div>
            <div style={{ ...typeStyle('monoLabel'), color: 'var(--color-text-faint)', marginBottom: 'var(--space-3)' }}>
              Standard formula
            </div>
            <FormulaBlock>{metric.standard}</FormulaBlock>
            {metric.standardNote && (
              <p style={{ ...typeStyle('bodyM'), color: 'var(--color-text-dim)', margin: 'var(--space-4) 0 0', maxWidth: '760px' }}>
                {renderInline(metric.standardNote)}
              </p>
            )}
          </div>

          <div>
            <div style={{ ...typeStyle('monoLabel'), color: 'var(--color-accent-mint)', marginBottom: 'var(--space-3)' }}>
              What Varense implements
            </div>
            <FormulaBlock tone="code">{metric.implementation}</FormulaBlock>
            <p style={{ ...typeStyle('bodyM'), color: 'var(--color-text-dim)', margin: 'var(--space-4) 0 0', maxWidth: '760px' }}>
              {renderInline(metric.implementationNote)}
            </p>
          </div>

          <div style={{ borderLeft: '2px solid var(--color-line-default)', paddingLeft: 'var(--space-5)' }}>
            <div style={{ ...typeStyle('monoLabel'), color: 'var(--color-text-faint)', marginBottom: 'var(--space-2)' }}>
              Where it appears
            </div>
            <p style={{ ...typeStyle('bodyS'), color: 'var(--color-text-ghost)', margin: 0 }}>
              {renderInline(metric.whereItAppears)}
            </p>
          </div>
        </div>
      </SectionFrame>
    </section>
  )
}

export default function Methodology() {
  const { brand, footer } = landingPage
  useWindowScroll()

  return (
    <div
      style={{
        position: 'relative',
        background: 'var(--color-bg-base)',
        color: 'var(--color-text-body)',
        fontFamily: 'var(--font-body)',
        fontWeight: 300,
      }}
    >
      <MethodologyNav />

      {/* Hero */}
      <header style={{ padding: '90px var(--layout-gutter) 70px', maxWidth: 'var(--layout-maxWidth)', margin: '0 auto' }}>
        <div style={{ ...typeStyle('monoEyebrow'), color: 'var(--color-text-faint)', marginBottom: 'var(--space-6)' }}>
          Technical reference
        </div>
        <h1 style={{ ...typeStyle('displayL'), color: 'var(--color-text-primary)', margin: '0 0 var(--space-6)', maxWidth: '820px', textWrap: 'balance' }}>
          Every formula, every constant, exactly as implemented.
        </h1>
        <p style={{ ...typeStyle('lead'), color: 'var(--color-text-muted)', maxWidth: '700px', margin: 0 }}>
          This page exists for readers who want to see the actual arithmetic before trusting a number on a dashboard.
          For every metric Varense computes, it states the standard textbook formula, then what the code literally
          does, pulled from <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>our code directly</code>,{' '}
          not from memory of convention. Where the implementation diverges from the textbook version, that divergence
          is named. Where the code documents why, the reasoning is included. Where it doesn't, this page says so
          rather than inventing one.
        </p>
      </header>

      {/* Constants at a glance */}
      <section style={{ padding: '0 var(--layout-gutter) 80px', maxWidth: 'var(--layout-maxWidth)', margin: '0 auto' }}>
        <div style={{ ...typeStyle('monoSection'), color: 'var(--color-text-ghost)', marginBottom: 'var(--space-6)', paddingTop: 'var(--space-4)', borderTop: '1px solid var(--color-line-default)' }}>
          Constants, read directly from the code
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1px', border: '1px solid var(--color-line-default)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
          {constants.map((c) => (
            <div key={c.label} style={{ background: 'var(--color-bg-panel)', padding: '20px 22px' }}>
              <div style={{ ...typeStyle('monoTile'), color: 'var(--color-text-faint)', marginBottom: 'var(--space-3)' }}>{c.label}</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '22px', color: 'var(--color-accent-mint)', marginBottom: 'var(--space-3)' }}>{c.value}</div>
              <div style={{ ...typeStyle('bodyXS'), color: 'var(--color-text-ghost)', lineHeight: 1.6 }}>{c.note}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Prominent divergences callout */}
      <section style={{ padding: '80px var(--layout-gutter)', background: 'var(--color-bg-raised)', borderTop: '1px solid var(--color-line-subtle)', borderBottom: '1px solid var(--color-line-subtle)' }}>
        <div style={{ maxWidth: 'var(--layout-maxWidth)', margin: '0 auto' }}>
          <h2 style={{ ...typeStyle('displayM'), color: 'var(--color-text-primary)', margin: '0 0 var(--space-4)', maxWidth: '720px', textWrap: 'balance' }}>
            Four choices worth knowing before you read the rest of this page.
          </h2>
          <p style={{ ...typeStyle('lead'), color: 'var(--color-text-muted)', maxWidth: '680px', margin: '0 0 var(--space-14)' }}>
            These aren't buried in a footnote. They shape how nearly every number below should be read.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1px', border: '1px solid var(--color-line-default)' }}>
            {divergencesCallout.map((d) => (
              <div key={d.title} style={{ background: 'var(--color-bg-panel)', boxShadow: 'var(--shadow-hairline)', padding: '32px 30px' }}>
                <h3 style={{ ...typeStyle('headingXS'), color: 'var(--color-accent-mint)', margin: '0 0 var(--space-4)' }}>{d.title}</h3>
                <p style={{ ...typeStyle('bodyS'), color: 'var(--color-text-dim)', margin: 0, lineHeight: 1.7 }}>{renderInline(d.body)}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Data sourcing */}
      <section style={{ padding: '80px var(--layout-gutter)', borderBottom: '1px solid var(--color-line-subtle)' }}>
        <SectionFrame index="" label="Data">
          <h2 style={{ ...typeStyle('displayM'), margin: '0 0 var(--space-8)', color: 'var(--color-text-primary)', maxWidth: '720px' }}>
            {dataSourcing.heading}
          </h2>
          {dataSourcing.paragraphs.map((p, i) => (
            <p key={i} style={{ ...typeStyle('bodyM'), color: 'var(--color-text-dim)', maxWidth: '760px', margin: i === dataSourcing.paragraphs.length - 1 ? 0 : '0 0 var(--space-5)', lineHeight: 1.75 }}>
              {renderInline(p)}
            </p>
          ))}
        </SectionFrame>
      </section>

      {/* Per-metric sections */}
      {metrics.map((m) => (
        <MetricSection key={m.id} metric={m} />
      ))}

      {/* Closing note */}
      <section style={{ padding: '90px var(--layout-gutter)', textAlign: 'center', borderTop: '1px solid var(--color-line-subtle)' }}>
        <h2 style={{ ...typeStyle('displayS'), color: 'var(--color-text-primary)', margin: '0 0 var(--space-5)' }}>
          Run it yourself.
        </h2>
        <p style={{ ...typeStyle('lead'), color: 'var(--color-text-muted)', maxWidth: '560px', margin: '0 auto var(--space-8)' }}>
          Every figure above is one portfolio away from being checked against your own holdings.
        </p>
        <a
          href={APP_URL}
          style={{
            display: 'inline-block', ...typeStyle('monoAction'), padding: '16px 30px', borderRadius: 'var(--radius-sm)',
            background: 'var(--color-accent-mint)', color: 'var(--color-bg-base)', textDecoration: 'none',
            boxShadow: 'var(--shadow-action)',
          }}
        >
          Launch the app
        </a>
      </section>

      <Footer brand={brand} footer={footer} />
    </div>
  )
}
