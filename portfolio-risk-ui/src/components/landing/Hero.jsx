import { useRef } from 'react'
import tokens from '../../../design/tokens.json'
import { useSmokeField } from '../../hooks/useSmokeField'
import { typeStyle, toneColor } from './tokens'
import ActionButton from './ActionButton'

function MetricCell({ metric }) {
  return (
    <div style={{ background: 'var(--color-bg-panel)', boxShadow: 'var(--shadow-hairline)', padding: '20px 22px' }}>
      <div style={{ ...typeStyle('monoTile'), color: 'var(--color-text-faint)', marginBottom: 'var(--space-3)' }}>
        {metric.label}
      </div>
      <div style={{ ...typeStyle('monoData'), color: toneColor(metric.tone) }}>
        {metric.value}
        {metric.suffix && (
          <span style={{ color: 'var(--color-text-ghost)', fontSize: '16px' }}>{metric.suffix}</span>
        )}
      </div>
    </div>
  )
}

export default function Hero({ block }) {
  const canvasRef = useRef(null)
  const prefersReduced =
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useSmokeField(canvasRef, { ...tokens.smoke, reducedMotion: prefersReduced })

  return (
    <section
      id={block.id}
      style={{
        position: 'relative',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        padding: '150px 40px 0',
      }}
    >
      <div style={{ position: 'absolute', inset: '-10% -5% 0', zIndex: 0, overflow: 'hidden' }}>
        <canvas
          ref={canvasRef}
          style={{ width: '110%', height: '100%', display: 'block', position: 'absolute', top: 0, left: '-5%', filter: 'saturate(1.15)' }}
        />
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background:
              'radial-gradient(125% 95% at 50% 6%, rgba(12,9,8,0) 0%, rgba(12,9,8,0.14) 54%, rgba(12,9,8,0.82) 86%, var(--color-bg-base) 100%)',
          }}
        />
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background:
              'linear-gradient(to bottom, rgba(12,9,8,0.32) 0%, rgba(12,9,8,0) 24%, rgba(12,9,8,0) 64%, var(--color-bg-base) 100%)',
          }}
        />
      </div>

      <div style={{ position: 'relative', zIndex: 1, maxWidth: 'var(--layout-maxWidth)', margin: '0 auto', width: '100%' }}>
        <div data-parallax="0.18" style={{ maxWidth: '880px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-3)',
              ...typeStyle('monoEyebrow'),
              color: 'var(--color-text-faint)',
              marginBottom: 'var(--space-8)',
            }}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: 'var(--color-accent-mint)',
                animation: 'varense-eyebrow-blink 2.8s ease-in-out infinite',
              }}
            />
            {block.eyebrow.map((seg, i) => (
              <span key={i} style={{ display: 'contents' }}>
                {i > 0 && <span style={{ color: 'var(--color-text-quiet)' }}>/</span>}
                <span>{seg}</span>
              </span>
            ))}
          </div>

          <h1 style={{ ...typeStyle('displayXL'), margin: '0 0 var(--space-8)', color: 'var(--color-text-primary)', textWrap: 'balance' }}>
            {block.headline.map((seg, i) => (
              <span key={i}>
                {seg.break && <br />}
                {seg.emphasis ? (
                  <span style={{ fontStyle: 'italic', color: 'var(--color-text-secondary)' }}>{seg.text}</span>
                ) : (
                  seg.text
                )}
              </span>
            ))}
          </h1>

          <p style={{ ...typeStyle('lead'), color: 'var(--color-text-muted)', maxWidth: '640px', margin: '0 0 var(--space-10)', textWrap: 'pretty' }}>
            {block.lead}
          </p>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', flexWrap: 'wrap' }}>
            {block.actions.map((link) => (
              <ActionButton key={link.href} link={link} />
            ))}
            {block.reassurance && (
              <span style={{ ...typeStyle('monoCaption'), color: 'var(--color-text-ghost)' }}>{block.reassurance}</span>
            )}
          </div>
        </div>

        <div
          data-parallax="0.08"
          style={{
            marginTop: '92px',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            gap: '1px',
            border: '1px solid var(--color-line-default)',
            borderRadius: 'var(--radius-md)',
            overflow: 'hidden',
          }}
        >
          {block.metrics.map((metric) => (
            <MetricCell key={metric.label} metric={metric} />
          ))}
        </div>
      </div>
    </section>
  )
}
