import { typeStyle } from './tokens'

// The two-column frame six blocks share: a numbered mono label on the left,
// content on the right. See README > Screens/Views. Grid columns use clamp()
// / minmax(0, ...) per README > Responsive rules — never a bare fixed track.
export default function SectionFrame({ index, label, children, style }) {
  return (
    <div
      style={{
        maxWidth: 'var(--layout-maxWidth)',
        margin: '0 auto',
        display: 'grid',
        gridTemplateColumns: 'var(--layout-labelColumn) minmax(0, 1fr)',
        gap: 'var(--layout-sectionGap)',
        alignItems: 'start',
        ...style,
      }}
    >
      <div
        style={{
          ...typeStyle('monoSection'),
          color: 'var(--color-text-ghost)',
          paddingTop: 'var(--space-4)',
          borderTop: '1px solid var(--color-line-default)',
        }}
      >
        {index} - {label}
      </div>
      <div style={{ minWidth: 0 }}>{children}</div>
    </div>
  )
}
