import { typeStyle } from './tokens'
import './section-frame.css'

// The two-column frame six blocks share: a numbered mono label on the left,
// content on the right. See README > Screens/Views. Grid columns use clamp()
// / minmax(0, ...) per README > Responsive rules — never a bare fixed track.
// The grid itself (display/gridTemplateColumns/gap, including the <768px
// stack-to-one-column override) lives in section-frame.css, not inline —
// see that file's own comment for why.
export default function SectionFrame({ index, label, children, style }) {
  return (
    <div
      className="section-frame"
      style={{
        maxWidth: 'var(--layout-maxWidth)',
        margin: '0 auto',
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
