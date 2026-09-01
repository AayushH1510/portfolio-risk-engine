/**
 * Logo.jsx — the only sanctioned way to render the Varense mark.
 *
 * Geometry is fixed and lives here, once. Do not inline the paths elsewhere,
 * and do not re-derive the arc: the envelope uses large-arc-flag=1, which puts
 * the ellipse centre at cy≈40.95 and makes the dome 37 tall, NOT 30. Rebuilding
 * a plain half-ellipse from rx27/ry30 produces a visibly shallower dome.
 *
 * Detail simplifies by stripping the interior path, never the envelope. The
 * dome and its baseline are the identity at every size.
 *
 *   <Logo />                        horizontal lockup, 104px wide
 *   <Logo variant="stacked" />
 *   <Logo variant="mark" size={58} />
 *   <Logo variant="mark" detail="simple" />   // ≤32px
 *   <Logo variant="mark" detail="solid" />    // ≤16px
 *   <Logo ink="var(--color-text-primary)" />
 */

const ENVELOPE = 'M5.19,44.5 A27,30 0 1 1 58.81,44.5';
const PATH_FULL = 'M5,48 L16,48 L23,41.5 L26.5,44.5 L36,34 L48,48 L59,48';
const PATH_SIMPLE = 'M5,48 L18,48 L36,34 L48,48 L59,48';

// Tight boxes around the true bounds (content spans y 10.95→48), ~2u margin.
// Anything shallower clips the apex — this was a real bug.
const BOX = { standard: '3 9 58 41', solid: '2 5 60 45' };
const ASPECT = { standard: 58 / 41, solid: 60 / 45 };

function Mark({ detail = 'full', ink = '#F6F1EA', height, title }) {
  const solid = detail === 'solid';
  const box = solid ? BOX.solid : BOX.standard;
  const aspect = solid ? ASPECT.solid : ASPECT.standard;
  const h = height ?? 41;

  return (
    <svg
      viewBox={box}
      width={Math.round(h * aspect)}
      height={h}
      fill="none"
      role={title ? 'img' : 'presentation'}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      style={{ display: 'block', flex: 'none' }}
    >
      {solid ? (
        <>
          <path d="M5.6,43 A27,30 0 1 1 58.4,43" stroke={ink} strokeWidth="6" />
          <path d="M16,48 L36,29 L56,48 Z" fill={ink} />
          <path d="M4,48 L60,48" stroke={ink} strokeWidth="6" />
        </>
      ) : (
        <>
          <path d={ENVELOPE} stroke={ink} strokeWidth={detail === 'simple' ? 4.5 : 3.5} />
          <path
            d={detail === 'simple' ? PATH_SIMPLE : PATH_FULL}
            stroke={ink}
            strokeWidth={detail === 'simple' ? 4.5 : 3.5}
          />
        </>
      )}
    </svg>
  );
}

export default function Logo({
  variant = 'horizontal',
  detail = 'full',
  size,
  ink = 'currentColor',
  ...rest
}) {
  if (variant === 'mark') {
    return <Mark detail={detail} ink={ink} height={size ?? 41} title="Varense" {...rest} />;
  }

  const stacked = variant === 'stacked';
  // Reference construction: horizontal 44×31 mark, 14px gap, wordmark 32px.
  // Stacked 58×41 mark, 14px gap, wordmark 24px. Scale as a whole only.
  const scale = (size ?? (stacked ? 58 : 44)) / (stacked ? 58 : 44);

  return (
    <span
      style={{
        display: 'inline-flex',
        flexDirection: stacked ? 'column' : 'row',
        alignItems: 'center',
        gap: 14 * scale,
        color: ink,
      }}
      {...rest}
    >
      <Mark detail={detail} ink="currentColor" height={(stacked ? 41 : 31) * scale} title="Varense" />
      <span
        style={{
          fontFamily: 'var(--font-primary)',
          fontWeight: 500,
          fontSize: (stacked ? 24 : 32) * scale,
          letterSpacing: '-0.035em',
          lineHeight: 1,
          whiteSpace: 'nowrap',
        }}
      >
        Varense
      </span>
    </span>
  );
}
