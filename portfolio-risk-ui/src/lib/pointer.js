// Single source for "does this device actually support hover" — a real
// touchscreen synthesizes mouseenter/mouseleave around a tap alongside the
// click event, so binding hover handlers unconditionally causes the synthetic
// pair to fight a click-toggle right after it opens something. `(hover:
// hover)` asks the browser directly rather than sniffing width or user
// agent. Originally lived only in MetricTooltip.jsx (RESPONSIVE_AUDIT §7
// calls it out as the one real touch/hover-capability check in the
// codebase, worth reusing) — pulled out here once ReturnHistogram needed
// the identical branch for its canvas hit-testing, rather than duplicating
// the expression a second time. Computed once at module load; nothing that
// currently reads it needs it to react live mid-session.
export const supportsHover = typeof window !== 'undefined' && window.matchMedia('(hover: hover)').matches

// Same reasoning as supportsHover above, for a different question: does this
// device have a mouse/trackpad (which can fire a vertical wheel gesture over
// a horizontally-scrolling element with no way to scroll it) as opposed to
// only touch (which already scrolls the tab bar directly via a horizontal
// drag, per `.tab-bar-scroll`'s own touch-scroll support). Gates App.jsx's
// wheel-to-horizontal-scroll handler on the header tab bar.
export const supportsFinePointer = typeof window !== 'undefined' && window.matchMedia('(pointer: fine)').matches
