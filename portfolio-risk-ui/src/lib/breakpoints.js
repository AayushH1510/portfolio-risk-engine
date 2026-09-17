// Single JS-side source for the "compact viewport" condition — a phone in
// either orientation, not just a narrow one. min(width, height) <= 767 is
// what actually identifies a phone (A54: 384x854 portrait / 852x393
// landscape, iPhone 14/15/16: 393x852 / 852x393) — equivalent to width<=767
// OR height<=767, since if either axis is <=767 the smaller one is. This
// excludes a landscape tablet (1024x768: neither axis qualifies) and real
// laptop viewports (1366x768, 1280x800: same).
//
// CSS can't reference this value — media query conditions can't read custom
// properties (see --breakpoint-phone's own comment in index.css), so the
// index.css "Compact viewport" block hand-writes the same 767 literal.
// COMPACT_VIEWPORT_MAX is what that block's banner comment points back to;
// keep the two in sync if this ever changes.
export const COMPACT_VIEWPORT_MAX = 767
export const COMPACT_VIEWPORT_QUERY = `(max-width: ${COMPACT_VIEWPORT_MAX}px), (max-height: ${COMPACT_VIEWPORT_MAX}px)`
