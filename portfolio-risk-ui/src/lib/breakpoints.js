// Single JS-side source for the "compact viewport" condition — a phone in
// either orientation, not just a narrow one.
//
// REGRESSION, fixed here: the height clause used to be a bare
// `(max-height: 767px)`, on the theory that a landscape phone (A54:
// 852x393, iPhone 14/15/16: 852x393) is the only realistic case with a
// width well past 767px but a short height. That's true for a full-height
// mobile browser viewport, but false the moment a desktop browser's own
// chrome (tabs, address bar, bookmarks bar) and the OS's own display
// scaling are in the picture: a real 1920x1080 laptop at 125% Windows
// scaling reports a ~1536x864 logical viewport before chrome, and browser
// chrome alone routinely eats another ~130px, landing well inside
// "max-height: 767px" — confirmed live, not theoretical (a 1536x730
// browser window triggered the full phone layout, folded RiskAnalysis
// labels included, on an ordinary laptop). 767px was never a real
// landscape-phone height (those are ~380-430px); it was copied from the
// width clause's own number, which happened to also bound every laptop
// window this project had been testing against full-height, chrome-free.
//
// The fix narrows the height clause on two axes at once, both required
// together (`and`, not another `,`-joined OR): max-height:500px — real
// landscape phones (~380-430px) clear this with headroom, while even a
// cramped laptop browser window rarely drops under it — AND
// pointer:coarse, which a touchscreen reports and a mouse/trackpad-driven
// laptop never does, independent of window height. A short desktop window
// (someone resizing Chrome down for a screenshot, say) has pointer:fine
// and no longer matches no matter how short it gets. The width clause
// (max-width:767px) is untouched — phone *portrait* was never the buggy
// half, only the height-alone OR was over-broad.
//
// CSS can't reference this value — media query conditions can't read custom
// properties (see --breakpoint-phone's own comment in index.css), so the
// index.css "Compact viewport" block hand-writes the same condition.
// COMPACT_VIEWPORT_QUERY is what that block's banner comment points back
// to; keep the two in sync if this ever changes.
export const COMPACT_VIEWPORT_MAX = 767
export const COMPACT_VIEWPORT_LANDSCAPE_MAX_HEIGHT = 500
export const COMPACT_VIEWPORT_QUERY = `(max-width: ${COMPACT_VIEWPORT_MAX}px), (max-height: ${COMPACT_VIEWPORT_LANDSCAPE_MAX_HEIGHT}px) and (pointer: coarse)`
