// Resolves var(--token) references to their actual computed value.
// Needed specifically for raw Canvas 2D drawing (ctx.fillStyle,
// ctx.strokeStyle, gradient.addColorStop) — the canvas API doesn't
// participate in the CSS cascade, so a string like 'var(--signal-positive)'
// is invalid there even though it works fine in a normal inline style.
// Handles composite strings too, e.g. 'rgba(var(--white-rgb),0.04)'.
export function cssVar(value) {
  if (typeof value !== 'string' || !value.includes('var(--')) return value
  return value.replace(/var\((--[a-zA-Z0-9-]+)\)/g, (_, name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  )
}
