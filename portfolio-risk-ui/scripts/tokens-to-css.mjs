// Generates src/landing-theme.css from design/tokens.json. Run via `prebuild`/
// `predev` (see package.json) so a token edit can never drift from the
// stylesheet — see design_handoff_varense_landing/README.md > Rule 1.
import tokens from '../design/tokens.json' with { type: 'json' };

const lines = [];

const walk = (node, path = []) => {
  for (const [k, v] of Object.entries(node)) {
    if (k.startsWith('$')) continue;
    if (v && typeof v === 'object' && 'value' in v) {
      lines.push(`  --${[...path, k].join('-')}: ${v.value};`);
    } else if (v && typeof v === 'object' && !Array.isArray(v)) {
      walk(v, [...path, k]);
    } else {
      lines.push(`  --${[...path, k].join('-')}: ${v};`);
    }
  }
};

walk(tokens.color, ['color']);
walk(tokens.space, ['space']);
walk(tokens.radius, ['radius']);
walk(tokens.shadow, ['shadow']);
walk(tokens.layout, ['layout']);

// Fonts: emit a full stack per role, family + fallback already paired in the
// source so a component never assembles its own font-family string.
for (const [role, f] of Object.entries(tokens.font)) {
  lines.push(`  --font-${role}: '${f.family}', ${f.fallback};`);
}

// Type scale: one role = one set of properties (size/weight/leading/tracking/
// transform), font resolved against the family stack above rather than
// re-stated per role.
for (const [role, t] of Object.entries(tokens.type)) {
  if (role.startsWith('$')) continue;
  lines.push(`  --type-${role}-font: var(--font-${t.font});`);
  lines.push(`  --type-${role}-size: ${t.size};`);
  lines.push(`  --type-${role}-weight: ${t.weight};`);
  if (t.leading !== undefined) lines.push(`  --type-${role}-leading: ${t.leading};`);
  if (t.tracking !== undefined) lines.push(`  --type-${role}-tracking: ${t.tracking};`);
  if (t.transform !== undefined) lines.push(`  --type-${role}-transform: ${t.transform};`);
}

// Motion: only the scalar fields CSS can consume directly (durations,
// easings, offsets). parallaxRange is a JS-only input (scroll-motion.ts).
for (const [name, m] of Object.entries(tokens.motion)) {
  if (name.startsWith('$') || typeof m !== 'object') continue;
  for (const [prop, val] of Object.entries(m)) {
    lines.push(`  --motion-${name}-${prop}: ${val};`);
  }
}

console.log(`/* GENERATED FILE — do not edit by hand.\n   Source: design/tokens.json via scripts/tokens-to-css.mjs (runs in prebuild/predev). */\n:root {\n${lines.join('\n')}\n}\n`);
