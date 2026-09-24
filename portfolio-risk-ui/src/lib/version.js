// Single source for the version string shown to users — previously
// hardcoded independently in three places (content/landing.ts's brand
// config, its hero eyebrow, and Sidebar.jsx's own in-app label), with
// nothing keeping them in sync beyond a human remembering to edit all
// three. Not wired to package.json's own "version" field (currently
// "0.0.1"): that value has never matched this one, and making it the
// source would mean either the displayed version changes to "v0.0.1" or
// package.json gets bumped to agree — either way a real, separate decision
// this file shouldn't make silently as a side effect of a refactor.
export const APP_VERSION = 'v1.3'
