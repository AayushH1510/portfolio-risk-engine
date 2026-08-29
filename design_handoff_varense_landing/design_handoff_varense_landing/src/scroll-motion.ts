/**
 * scroll-motion.ts — parallax + reveal, driven by data attributes.
 *
 * One rAF-throttled scroll listener for the whole page, regardless of how many
 * sections exist. Adding a section never means adding a listener: mark up the
 * element and it participates.
 *
 *   <div data-parallax="0.18">    depth layer; the number is a speed multiplier
 *   <div data-reveal>             fades and rises in when scrolled to
 *
 * Both are progressive enhancement. If this module never runs, or JS is off,
 * every element stays fully visible at its natural position — reveal opacity
 * is applied by script, never in CSS. Do not move it into a stylesheet.
 */

export interface ScrollMotionOptions {
  /** 0..1 global damper. Wire to a CMS/theme setting; 0 disables parallax. */
  intensity?: number;
  /** px of travel at speed 1.0 and intensity 1.0. tokens.json > motion.parallaxRange */
  range?: number;
  /** Scope. Defaults to document. Pass a ref for embedded/multi-instance use. */
  root?: ParentNode;
  /** Called on every throttled scroll tick with scrollY — for the sticky nav. */
  onScroll?: (scrollY: number) => void;
  reveal?: {
    duration?: string;
    easing?: string;
    offsetY?: string;
    staggerStep?: number;
    staggerMax?: number;
  };
}

export function initScrollMotion(options: ScrollMotionOptions = {}) {
  const {
    intensity = 0.35,
    range = 190,
    root = document,
    onScroll,
    reveal = {},
  } = options;

  const {
    duration = '900ms',
    easing = 'cubic-bezier(.16,1,.3,1)',
    offsetY = '22px',
    staggerStep = 90,
    staggerMax = 360,
  } = reveal;

  const prefersReduced =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const parallaxEls = Array.from(
    root.querySelectorAll<HTMLElement>('[data-parallax]')
  );
  const revealEls = Array.from(
    root.querySelectorAll<HTMLElement>('[data-reveal]')
  );

  let pending = revealEls.slice();
  let observer: IntersectionObserver | null = null;

  const show = (el: HTMLElement) => {
    el.style.opacity = '1';
    el.style.transform = 'translateY(0)';
    pending = pending.filter((n) => n !== el);
  };

  // --- reveal -------------------------------------------------------------
  if (!prefersReduced && 'IntersectionObserver' in window) {
    for (const el of revealEls) {
      el.style.opacity = '0';
      el.style.transform = `translateY(${offsetY})`;
      el.style.transition = `opacity ${duration} ${easing}, transform ${duration} ${easing}`;
    }

    observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const el = entry.target as HTMLElement;
          // stagger against reveal siblings so a row of cards cascades
          const siblings = Array.from(el.parentElement?.children ?? []).filter(
            (n) => n instanceof HTMLElement && n.hasAttribute('data-reveal')
          );
          const idx = Math.max(0, siblings.indexOf(el));
          el.style.transitionDelay = `${Math.min(idx * staggerStep, staggerMax)}ms`;
          show(el);
          observer!.unobserve(el);
        }
      },
      { rootMargin: '0px 0px -12% 0px', threshold: 0.08 }
    );

    revealEls.forEach((el) => observer!.observe(el));
  } else {
    pending = [];
  }

  // --- parallax + nav + reveal safety net ---------------------------------
  let raf = 0;

  const tick = () => {
    const scrollY = window.scrollY;
    const vh = window.innerHeight;

    onScroll?.(scrollY);

    // Safety net: a fast jump (scrollbar drag, Cmd+End, deep hash link) can
    // outrun the observer and strand a block at opacity 0. Anything already
    // in view gets revealed regardless.
    if (pending.length) {
      for (const el of pending.slice()) {
        if (el.getBoundingClientRect().top < vh * 0.94) show(el);
      }
    }

    if (prefersReduced || intensity <= 0) return;

    for (const el of parallaxEls) {
      const rect = el.getBoundingClientRect();
      if (rect.bottom < -300 || rect.top > vh + 300) continue; // offscreen
      const centreOffset = (rect.top + rect.height / 2 - vh / 2) / vh;
      const speed = parseFloat(el.dataset.parallax || '0');
      el.style.transform = `translate3d(0, ${(-centreOffset * speed * range * intensity).toFixed(2)}px, 0)`;
      el.style.willChange = 'transform';
    }
  };

  const onScrollEvent = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      tick();
    });
  };

  window.addEventListener('scroll', onScrollEvent, { passive: true });
  window.addEventListener('resize', onScrollEvent, { passive: true });
  tick();

  return {
    destroy() {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onScrollEvent);
      window.removeEventListener('resize', onScrollEvent);
      observer?.disconnect();
    },
  };
}
