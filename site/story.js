/* ATCO site: in-page navigation, the highlighted menu item and print handling.
   radio.js keeps the audio demo to itself. */
(function () {
  'use strict';

  /* Mirrored in a comment at the top of index.html. Static HTML is written for the
     null state; when a number arrives, the HTML for that variant is edited by hand.
     Nothing here switches copy at runtime. */
  var STATE = { baseline: null, handLabels: null, noiseSweep: false };
  window.ATCO_STATE = STATE;

  /* ---------- keep the anchor offset equal to the real header height ----------
     The header wraps to two rows on narrow screens and grows with enlarged text,
     so a fixed breakpoint value is always slightly wrong. Measure it instead. */
  var header = document.querySelector('.topbar');
  function syncHeaderOffset() {
    if (!header) return;
    var h = Math.round(header.getBoundingClientRect().height);
    if (h > 0) document.documentElement.style.setProperty('--hdr', (h + 8) + 'px');
  }
  syncHeaderOffset();
  window.addEventListener('resize', syncHeaderOffset);
  if (window.ResizeObserver && header) new ResizeObserver(syncHeaderOffset).observe(header);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(syncHeaderOffset);

  /* ---------- in-page navigation, done without relying on the browser ----------
     Fragment navigation is silently dropped in some contexts the page legitimately
     ends up in: a `data:` document (which is how a downloaded copy of this file is
     previewed), and some sandboxed iframes. There the address never gains a hash
     and nothing scrolls, so the top menu appears dead. Doing the scroll ourselves
     works in every context and lets us subtract the sticky header exactly. */

  function headerOffset() {
    var v = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--hdr'), 10);
    return isNaN(v) ? 76 : v;
  }

  function openAncestors(el) {
    var d = el.closest ? el.closest('details') : null;
    while (d) {
      if (!d.open) d.open = true;
      d = d.parentElement && d.parentElement.closest ? d.parentElement.closest('details') : null;
    }
  }

  function goTo(el, hash) {
    openAncestors(el);                       /* opening first, so the offset is measured after reflow */
    var top = Math.max(0, el.getBoundingClientRect().top + window.pageYOffset - headerOffset());
    /* 'instant', never 'smooth': an animated scroll needs the animation-frame loop,
       and that loop is throttled to a stop in a backgrounded tab or a snapshot
       preview. There the jump stalls part-way or lands one click behind. An instant
       jump has no such dependency, and it also satisfies reduced-motion for free. */
    try {
      window.scrollTo({ top: top, left: 0, behavior: 'instant' });
    } catch (e) {
      window.scrollTo(0, top);              /* older browsers: no options object */
    }
    if (Math.abs(window.pageYOffset - top) > 2) window.scrollTo(0, top);   /* last resort */
    /* keep the address in step where the context allows it; a data: document throws */
    if (hash) { try { history.replaceState(null, '', hash); } catch (e) {} }
    /* move focus so keyboard and screen-reader users follow the jump */
    if (!el.hasAttribute('tabindex')) {
      el.setAttribute('tabindex', '-1');
      el.addEventListener('blur', function once() {
        el.removeAttribute('tabindex');
        el.removeEventListener('blur', once);
      });
    }
    try { el.focus({ preventScroll: true }); } catch (e) {}
  }

  function targetOf(href) {
    if (!href || href.charAt(0) !== '#' || href.length < 2) return null;
    try { return document.getElementById(decodeURIComponent(href.slice(1))); } catch (e) { return null; }
  }

  document.addEventListener('click', function (e) {
    if (e.defaultPrevented || e.button !== 0) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;   /* let "open in new tab" work */
    var a = e.target && e.target.closest ? e.target.closest('a[href]') : null;
    if (!a) return;
    var el = targetOf(a.getAttribute('href'));
    if (!el) return;
    e.preventDefault();
    goTo(el, a.getAttribute('href'));
  });

  /* a deep link present on load, and Back/Forward between fragments */
  window.addEventListener('hashchange', function () {
    var el = targetOf(location.hash);
    if (el) goTo(el, null);
  });
  if (location.hash) {
    var initial = targetOf(location.hash);
    if (initial) setTimeout(function () { goTo(initial, null); }, 0);
  }

  /* ---------- highlight the part of the page the reader is in ----------
     Each nav link names the sections it covers (data-covers). Every <section> is
     watched, so reading a section no link covers (the opening, "why I started")
     clears the highlight instead of leaving a stale one behind. */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.topbar nav a[href^="#"]'));
  var owner = {};                                   /* section id -> nav link */
  navLinks.forEach(function (a) {
    var covers = (a.getAttribute('data-covers') || a.getAttribute('href').slice(1)).split(/\s+/);
    covers.forEach(function (id) { if (id) owner[id] = a; });
  });
  var sections = Array.prototype.slice.call(document.querySelectorAll('main section[id]'));

  /* The current section is the last one whose top has passed the bottom of the
     sticky header. Computed from positions on each scroll, which is exact at
     section boundaries (an observer band lit the section above for a few pixels)
     and needs no animation frame, so it also works in a backgrounded tab. */
  var current = null;
  function setCurrent(link) {
    if (current === link) return;
    if (current) current.removeAttribute('aria-current');
    current = link;
    if (current) current.setAttribute('aria-current', 'true');
  }
  function updateCurrent() {
    var line = headerOffset() + 12, active = null;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].getBoundingClientRect().top <= line) active = sections[i]; else break;
    }
    var doc = document.documentElement;
    if (window.innerHeight + window.pageYOffset >= doc.scrollHeight - 2) active = sections[sections.length - 1];
    setCurrent(active ? (owner[active.id] || null) : null);
  }
  if (sections.length) {
    window.addEventListener('scroll', updateCurrent, { passive: true });
    window.addEventListener('resize', updateCurrent);
    updateCurrent();
  }

  /* ---------- printing: nothing should stay folded away on paper ---------- */
  var reopened = [];
  window.addEventListener('beforeprint', function () {
    Array.prototype.forEach.call(document.querySelectorAll('details:not([open])'), function (d) {
      d.open = true; reopened.push(d);
    });
  });
  window.addEventListener('afterprint', function () {
    reopened.forEach(function (d) { d.open = false; }); reopened = [];
  });
})();
