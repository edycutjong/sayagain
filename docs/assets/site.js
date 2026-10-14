/* SayAgain landing — progressive enhancement only. Every element already shows its final, real state in the HTML;
   this script adds the one orchestrated moment (the read-back arriving, two-oh-eight blocked red), connectors between the
   words the model pointed at, the exchange picker, the terminal replay and count-ups. No verdict is computed here. */
(function () {
  'use strict';
  var doc = document, root = doc.documentElement;
  var RM = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var STILL = /(^|[?&])still\b/.test(location.search);
  var MOTION = !RM && !STILL;
  if (STILL) root.classList.add('settled');
  if (!MOTION) root.classList.add('no-motion');
  function $(s, r) { return (r || doc).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || doc).querySelectorAll(s)); }
  function later(fn, ms) { return setTimeout(fn, ms); }
  function onView(el, fn, margin) {
    if (!el) return;
    if (!('IntersectionObserver' in window)) { fn(); return; }
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { io.disconnect(); fn(); } });
    }, { rootMargin: margin || '0px 0px -18% 0px' });
    io.observe(el);
  }

  // ------------------------------------------------------------------ connectors: the words the model pointed at
  function drawLinks(card) {
    if (!card || card.hidden || !card.offsetParent) return;
    var strip = $('.xc-strip', card), svg = $('svg.connectors', card);
    if (!strip || !svg) return;
    var ins = $('.xc-turn[data-t="' + card.getAttribute('data-ins') + '"]', card);
    var rb = $('.xc-turn[data-t="' + card.getAttribute('data-rb') + '"]', card);
    svg.innerHTML = '';
    if (!ins || !rb) return;
    var box = strip.getBoundingClientRect();
    var v = card.className;
    // a wrapped line makes span-to-span ticks cross the text: then only the red tick is drawn, aimed at the red block
    function wraps(turn) { var sd = $('.said', turn); if (!sd) return false; var lh = parseFloat(getComputedStyle(sd).lineHeight) || 30; return sd.getBoundingClientRect().height > lh * 1.5; }
    var wrapped = wraps(ins) || wraps(rb);
    $$('.sp', ins).forEach(function (a) {
      var b = $('.sp[data-f="' + a.getAttribute('data-f') + '"][data-c="' + a.getAttribute('data-c') + '"]', rb);
      if (!b) return;
      var ra = a.getClientRects(), rbx = b.getClientRects();
      if (!ra.length || !rbx.length) return;
      var red = /v-MISMATCH/.test(v) && $('mark.red', b);
      if (wrapped && !red) return;
      var A = ra[ra.length - 1], B = red ? red.getBoundingClientRect() : rbx[0];
      var x1 = A.left + A.width / 2 - box.left, y1 = A.bottom - box.top + 3;
      var x2 = B.left + B.width / 2 - box.left, y2 = B.top - box.top - 3;
      if (y2 <= y1) return;
      var ln = doc.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', x1); ln.setAttribute('y1', y1); ln.setAttribute('x2', x2); ln.setAttribute('y2', y2);
      if (red) ln.setAttribute('class', 'red');
      else if (a.classList.contains('amb')) ln.setAttribute('class', 'amb');
      svg.appendChild(ln);
    });
  }
  function drawAll() { $$('.xcard').forEach(drawLinks); }
  window.addEventListener('resize', function () { clearTimeout(drawAll.t); drawAll.t = later(drawAll, 120); });
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(drawAll);
  window.addEventListener('load', drawAll);

  // ------------------------------------------------------------------ word tokens for the arrival animation
  function tokenize(said) {
    if (said.getAttribute('data-tok')) return;
    var walker = doc.createTreeWalker(said, NodeFilter.SHOW_TEXT, null), nodes = [], n;
    while ((n = walker.nextNode())) nodes.push(n);
    nodes.forEach(function (t) {
      var parts = t.nodeValue.split(/(\s+)/), frag = doc.createDocumentFragment();
      parts.forEach(function (p) {
        if (!p) return;
        if (/^\s+$/.test(p)) { frag.appendChild(doc.createTextNode(p)); return; }
        var w = doc.createElement('span'); w.className = 'w'; w.textContent = p; frag.appendChild(w);
      });
      t.parentNode.replaceChild(frag, t);
    });
    said.setAttribute('data-tok', '1');
  }

  // ------------------------------------------------------------------ the opening card: the read-back arrives, two-oh-eight turns red
  var opening = $('.fold-card .xcard');
  // full: words arrive, the model's spans underline, connectors, the red block, the verdict (~2.1 s to the verdict)
  // quick: a picked card settles in under a second, so ten cards can be compared without waiting
  function play(card, force, quick) {
    if (!card) return;
    (card._timers || []).forEach(clearTimeout);
    var T = card._timers = [];
    var cls = ['armed', 'p-point', 'p-link', 'p-slam', 'p-verdict', 'settled'];
    cls.forEach(function (c) { card.classList.remove(c); });
    if (!MOTION && !force) { drawLinks(card); return; }
    if (!MOTION) { drawLinks(card); return; }
    var turns = $$('.xc-turn', card);
    turns.forEach(function (t) { var s = $('.said', t); if (s) tokenize(s); });
    $$('.w', card).forEach(function (w) { w.classList.remove('on'); });
    void card.offsetWidth;
    card.classList.add('armed');
    var t0 = quick ? 0 : 100;
    turns.forEach(function (turn, i) {
      var ws = $$('.w', turn), step = quick ? 0 : Math.min(36, 260 / Math.max(ws.length, 1));
      ws.forEach(function (w, j) { T.push(later(function () { w.classList.add('on'); }, t0 + j * step)); });
      t0 += quick ? 0 : ws.length * step + (i === 0 ? 150 : 110);
    });
    var g = quick ? [60, 160, 320, 520, 1000] : [0, 180, 340, 600, 1500];
    T.push(later(function () { card.classList.add('p-point'); }, t0 + g[0]));
    T.push(later(function () { drawLinks(card); card.classList.add('p-link'); }, t0 + g[1]));
    T.push(later(function () { card.classList.add('p-slam'); }, t0 + g[2]));
    T.push(later(function () { card.classList.add('p-verdict'); }, t0 + g[3]));
    T.push(later(function () {
      cls.forEach(function (c) { card.classList.remove(c); });
      $$('.w', card).forEach(function (w) { w.classList.remove('on'); });
      card.classList.add('settled'); drawLinks(card);
    }, t0 + g[4]));
  }
  if (opening) {
    if (MOTION) later(function () { play(opening); }, 120); else drawLinks(opening);
    var again = $('#replay-strip');
    if (again) again.addEventListener('click', function () { play(opening, true); });
  }

  // ------------------------------------------------------------------ Confirm / Dismiss (needs a reason), like the queue
  function bindActions(card) {
    var act = $('.xc-act', card);
    if (!act || act.getAttribute('data-bound')) return;
    act.setAttribute('data-bound', '1');
    var stamp = $('.stamp', act), form = $('form.why', act);
    function done(msg) {
      $$('button[data-act]', act).forEach(function (b) { b.disabled = true; });
      if (form) form.hidden = true;
      stamp.textContent = msg;
    }
    var c = $('[data-act="confirm"]', act), d = $('[data-act="dismiss"]', act);
    if (c) c.addEventListener('click', function () { done('Confirmed · in the queue this card now leaves with your decision logged'); });
    if (d) d.addEventListener('click', function () { form.hidden = false; $('input', form).focus(); });
    if (form) form.addEventListener('submit', function (e) {
      e.preventDefault();
      var v = $('input', form).value.trim();
      if (!v) { $('input', form).reportValidity(); return; }
      done('Dismissed · reason logged: ' + v.slice(0, 120));
    });
  }
  $$('.xcard').forEach(bindActions);

  // ------------------------------------------------------------------ picker: ten real cards from the stored shift
  var list = $('#picklist');
  function pick(id, focusRow) {
    var rows = $$('.row', list);
    rows.forEach(function (r) { r.setAttribute('aria-selected', r.getAttribute('data-pick') === id ? 'true' : 'false'); });
    $$('[data-pick-detail]').forEach(function (d) { d.hidden = d.getAttribute('data-pick-detail') !== id; });
    var det = $('[data-pick-detail="' + id + '"]'), card = det && $('.xcard', det);
    if (card) { drawLinks(card); if (MOTION) play(card, true, true); }
    if (focusRow) { var r = $('.row[data-pick="' + id + '"]', list); if (r) { r.focus({ preventScroll: true }); r.scrollIntoView({ block: 'nearest' }); } }
    if (det && window.innerWidth < 1024) det.scrollIntoView({ behavior: MOTION ? 'smooth' : 'auto', block: 'start' });
    else if (det) {
      var r = det.getBoundingClientRect();
      if (r.top < 80 || r.top > window.innerHeight * 0.6) det.scrollIntoView({ behavior: MOTION ? 'smooth' : 'auto', block: 'start' });
    }
  }
  if (list) {
    // the opening row sits mid-list at laptop heights: bring it into the list's own view (never the page's)
    var sel0 = $('.row[aria-selected="true"]', list), rows0 = $$('.row', list), head0 = $('.log-h', list);
    if (sel0 && list.scrollHeight > list.clientHeight) {
      var k = rows0.indexOf(sel0), top = k, h = head0 ? head0.offsetHeight : 0;
      while (top > 0 && sel0.offsetTop + sel0.offsetHeight - rows0[top - 1].offsetTop <= list.clientHeight - h) top--;
      list.scrollTop = Math.max(0, rows0[top].offsetTop - h);  // whole rows, the selected one last in view
    }
    var edge = function () { list.classList.toggle('more', list.scrollHeight - list.clientHeight - list.scrollTop > 2); };
    edge();
    list.addEventListener('scroll', edge, { passive: true });
    window.addEventListener('resize', edge);
    list.addEventListener('click', function (e) { var r = e.target.closest('.row'); if (r) pick(r.getAttribute('data-pick')); });
    list.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
      var rows = $$('.row', list), i = rows.indexOf(doc.activeElement);
      if (i < 0) return;
      e.preventDefault();
      var n = rows[Math.max(0, Math.min(rows.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)))];
      pick(n.getAttribute('data-pick'), true);
    });
    $$('button.refused').forEach(function (b) {
      b.addEventListener('click', function () {
        pick(b.getAttribute('data-pick'));
        var d = $('[data-pick-detail="' + b.getAttribute('data-pick') + '"]');
        if (d && window.innerWidth >= 1024) d.scrollIntoView({ behavior: MOTION ? 'smooth' : 'auto', block: 'center' });
      });
    });
  }
  $$('details').forEach(function (d) { d.addEventListener('toggle', function () { drawAll(); }); });

  // ------------------------------------------------------------------ terminal: replay the recorded session
  var pre = $('#term pre'), termTimers = [];
  function typeTerm() {
    if (!pre) return;
    termTimers.forEach(clearTimeout); termTimers = [];
    var lines = $$('.ln', pre);
    if (!MOTION) { lines.forEach(function (l) { l.hidden = false; }); return; }
    lines.forEach(function (l) { l.hidden = true; if (l.getAttribute('data-full') === null) l.setAttribute('data-full', l.innerHTML); });
    pre.scrollTop = 0;
    var t = 200;
    lines.forEach(function (l) {
      var isCmd = l.classList.contains('cmd'), full = l.getAttribute('data-full');
      if (isCmd) {
        var txt = l.textContent, n = txt.length;
        termTimers.push(later(function () { l.hidden = false; l.textContent = ''; }, t));
        for (var k = 1; k <= n; k += 4) (function (k) { termTimers.push(later(function () { l.textContent = txt.slice(0, k); pre.scrollTop = pre.scrollHeight; }, t + k * 3)); })(k);
        t += n * 3 + 220;
        termTimers.push(later(function () { l.innerHTML = full; }, t));
      } else {
        termTimers.push(later(function () { l.hidden = false; pre.scrollTop = pre.scrollHeight; }, t));
        t += l.classList.contains('com') ? 200 : 34;
      }
    });
    termTimers.push(later(function () { pre.scrollTop = 0; }, t + 1600));
  }
  onView(pre, typeTerm, '0px 0px -25% 0px');
  var tp = $('#term-play'); if (tp) tp.addEventListener('click', typeTerm);
  $$('button.copy').forEach(function (b) {
    b.addEventListener('click', function () {
      var t = b.getAttribute('data-copy'), old = b.textContent;
      (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(function () { b.textContent = 'Copied'; }, function () { b.textContent = t; })
        .then(function () { later(function () { b.textContent = old; }, 1600); });
    });
  });

  // ------------------------------------------------------------------ count-ups (final values are already in the HTML)
  function countUp(el) {
    var to = parseInt(el.getAttribute('data-count'), 10);
    if (!MOTION || !to) return;
    var t0 = performance.now(), dur = 1100;
    (function tick(now) {
      var p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(2, -10 * p);
      el.textContent = String(Math.round(to * (p >= 1 ? 1 : e)));
      if (p < 1) requestAnimationFrame(tick);
    })(t0);
  }
  onView($('.receipts'), function () { $$('.receipts [data-count]').forEach(countUp); });

  // ------------------------------------------------------------------ the plant mimic and the ladder cells
  var mimic = $('#mimic');
  if (mimic && MOTION) {
    mimic.classList.add('armed');
    onView(mimic, function () { mimic.classList.remove('armed'); mimic.classList.add('play'); later(function () { mimic.classList.add('settled'); }, 3200); });
  }
  var rungs = $('.rungs');
  if (rungs) {
    $$('.cells', rungs).forEach(function (c) { $$('i', c).forEach(function (i, k) { i.style.setProperty('--i', k); }); });
    if (MOTION) {
      rungs.classList.add('armed');
      onView(rungs, function () { rungs.classList.remove('armed'); rungs.classList.add('play'); });
    }
  }

  // ------------------------------------------------------------------ nav: which section is in view
  var links = $$('.links a');
  if (links.length && 'IntersectionObserver' in window) {
    var map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (a) { a.removeAttribute('aria-current'); });
        var a = map[e.target.id]; if (a) a.setAttribute('aria-current', 'true');
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    Object.keys(map).forEach(function (id) { var s = doc.getElementById(id); if (s) io.observe(s); });
  }
  root.classList.add('ready');
})();
