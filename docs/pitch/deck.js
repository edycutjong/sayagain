/* SayAgain deck: navigation, overview (Esc), presenter notes (P), contrast check (C), fullscreen (F), touch swipe. */
(function () {
  'use strict';
  var doc = document, body = doc.body, stage = doc.getElementById('stage');
  var slides = [].slice.call(doc.querySelectorAll('.stage > .slide')), n = slides.length, cur = 0;
  var RM = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches, still = /(^|[?&])still\b/.test(location.search);
  function pad(k) { return String(k).padStart(2, '0'); }

  // footers + counters written once, so print and the overview carry them too
  slides.forEach(function (s, i) {
    var f = doc.createElement('div'); f.className = 'foot';
    f.innerHTML = '<span class="wm">SAYAGAIN</span><span>AMD Developer Hackathon · ACT III</span><span>' + (window.DECK_VERSION || '') + '</span><span class="ct">' + pad(i + 1) + ' / ' + pad(n) + '</span>';
    s.appendChild(f);
  });

  function fit() {
    var W = doc.documentElement.clientWidth, H = doc.documentElement.clientHeight * (body.classList.contains('presenter') ? .78 : 1);
    var sc = Math.min(W / 1920, H / 1080);
    stage.style.setProperty('--s', sc);
    doc.documentElement.style.setProperty('--s', sc); // the window background scales its ruled lines with the slide
    links();
  }

  // connectors between the words the model pointed at (same rule as the landing)
  function links() {
    [].forEach.call(doc.querySelectorAll('.slide.on .xcard'), function (card) {
      var strip = card.querySelector('.xc-strip'), svg = card.querySelector('svg.connectors');
      if (!strip || !svg) return;
      var ins = card.querySelector('.xc-turn[data-t="' + card.getAttribute('data-ins') + '"]');
      var rb = card.querySelector('.xc-turn[data-t="' + card.getAttribute('data-rb') + '"]');
      svg.innerHTML = '';
      if (!ins || !rb) return;
      var k = card.getBoundingClientRect().width / card.offsetWidth || 1, box = strip.getBoundingClientRect();
      [].forEach.call(ins.querySelectorAll('.sp'), function (a) {
        var b = rb.querySelector('.sp[data-f="' + a.getAttribute('data-f') + '"][data-c="' + a.getAttribute('data-c') + '"]');
        if (!b) return;
        var A = a.getBoundingClientRect(), B = b.getBoundingClientRect();
        var ln = doc.createElementNS('http://www.w3.org/2000/svg', 'line');
        ln.setAttribute('x1', (A.left + A.width / 2 - box.left) / k); ln.setAttribute('y1', (A.bottom - box.top) / k + 4);
        ln.setAttribute('x2', (B.left + B.width / 2 - box.left) / k); ln.setAttribute('y2', (B.top - box.top) / k - 4);
        if (/v-MISMATCH/.test(card.className) && b.querySelector('mark.red')) ln.setAttribute('class', 'red');
        else if (a.classList.contains('amb')) ln.setAttribute('class', 'amb');
        svg.appendChild(ln);
      });
    });
  }
  window.addEventListener('resize', fit);
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(fit);
  fit();

  function go(k, fromHash) {
    k = Math.max(0, Math.min(n - 1, k));
    if (k === cur && !fromHash) return;
    slides[cur].classList.remove('on'); cur = k; slides[cur].classList.add('on');
    if (!fromHash) history.replaceState(null, '', '#' + (cur + 1));
    notes(); links();
  }
  function notes() {
    var a = slides[cur].querySelector('.speaker-notes');
    doc.getElementById('noteText').textContent = a ? a.textContent.trim() : '';
    var nx = slides[cur + 1];
    doc.getElementById('noteMeta').innerHTML = '<b>' + pad(cur + 1) + ' / ' + n + '</b><br>' + (nx ? 'next: ' + nx.dataset.title : 'last slide') + '<br>elapsed <span id="tm">' + elapsed() + '</span>';
  }
  var t0 = Date.now();
  function elapsed() { var s = Math.floor((Date.now() - t0) / 1000); return Math.floor(s / 60) + ':' + pad(s % 60); }
  setInterval(function () { var e = doc.getElementById('tm'); if (e) e.textContent = elapsed(); }, 1000);

  var ov = doc.getElementById('overview'), built = false;
  function overview(show) {
    if (show && !built) {
      slides.forEach(function (s, i) {
        var t = doc.createElement('button'); t.type = 'button'; t.className = 'thumb'; t.setAttribute('aria-label', 'Go to slide ' + (i + 1) + ', ' + s.dataset.title);
        var c = s.cloneNode(true); c.classList.add('on'); c.removeAttribute('aria-label');
        [].forEach.call(c.querySelectorAll('[id]'), function (e) { e.removeAttribute('id'); });
        t.appendChild(c);
        var lb = doc.createElement('b'); lb.textContent = pad(i + 1) + ' · ' + s.dataset.title; t.appendChild(lb);
        t.addEventListener('click', function () { go(i); overview(false); });
        ov.appendChild(t);
      });
      built = true;
    }
    body.classList.toggle('show-overview', show);
    if (show) {
      [].forEach.call(ov.children, function (t, i) {
        t.classList.toggle('cur', i === cur);
        t.firstChild.style.transform = 'scale(' + (t.clientWidth / 1920) + ')';
      });
      (ov.children[cur] || ov).focus();
    }
  }
  function presenter() { body.classList.toggle('presenter'); fit(); notes(); }
  var idleT, mouseT;
  doc.addEventListener('mousemove', function () {
    body.classList.add('mouse'); clearTimeout(mouseT); mouseT = setTimeout(function () { body.classList.remove('mouse'); }, 2500);
    body.classList.remove('idle'); clearTimeout(idleT);
    if (body.classList.contains('presenter')) idleT = setTimeout(function () { body.classList.add('idle'); }, 2000);
  });
  doc.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var k = e.key, open = body.classList.contains('show-overview');
    if (k === 'Escape') { overview(!open); e.preventDefault(); return; }
    if (open) return;
    if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'Enter') { go(cur + 1); e.preventDefault(); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'Backspace') { go(cur - 1); e.preventDefault(); }
    else if (k === 'Home') go(0);
    else if (k === 'End') go(n - 1);
    else if (k === 'p' || k === 'P') presenter();
    else if (k === 'c' || k === 'C') body.classList.toggle('debug-contrast');
    else if ((k === 'f' || k === 'F') && doc.documentElement.requestFullscreen) { doc.fullscreenElement ? doc.exitFullscreen() : doc.documentElement.requestFullscreen(); }
  });
  doc.getElementById('prev').addEventListener('click', function () { go(cur - 1); });
  doc.getElementById('next').addEventListener('click', function () { go(cur + 1); });
  doc.getElementById('grid').addEventListener('click', function () { overview(true); });
  doc.getElementById('notes').addEventListener('click', presenter);
  var tx = null, vp = doc.getElementById('viewport');
  vp.addEventListener('touchstart', function (e) { tx = e.touches[0].clientX; }, { passive: true });
  vp.addEventListener('touchend', function (e) {
    if (tx === null) return; var dx = e.changedTouches[0].clientX - tx; tx = null;
    if (Math.abs(dx) > 40) go(cur + (dx < 0 ? 1 : -1));
    else go(cur + (e.changedTouches[0].clientX > doc.documentElement.clientWidth / 2 ? 1 : -1));
  });
  window.addEventListener('hashchange', function () { var h = parseInt(location.hash.slice(1), 10); if (h) go(h - 1, true); });
  window.addEventListener('beforeprint', function () { slides.forEach(function (s) { s.classList.add('on'); }); });
  window.addEventListener('afterprint', function () { slides.forEach(function (s, i) { s.classList.toggle('on', i === cur); }); });
  var h0 = parseInt(location.hash.slice(1), 10); if (h0 > 1) go(h0 - 1, true);
  notes();
  if (still) body.classList.add('still');
  setTimeout(function () { var h = doc.querySelector('.hint'); if (h) h.classList.add('gone'); }, 6000);
  if (!RM && !still) body.classList.add('play');
  setTimeout(links, 3600);
})();
