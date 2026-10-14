/* SayAgain supervisor queue. One client for two sources:
   - live:   the FastAPI server (sayagain/server.py) streams turn / judged / end_of_shift events over SSE;
   - replay: the static export (scripts/export_run.py) ships the same events in data/<run>--<rung>.json.
   The page renders only what the events carry: verdicts, field diffs, spans and receipts. No number is typed in here. */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var LANG = { en: 'English', id: 'Bahasa Indonesia', mix: 'English + Bahasa' };
  var VERDICT_WORD = { MISMATCH: 'MISMATCH', ABSTAIN: 'ABSTAIN', NO_READBACK: 'NO-READBACK', NOT_AUDITED: 'NOT AUDITED', VERIFIED: 'VERIFIED' };
  var TICK = '<svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.6l3.1 3.1L13 4.8"/></svg>';
  var DISMISS_REASONS = [
    'The read-back was correct: the referee is wrong',
    'Not a read-back (chatter or status talk)',
    'The plant registry is out of date',
  ];

  var S = {
    cfg: null, mode: 'replay', run: null, rung: null, meta: null, extract: 'stored',
    cards: {}, order: [], liveIds: [], eos: null, reviews: {}, selected: null, follow: true, tab: 'live',
    speed: 8, paused: false, done: false, src: null, cardsDone: 0, clockBase: 0, lastAt: 0,
    data: null, chime: false, audio: null, tried: null,
  };

  // ------------------------------------------------------------------------------------------- utilities
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function clock(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    var mm = (m < 10 ? '0' : '') + m, ss = (s < 10 ? '0' : '') + s;
    return h ? h + ':' + mm + ':' + ss : mm + ':' + ss;
  }
  function usdFixed(x) {
    if (typeof x !== 'number') return null;
    if (x === 0) return '$0';
    return '$' + x.toFixed(Math.min(12, Math.max(2, 2 - Math.floor(Math.log10(Math.abs(x))))));
  }
  function secs(x) { return typeof x === 'number' ? x.toFixed(2) + ' s' : null; }
  function getJSON(url) {
    return fetch(url, { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
      return r.json();
    });
  }
  function vkey(card) { return card && card.verdict ? card.verdict.verdict : null; }
  function isSpot(id) { return !!(S.eos && S.eos.spot_checks.indexOf(id) >= 0); }
  function reviewKey() { return 'sayagain.reviews.' + S.run + '.' + S.rung; }
  function qs(k) { return new URLSearchParams(location.search).get(k); }

  // ------------------------------------------------------------------------------------------- boot
  function boot() {
    getJSON('config.json').then(function (cfg) {
      S.cfg = cfg;
      S.mode = cfg.mode === 'live' ? 'live' : 'replay';
      var sel = $('rung');
      cfg.runs.forEach(function (r) {
        var g = document.createElement('optgroup');
        g.label = 'run ' + r.run + (r.n_cards ? ' · ' + r.n_cards + ' cards' : '');
        r.rungs.forEach(function (rg) {
          var o = document.createElement('option');
          o.value = r.run + '|' + rg;
          o.textContent = rg === 'regex' ? 'regex · rung 0, no model' : rg;
          g.appendChild(o);
        });
        sel.appendChild(g);
      });
      var want = (qs('run') || cfg.default.run) + '|' + (qs('rung') || cfg.default.rung);
      sel.value = Array.prototype.some.call(sel.options, function (o) { return o.value === want; })
        ? want : cfg.default.run + '|' + cfg.default.rung;
      sel.addEventListener('change', function () { loadRun(); });
      if (S.mode === 'live' && cfg.live_extract) {
        $('extract').hidden = false; $('extract-lab').hidden = false;
        $('extract').addEventListener('change', function () { S.extract = $('extract').value; loadRun(); });
      }
      var m = $('mode');
      m.textContent = S.mode === 'live' ? 'Live server · referee judges each event' : 'Static replay · verdicts precomputed';
      m.classList.toggle('live', S.mode === 'live');
      var sp = Number(qs('speed'));
      if ([1, 8, 32].indexOf(sp) >= 0) S.speed = sp;
      syncSpeed();
      loadRun();
    }).catch(fatal);
    wire();
  }

  function fatal(e) {
    $('card-body').innerHTML = '<div class="placeholder"><p class="lab">Could not load the shift</p><p>' + esc(e.message || e) + '</p></div>';
  }

  function wire() {
    $('play').addEventListener('click', function () {
      if (S.done) { loadRun(); return; }
      S.paused ? resume() : pause();
    });
    Array.prototype.forEach.call(document.querySelectorAll('.seg button'), function (b) {
      b.addEventListener('click', function () {
        S.speed = Number(b.getAttribute('data-speed'));
        syncSpeed();
        if (!S.paused && !S.done) { stopSource(); openSource(S.speed); }
      });
    });
    $('skip').addEventListener('click', function () {
      if (S.done) { showTab('eos'); return; }
      stopSource(); S.paused = false; syncPlay(); openSource(0);
    });
    $('refused').addEventListener('click', tryRefused);
    $('follow').addEventListener('click', function () {
      setFollow(!S.follow);
      if (S.follow) {
        S.tried = null;
        var last = openLive().slice(-1)[0];
        if (last) select(last, false);
      }
    });
    $('chime').addEventListener('click', function () {
      S.chime = !S.chime;
      $('chime').setAttribute('aria-pressed', String(S.chime));
      $('chime').querySelector('.t').textContent = S.chime ? 'Chime on' : 'Chime off';

      if (S.chime) chime(true);
    });
    ['live', 'eos', 'all'].forEach(function (t) { $('tab-' + t).addEventListener('click', function () { showTab(t); }); });
    document.querySelector('.tabs').addEventListener('keydown', function (e) {
      var tabs = ['live', 'eos', 'all'], i = tabs.indexOf(S.tab);
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        var n = tabs[(i + (e.key === 'ArrowRight' ? 1 : 2)) % 3];
        showTab(n); $('tab-' + n).focus();
      }
    });
    $('back').addEventListener('click', closeCard);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && document.body.classList.contains('card-open')) closeCard();
      if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      if (e.key === 'j' || e.key === 'k') moveSel(e.key === 'j' ? 1 : -1);
    });
    var ro = window.ResizeObserver ? new ResizeObserver(function () { drawLinks(); }) : null;
    if (ro) ro.observe($('card-body')); else window.addEventListener('resize', drawLinks);
  }

  // ------------------------------------------------------------------------------------------- runs + sources
  function storedReviews() {
    try {
      var v = JSON.parse(localStorage.getItem(reviewKey()) || '[]');
      return Array.isArray(v) ? v.filter(function (r) { return r && typeof r === 'object' && typeof r.exchange_id === 'string'; }) : [];
    }
    catch (e) { return []; }
  }
  function loadRun() {
    stopSource();
    var isRegex = $('rung').value.split('|')[1] === 'regex';
    var liveOpt = $('extract').querySelector('option[value="live"]');
    if (liveOpt) liveOpt.disabled = isRegex;  // rung 0 makes no model call: nothing to call live
    if (isRegex && S.extract === 'live') { S.extract = 'stored'; $('extract').value = 'stored'; }
    S.gen = (S.gen || 0) + 1;
    var gen = S.gen;
    (S.timers || []).forEach(clearTimeout); S.timers = [];
    $('refused').disabled = false;
    var v = $('rung').value.split('|');
    S.run = v[0]; S.rung = v[1];
    S.cards = {}; S.order = []; S.liveIds = []; S.eos = null; S.selected = null; setFollow(true);
    S.done = false; S.paused = false; S.cardsDone = 0; S.clockBase = 0; S.lastAt = 0; S.data = null; S.tried = null;
    syncPlay();
    $('count').textContent = '0';
    $('count-lab').innerHTML = S.mode === 'live' && S.extract === 'live' ? 'Cards<br>this run' : 'Cards in<br>this replay';
    $('clock').textContent = '00:00';
    $('progbar').style.width = '0';
    $('ch-who').textContent = '·';
    $('ch-txt').textContent = 'Waiting for the first transmission.';
    $('card-body').innerHTML = '<div class="placeholder"><p class="lab">Alert card</p><p>The first card opens here when the referee flags a read-back.</p></div>';
    var u = new URL(location.href);
    u.searchParams.set('run', S.run); u.searchParams.set('rung', S.rung);
    history.replaceState(null, '', u);
    var metaP = S.mode === 'live'
      ? getJSON('api/shift?run=' + encodeURIComponent(S.run) + '&rung=' + encodeURIComponent(S.rung))
      : getJSON('data/' + S.run + '--' + S.rung + '.json').then(function (d) { S.data = d; return d.meta; });
    var revP = S.mode === 'live'
      ? getJSON('api/reviews?run=' + encodeURIComponent(S.run) + '&rung=' + encodeURIComponent(S.rung))
      : Promise.resolve(storedReviews());
    Promise.all([metaP, revP]).then(function (res) {
      if (gen !== S.gen) return;  // a newer rung was picked while this one loaded
      S.meta = res[0];
      S.reviews = {};
      res[1].forEach(function (r) { S.reviews[r.exchange_id] = r; });
      renderStages();
      renderList();
      openSource(S.speed);
    }).catch(fatal);
  }

  function openSource(speed) {
    S.paused = false; syncPlay();
    if (S.mode === 'live') {
      var url = 'api/stream?run=' + encodeURIComponent(S.run) + '&rung=' + encodeURIComponent(S.rung) +
        '&speed=' + speed + '&start=' + S.cardsDone + '&extract=' + S.extract;
      var es = new EventSource(url);
      // stored replays carry the shift clock; live model calls run on wall time from where the last stream stopped
      S.clockBase = S.extract === 'live' ? S.lastAt : 0;
      es.onmessage = function (m) {
        var ev = JSON.parse(m.data);
        if (ev.type === 'closed') es.close();  // or the browser reconnects and replays the tail into the next run
        onEvent(ev);
      };
      es.onerror = function () { if (!S.done) { es.close(); notice('Stream interrupted. Press play to resume from the last judged card.'); S.paused = true; syncPlay(); } };
      S.src = { close: function () { es.close(); } };
    } else {
      S.src = localSource(speed);
    }
  }

  function localSource(speed) {
    // replays data.events from the first card not judged yet, on the same replay clock the server uses
    var evs = S.data.events, i = 0, judged = 0;
    while (i < evs.length && judged < S.cardsDone) { if (evs[i].type === 'judged') judged++; i++; }
    var startAt = i < evs.length ? evs[i].at : 0, t0 = performance.now(), timer = null, stopped = false;
    S.clockBase = 0;
    function step() {
      if (stopped) return;
      if (speed === 0) {
        while (i < evs.length) onEvent(evs[i++], true);
        flushBulk();
        onEvent({ type: 'closed' });
        return;
      }
      var now = (performance.now() - t0) / 1000 * speed + startAt;
      while (i < evs.length && evs[i].at <= now) onEvent(evs[i++]);
      if (i >= evs.length) { onEvent({ type: 'closed' }); return; }
      timer = setTimeout(step, Math.max(30, Math.min(400, (evs[i].at - now) / speed * 1000)));
    }
    step();
    return { close: function () { stopped = true; clearTimeout(timer); } };
  }

  function notice(msg) {
    $('ch-sub').textContent = msg;
    $('ch-who').textContent = 'Notice';
    $('ch-txt').textContent = msg;
  }
  function setFollow(on) {
    S.follow = on;
    $('follow').setAttribute('aria-pressed', String(on));
  }
  function stopSource() { if (S.src) { S.src.close(); S.src = null; } }
  function pause() { stopSource(); S.paused = true; syncPlay(); }
  function resume() { openSource(S.speed); }
  function syncPlay() {
    var b = $('play');
    b.setAttribute('aria-pressed', String(S.paused || S.done));
    b.querySelector('.t').textContent = S.done ? 'Replay the shift' : S.paused ? 'Play' : 'Pause';
    b.setAttribute('aria-label', b.querySelector('.t').textContent);
    $('skip').textContent = S.done ? 'Open the end-of-shift list' : 'Skip to end of shift';
  }
  function syncSpeed() {
    Array.prototype.forEach.call(document.querySelectorAll('.seg button'), function (b) {
      b.setAttribute('aria-pressed', String(Number(b.getAttribute('data-speed')) === S.speed));
    });
  }

  // ------------------------------------------------------------------------------------------- events
  var bulk = false;
  function onEvent(ev, isBulk) {
    bulk = !!isBulk;
    if (ev.type === 'turn') {
      var at = S.clockBase + ev.at;
      S.lastAt = Math.max(S.lastAt, at);
      if (!bulk) {
        $('clock').textContent = clock(at);
        var t = ev.turn;
        $('ch-who').textContent = t.role === 'operator' ? 'Operator' : t.role === 'field' ? 'Field tech' : t.role;
        var tx = $('ch-txt');
        tx.textContent = t.text;
        tx.classList.remove('in'); void tx.offsetWidth; tx.classList.add('in');
        $('ch-line').classList.add('tx');
      }
    } else if (ev.type === 'judged') {
      var c = ev.card;
      c._at = S.clockBase + ev.at;
      c._extract = ev.extract || 'stored';
      S.lastAt = Math.max(S.lastAt, c._at);
      if (!S.cards[c.exchange_id]) S.order.push(c.exchange_id);
      S.cards[c.exchange_id] = c;
      S.cardsDone = S.order.length;
      if (c.live && S.liveIds.indexOf(c.exchange_id) < 0) {
        S.liveIds.push(c.exchange_id);
        if (!bulk) {
          chime();
          $('sr-live').textContent = 'New card: ' + c.exchange_id + ', ' + (c.verdict ? c.verdict.label : '');
          var n = $('count'); n.classList.remove('bump'); void n.offsetWidth; n.classList.add('bump');
        }
      }
      if (!bulk) {
        $('clock').textContent = clock(c._at);
        if (!c.live && vkey(c) === null) $('ch-line').classList.remove('tx');
        afterJudged(c);
      }
    } else if (ev.type === 'end_of_shift') {
      S.eos = ev;
      if (!bulk) { renderList(); renderEos(); }
    } else if (ev.type === 'closed') {
      S.done = true; S.src = null; syncPlay();
      $('ch-line').classList.remove('tx');
      $('ch-who').textContent = 'Shift over';
      $('ch-txt').textContent = 'Error-dense test corpus: ' + S.liveIds.length + ' of ' + S.cardsDone + ' exchanges sent to the supervisor';
      if (S.eos) $('clock').textContent = clock(S.clockBase + S.eos.at);
      renderSub();
    }
  }
  function flushBulk() {
    bulk = false;
    renderList(); renderEos(); renderSub();
    if (S.follow) {
      var last = S.liveIds.filter(function (id) { return !S.reviews[id]; }).slice(-1)[0];
      if (last && !S.tried) select(last, false);
    }
    if (S.eos) $('clock').textContent = clock(S.clockBase + S.eos.at);
  }

  function afterJudged(c) {
    $('count').textContent = String(S.liveIds.length);
    renderSub();
    if (c.live) {
      if (S.tab === 'live' || S.tab === 'all') prependRow(c);
      if (S.follow && !S.tried) select(c.exchange_id, true);
    } else if (S.tab === 'all') {
      prependRow(c);
    }
    $('n-live').textContent = String(openLive().length);
    $('n-all').textContent = String(S.order.length);
    $('rows-empty').hidden = $('rows').children.length > 0 || S.tab === 'eos';
  }

  function renderSub() {
    var total = S.meta ? S.meta.n_cards_run || S.meta.n_cards || 0 : 0;
    var pct = total ? Math.min(100, S.cardsDone / total * 100) : 0;
    $('progbar').style.width = pct + '%';
    $('count').textContent = String(S.liveIds.length);
    $('n-live').textContent = String(openLive().length);
    $('n-all').textContent = String(S.order.length);
    $('n-eos').textContent = S.eos ? String(eosIds().length) : '·';
    var step = S.meta && S.meta.replay_clock ? S.meta.replay_clock.turn_step_s : null;
    var speedTxt = S.mode === 'live' && S.extract === 'live' ? 'live model calls, wall clock'
      : 'scripted timing' + (step ? ' (' + step + ' s a turn)' : '') + ' played at ' + S.speed + '×';
    $('ch-sub').textContent = S.cardsDone + ' of ' + total + ' exchanges judged · ' + (S.done ? 'shift over' : speedTxt) +
      ' · a read-back that matches is logged, not queued';
  }

  // ------------------------------------------------------------------------------------------- stages banner
  function renderStages() {
    var m = S.meta, st = m.stages, ho = m.held_out || {};
    var kn = function (w) { return w && w.n ? w.k + '/' + w.n : null; };
    var parts = [];
    if (kn(ho.false_verified)) parts.push('<span class="nw"><b>' + esc(kn(ho.false_verified)) + '</b> wrong read-backs passed (false-VERIFIED)</span>');
    if (kn(ho.errors_flagged)) parts.push('<span class="nw"><b>' + esc(kn(ho.errors_flagged)) + '</b> seeded errors flagged</span>');
    if (kn(ho.false_alarm_load)) parts.push('<span class="nw"><b>' + esc(kn(ho.false_alarm_load)) + '</b> false alarms</span>');
    if (ho.not_audited && ho.not_audited.k) parts.push('<span class="nw"><b>' + esc(kn(ho.not_audited)) + '</b> not audited</span>');
    var lead = S.mode === 'live'
      ? 'Live queue over run <b>' + esc(m.run) + '</b>'
      : 'Replay of run <b>' + esc(m.run) + '</b>';
    if (m.reaudit_of) lead += ' (a re-audit of <span class="data">' + esc(m.reaudit_of) + '</span>: only the transcripts whose call failed there were requested again)';
    if (!ho.n) lead += ' · the development slice of ' + esc(m.n_cards_run) + ' cards, used for tuning only: it has no held-out figures';
    if (S.mode === 'live' && S.extract === 'live') lead += ' · <b>model calls are live now</b>; the held-out figures and the stage p50 below are the stored run\u2019s, not these calls';
    var cps = m.cards_per_shift_assumption, cpsTxt = '';
    if (cps) {
      var lo = cps['2_errors_per_180'], hi = cps['5_errors_per_180'];
      var range = Math.round(lo) === Math.round(hi) ? esc(Math.round(lo)) : esc(Math.round(lo)) + '\u2013' + esc(Math.round(hi));
      if (lo != null && hi != null) cpsTxt = ' The test corpus is dense with seeded errors; at the protocol\u2019s shift assumption (' + esc(m.shift_assumption || '') +
        ') this rung would send about ' + range + ' cards a shift' + (ho.not_audited && ho.not_audited.k ? ' (an exchange not audited counts as no card here)' : '') + '.';
    }
    $('banner').innerHTML = lead + ' · fictional plant radio traffic · ' + esc(m.provenance) + ': each turn is the written script of a recording card, there is no recording yet' +
      (parts.length ? ' · this rung on the ' + esc(ho.n || '') + ' held-out cards: ' + parts.join(' · ') : '') +
      ' · <span class="data">' + esc(m.result_file) + '</span>.' + cpsTxt;
    $('stage-sum').setAttribute('data-short', 'fictional radio · no audio · ' + (st.extract.on_amd ? 'AMD-hosted extraction' : 'local extraction'));
    $('stage-sum').innerHTML = 'Run <b>' + esc(m.run) + '</b> · fictional radio, written scripts, <b>no audio</b> · extraction <b>' +
      esc(st.extract.name) + '</b>' + (st.extract.on_amd ? ' on an <b>AMD-hosted API</b>' : '') + ' · referee <b>local</b>';
    var rf = m.refused;
    $('refused-tag').textContent = rf ? rf.tag : '';
    $('refused').disabled = !rf;
    $('refused').title = rf ? 'Judge ' + rf.exchange_id + ': ' + rf.tag + ' is not in the plant registry' : 'The refused exchange is not in this run';
    var liveX = S.mode === 'live' && S.extract === 'live';
    Array.prototype.forEach.call(document.querySelectorAll('.seg button'), function (b) { b.disabled = liveX; });
    $('skip').disabled = liveX;
    $('clock-lab').textContent = liveX ? 'Wall clock T+' : 'Replay T+';
    var ex = st.extract;
    var exVal = esc(ex.name) + (ex.model_served && ex.model_served.length ? ' <span class="data">(' + esc(ex.model_served.join(', ')) + ')</span>' : '') +
      (ex.hosts && ex.hosts.length ? ' · ' + esc(ex.where) + ' · ' + esc(ex.hosts.join(', ')) : ' · ' + esc(ex.where)) +
      (ex.latency_s_p50 != null ? ' · p50 ' + esc(ex.latency_s_p50.toFixed(2)) + ' s over the ' + esc(ex.calls) + ' successful calls behind this run' +
        (ex.calls_reused ? ' (' + esc(ex.calls_reused) + ' reused from <span class="data">' + esc(ex.reused_from) + '</span>, so this differs from the result file\u2019s own-call figure)' : '') : '') +
      (ex.on_amd ? '<span class="where">AMD-hosted API</span>' : '<span class="where off">local</span>');
    var refVal = esc(st.referee.name) + ' · ' + (S.mode === 'live' ? 'judging in this server' : 'run at export (' + esc(m.referee_measured_on) + '), trace kept') + '<span class="where off">local</span>';
    $('stage-list').innerHTML =
      stage('1', 'Speech-to-text', esc(st.asr.name) + '<span class="where off">no audio</span>', true) +
      stage('2', 'Extraction', exVal, false) +
      stage('3', 'Referee', refVal, false);
    $('foot-run').innerHTML = 'Run <span class="data">' + esc(m.run) + '</span> · rung <span class="data">' + esc(m.rung) +
      '</span> · finished ' + esc(m.finished_at || '') + ' · git ' + esc(m.git_head || '') + ' · receipts in <span class="data">bench/results/receipts/</span>';
  }
  function stage(k, name, val, none) {
    return '<li class="stage' + (none ? ' none' : '') + '"><span class="k">' + k + '</span><span class="nm">' + name + '</span><span class="vl">' + val + '</span></li>';
  }

  // ------------------------------------------------------------------------------------------- the list
  function openLive() { return S.liveIds.filter(function (id) { return !S.reviews[id]; }); }
  function listIds() {
    if (S.tab === 'all') return S.order.slice().reverse();
    var open = openLive().slice().reverse();
    var done = S.liveIds.filter(function (id) { return S.reviews[id]; }).reverse();
    return open.concat(done);
  }
  function showTab(t) {
    S.tab = t;
    ['live', 'eos', 'all'].forEach(function (k) {
      var b = $('tab-' + k);
      b.setAttribute('aria-selected', String(k === t));
      b.tabIndex = k === t ? 0 : -1;
    });
    $('rows').hidden = t === 'eos';
    document.querySelector('.cols').hidden = t === 'eos';
    $('eos').hidden = t !== 'eos';
    if (t === 'eos') renderEos(); else renderList();
  }
  function renderList() {
    if (S.tab === 'eos') return;
    var ol = $('rows');
    ol.innerHTML = listIds().map(function (id) { return rowHTML(S.cards[id], false); }).join('');
    bindRows(ol);
    $('rows-empty').hidden = ol.children.length > 0;
    $('n-live').textContent = String(openLive().length);
    $('n-all').textContent = String(S.order.length);
  }
  function prependRow(c) {
    var ol = $('rows');
    if (S.tab === 'live') {  // keep open cards above reviewed ones
      var tmp = document.createElement('ol');
      tmp.innerHTML = rowHTML(c, true);
      var li = tmp.firstChild;
      ol.insertBefore(li, ol.firstChild);
      bindRows(li);
    } else {
      var t2 = document.createElement('ol');
      t2.innerHTML = rowHTML(c, c.live);
      var li2 = t2.firstChild;
      ol.insertBefore(li2, ol.firstChild);
      bindRows(li2);
    }
    $('rows-empty').hidden = true;
  }
  function rowFields(c) {
    var v = c.verdict;
    if (!v && c.orphans && c.orphans.length) return { f: 'read-back', ft: '<span>no order heard · end of shift</span>' };
    if (!v) return { f: 'no instruction', ft: '<span>chatter · no card</span>' };
    if (v.diffs && v.diffs.length) {
      var d = v.diffs[0], more = v.diffs.length > 1 ? ' +' + (v.diffs.length - 1) : '';
      if (d.field === 'command') return { f: 'command', ft: '<b>' + esc(d.note) + '</b>' + more };
      return { f: esc(d.field), ft: '<span class="ok nw">' + esc(d.expected || '—') + '\u00a0→</span> <b>' + esc(d.got || 'missing') + '</b>' + more };
    }
    if (v.verdict === 'VERIFIED') {
      var ins = c.turns.filter(function (t) { return c.strip && t.id === c.strip.instruction; })[0];
      var cm = ins && ins.commands[0];
      return { f: '', ft: cm ? '<span class="ok">' + esc(cm.action || '') + ' <span class="nw">' + esc(cm.tag || '') + '</span>' + (cm.line ? ' <span class="nw">line ' + esc(cm.line) + '</span>' : '') + '</span>' : '' };
    }
    if (v.verdict === 'NO_READBACK') return { f: 'read-back', ft: '<span>no reply in the window</span>' };
    return { f: esc(v.reason || ''), ft: esc(tagOf(c) || '') };
  }
  function tagOf(c) {
    var ins = c.turns.filter(function (t) { return c.strip && t.id === c.strip.instruction; })[0];
    return ins && ins.commands[0] ? ins.commands[0].tag : null;
  }
  function rowHTML(c, isNew) {
    var vk = vkey(c), spot = vk === 'VERIFIED' && isSpot(c.exchange_id);
    var rf0 = rowFields(c);
    var cls = 'row v-' + (spot ? 'SPOT' : vk || 'none') + (isNew ? ' new' : '') + (S.reviews[c.exchange_id] ? ' reviewed' : '') + (rf0.f ? '' : ' nof');
    var word = spot ? 'SPOT-CHECK' : vk ? VERDICT_WORD[vk] : '·';
    if (vk === 'VERIFIED' && !spot) word = TICK + ' VERIFIED';
    var rf = rowFields(c), r = S.reviews[c.exchange_id];
    return '<li><button type="button" class="' + cls + '" data-id="' + esc(c.exchange_id) + '"' +
      (S.selected === c.exchange_id ? ' aria-current="true"' : '') + ' aria-label="' + esc(c.exchange_id + ', ' + (c.verdict ? c.verdict.label : 'no card')) + '">' +
      '<span class="t">' + clock(c._at) + '</span><span class="c">' + esc((S.meta && S.meta.channel || 'CH-01').replace('CH-', '')) + '</span>' +
      '<span class="vw">' + word + (r ? '<span class="stamp">' + esc(r.decision) + '</span>' : '') + '</span>' +
      '<span class="f">' + rf.f + '</span><span class="ft">' + rf.ft + '</span></button></li>';
  }
  function bindRows(root) {
    Array.prototype.forEach.call(root.querySelectorAll ? root.querySelectorAll('.row') : [], function (b) {
      b.addEventListener('click', function () { setFollow(false); S.tried = null; select(b.getAttribute('data-id'), false, true); });
    });
  }
  function moveSel(dir) {
    var ids = S.tab === 'eos' ? eosIds() : listIds();
    if (!ids.length) return;
    var i = ids.indexOf(S.selected);
    var n = ids[Math.max(0, Math.min(ids.length - 1, i < 0 ? 0 : i + dir))];
    setFollow(false); S.tried = null;
    select(n, false, false);
    var el = document.querySelector('.row[data-id="' + n + '"]');
    if (el) { el.focus(); el.scrollIntoView({ block: 'nearest' }); }
  }

  // ------------------------------------------------------------------------------------------- end of shift
  function eosIds() {
    if (!S.eos) return [];
    var seen = {};
    return S.eos.spot_checks.concat(S.eos.caught_on_channel, S.eos.not_audited, orphansOf(S.eos)).filter(function (id) { return !seen[id] && (seen[id] = 1); });
  }
  function orphansOf(e) { return (e && e.orphan_readbacks) || []; }  // replays exported before the list existed have none
  function renderEos() {
    var box = $('eos');
    if (S.tab !== 'eos') { renderSub(); return; }
    if (!S.eos) {
      box.innerHTML = '<p class="pending">The end-of-shift list is drawn when the shift ends: ' + esc(S.meta ? S.meta.spot_checks_per_shift : '') +
        ' random VERIFIED exchanges to re-check, the read-backs the operator corrected on the channel, and every exchange nobody audited.</p>' +
        '<p class="pending"><button type="button" class="btn" id="eos-skip">Skip to end of shift</button></p>';
      $('eos-skip').addEventListener('click', function () { $('skip').click(); });
      return;
    }
    var e = S.eos;
    var orph = orphansOf(e), all = e.spot_checks.length + e.caught_on_channel.length + e.not_audited.length + orph.length, dup = all - eosIds().length;
    box.innerHTML = (dup > 0 ? '<p class="why eos-note">' + esc(eosIds().length) + ' exchanges; ' + esc(dup) + (dup === 1 ? ' appears' : ' appear') + ' in two lists.</p>' : '') +
      section('Spot-checks', e.spot_checks, 'Random VERIFIED exchanges a human re-checks at the end of the shift: how the plant learns whether the referee lets errors through.', 'No VERIFIED exchange to sample this shift.') +
      caughtSection(e.caught_on_channel) +
      section('Not audited', e.not_audited, 'Each of these has at least one turn whose model call failed, so part of it was never judged. They are listed, never passed.', 'Every exchange was judged.') +
      (orph.length ? section('Read-backs with no order heard', orph, 'A field read-back with no instruction heard before it on the channel: work nobody was heard ordering. Listed for a human, never passed.', '') : '');
    bindRows(box);
  }
  function caughtSection(ids) {
    var n = ids.reduce(function (a, id) { return a + ((S.cards[id] && S.cards[id].caught) || []).length; }, 0);
    var title = '<h3>Caught on channel<span class="n">' + n + '</span></h3>';
    if (!n) return title + '<p class="why">No read-back was corrected on air this shift.</p>';
    return title + '<p class="why">The operator heard a wrong read-back and corrected it on air. Each row shows that first, wrong read-back; open it to see how the exchange ended.</p><ol class="rows">' +
      ids.map(function (id) {
        var c = S.cards[id];
        if (!c) return '';
        return (c.caught || []).map(function (k) {
          var diff = String(k.label).split(' · ').slice(1).join(' · ');
          var m = /^(\w+): (.+) ≠ (.+)$/.exec(diff);
          return '<li><button type="button" class="row v-CAUGHT" data-id="' + esc(id) + '" aria-label="' + esc(id + ', caught on channel: ' + k.label) + '">' +
            '<span class="t">' + clock(c._at) + '</span><span class="c">' + esc((S.meta.channel || 'CH-01').replace('CH-', '')) + '</span>' +
            '<span class="vw">CAUGHT<span class="stamp">on air</span></span><span class="f">' + esc(m ? m[1] : diff.split(':')[0]) + '</span>' +
            '<span class="ft">' + (m ? '<span class="ok nw">' + esc(m[2]) + '\u00a0→</span> <b>' + esc(m[3]) + '</b>' : '<b>' + esc(diff || k.label) + '</b>') +
            ' · then ' + esc(VERDICT_WORD[vkey(c)] || '') + '</span></button></li>';
        }).join('');
      }).join('') + '</ol>';
  }
  function section(title, ids, why, none) {
    if (!ids.length) return '<h3>' + esc(title) + '<span class="n">0</span></h3><p class="why">' + esc(none) + '</p>';
    return '<h3>' + esc(title) + '<span class="n">' + ids.length + '</span></h3><p class="why">' + esc(why) + '</p>' +
      '<ol class="rows">' + ids.map(function (id) { return S.cards[id] ? rowHTML(S.cards[id], false) : ''; }).join('') + '</ol>';
  }

  // ------------------------------------------------------------------------------------------- the card
  function select(id, arriving, fromClick) {
    S.selected = id;
    Array.prototype.forEach.call(document.querySelectorAll('.row'), function (b) {
      if (b.getAttribute('data-id') === id) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
    });
    // entrance motion only when cards arrive slower than it plays; at 32x it would keep restarting
    var now = performance.now(), calm = now - (S.lastCardAt || 0) > 1500;
    S.lastCardAt = now;
    renderCard(S.cards[id], { enter: calm || !!fromClick });
    $('card').scrollTop = 0;
    if (fromClick && window.matchMedia('(max-width: 767px)').matches) openCard();
  }
  function openCard() {
    if (!$('card-body').querySelector('.acard')) return;
    document.body.classList.add('card-open'); $('card').focus(); requestAnimationFrame(drawLinks); }
  function closeCard() {
    document.body.classList.remove('card-open');
    var el = document.querySelector('.row[data-id="' + S.selected + '"]');
    if (el) el.focus();
  }

  function turnById(c, id) { return c.turns.filter(function (t) { return t.id === id; })[0]; }

  function diffFor(c, t, sp) {
    // the read-back span carrying a differing field (multi-command: the command whose value is the `got`)
    var v = c.verdict;
    if (!v || !v.diffs || !c.strip || t.id !== c.strip.readback) return null;
    for (var i = 0; i < v.diffs.length; i++) {
      var d = v.diffs[i];
      if (d.field !== sp.field || d.got == null) continue;
      var cmd = t.commands[sp.cmd];
      if (!cmd || String(cmd[sp.field]) === String(d.got) || t.commands.length === 1) return d;
    }
    return null;
  }

  function blockHTML(c, t, s, attrs) {
    // red covers only the words of the read-back span that the instruction's span of the same field does not have
    var text = t.text.slice(s.start, s.end);
    var ins = c.strip && turnById(c, c.strip.instruction), have = {};
    var sameCmd = ins ? ins.spans.filter(function (o) { return o.field === s.field && o.cmd === s.cmd; }) : [];
    if (ins) (sameCmd.length ? sameCmd : ins.spans).forEach(function (o) {
      if (o.field === s.field) ins.text.slice(o.start, o.end).toLowerCase().split(/[^\w-]+/).forEach(function (w) { if (w) have[w] = 1; });
    });
    var parts = text.split(/(\s+)/), shared = parts.filter(function (p) { return p.trim() && have[p.toLowerCase().replace(/[^\w-]/g, '')]; }).length;
    var words = parts.filter(function (p) { return p.trim(); }).length;
    if (!shared || shared === words) return '<mark class="blk sp"' + attrs + ' data-x="1">' + esc(text) + '</mark>';
    var out = '<span class="sp"' + attrs + ' data-x="1">';
    parts.forEach(function (p) {
      if (!p.trim()) out += esc(p);
      else if (have[p.toLowerCase().replace(/[^\w-]/g, '')]) out += esc(p);
      else out += '<mark class="blk">' + esc(p) + '</mark>';
    });
    return out + '</span>';
  }
  function insDiffers(c, field) {
    return !!(c.verdict && c.verdict.diffs && c.verdict.diffs.some(function (d) { return d.field === field; }));
  }
  function lineHTML(c, t, role) {
    var chip = false;
    var text = t.text, spans = t.spans.slice().sort(function (a, b) { return a.start - b.start; });
    var out = '', pos = 0, merged = [];
    spans.forEach(function (s) {  // shared spans (one verb for two commands) collapse to one mark
      var last = merged[merged.length - 1];
      if (last && last.start === s.start && last.end === s.end) return;
      if (last && s.start < last.end) return;
      merged.push(s);
    });
    var retracted = (t.retracted || []).map(function (r) { return { start: r[0], end: r[1], ret: true }; });
    merged = merged.concat(retracted).sort(function (a, b) { return a.start - b.start; });
    merged.forEach(function (s) {
      if (s.start < pos) return;
      out += esc(text.slice(pos, s.start));
      var w = esc(text.slice(s.start, s.end));
      if (s.ret) out += '<span class="ret" title="retracted on the channel">' + w + '</span>';
      else {
        var d = diffFor(c, t, s);
        var xi = role === 'ins' && insDiffers(c, s.field);
        var cmdS = t.commands[s.cmd];
        var amb = role === 'ins' && s.field === 'tag' && c.verdict && c.verdict.reason === 'unknown tag' && cmdS && cmdS.tag_status === 'not_in_registry';
        if (amb) chip = true;
        var attrs = ' data-f="' + esc(s.field) + '" data-c="' + esc(s.cmd) + '" data-side="' + esc(role) + '"';
        if (d) out += blockHTML(c, t, s, attrs);
        else out += '<span class="sp' + (s.field === 'state' ? ' st' : '') + (xi ? ' xi' : '') + (amb ? ' amb' : '') + '"' + attrs + ' title="' + esc(s.field) + ' span">' + w + '</span>';
      }
      pos = s.end;
    });
    out += esc(text.slice(pos));
    if (chip) out += ' <span class="nreg">' + esc(t.commands.filter(function (k) { return k.tag_status === 'not_in_registry'; }).map(function (k) { return k.tag; }).join(', ')) + ' not in registry</span>';
    if (role === 'ins' && c.verdict && c.verdict.reason === 'unknown tag') {
      t.commands.forEach(function (k) {
        if (k.tag_status === 'ok' && k.line && k.registry_line && k.line !== k.registry_line)
          out += ' <span class="nreg">' + esc(k.tag) + ' is on line ' + esc(k.registry_line) + ' in the registry, not ' + esc(k.line) + '</span>';
      });
    }
    if (role === 'rb' && c.verdict && c.verdict.diffs) {
      c.verdict.diffs.forEach(function (d) {
        if (d.got == null && d.field !== 'command') out += ' <span class="miss" data-f="' + esc(d.field) + '" data-c="0" data-side="rb" data-x="1">no ' + esc(d.field) + ' read back</span>';
      });
    }
    return out;
  }

  function laneHTML(t) {
    if (t.peaks && t.peaks.length) {
      var w = t.peaks.length, bars = t.peaks.map(function (p, i) {
        var h = Math.max(1, p * 14);
        return '<rect x="' + i + '" y="' + (8 - h / 2).toFixed(2) + '" width=".7" height="' + h.toFixed(2) + '"/>';
      }).join('');
      return '<div class="lane"><svg viewBox="0 0 ' + w + ' 16" preserveAspectRatio="none" aria-hidden="true">' + bars + '</svg></div>';
    }
    return '';  // no recording: the speaker label says so, nothing is drawn where audio would be
  }

  function stripLine(c, t, role) {
    var who = t.role === 'operator' ? 'Operator' : 'Field tech';
    var cls = 'sline' + (t.status === 'abstain' ? ' ab' : '') + (t.status === 'not_audited' ? ' na' : '');
    var noRec = !(t.peaks && t.peaks.length) ? '<small class="norec" title="This exchange has no recording: the transcript is the card script">no recording</small>' : '';
    return '<div class="' + cls + '"><span class="who">' + who + '<small>' + esc(t.id) + '</small>' + noRec + '</span><span class="w">' + lineHTML(c, t, role) + '</span>' + laneHTML(t) + '</div>';
  }

  function receiptLine(t) {
    var who = t.role === 'operator' ? 'Operator' : 'Field tech';
    var r = t.receipt, bits;
    if (t.not_audited) bits = '<b>not audited</b> · ' + esc(t.not_audited);
    else if (!r) bits = S.rung === 'regex' ? 'regex rung 0 · no model, no call' : 'no receipt';
    else {
      bits = (r.live ? '<b>live call</b>' + (r.started_at ? ' at ' + esc(String(r.started_at).slice(11, 19)) + ' UTC' : '') + ' · ' : '') + '<b>' + esc(r.model_served || S.rung) + '</b> · ' + esc(r.host || '') +
        (r.on_amd ? ' (AMD-hosted API)' : '') + ' · <b>' + esc(secs(r.latency_s) || '') + '</b>' +
        (r.cost_usd != null ? ' · ' + esc(usdFixed(r.cost_usd)) : '') + ' · request <span title="' + esc(r.request_id) + '">' + esc(String(r.request_id || '').slice(0, 12)) + '…</span>' +
        (r.run ? ' · run ' + esc(r.run) : '');
    }
    return '<li><span class="rl">' + who + '</span><span>' + bits + '</span></li>';
  }

  function renderCard(c, opts) {
    if (!c) return;
    opts = opts || {};
    var v = c.verdict, vk = vkey(c), spot = vk === 'VERIFIED' && isSpot(c.exchange_id);
    var word = vk ? VERDICT_WORD[vk] : 'NO CARD';
    var h = '<article class="acard v-' + (vk || 'none') + (opts.enter ? ' enter' : '') + '">';
    h += '<header class="ac-h"><div class="vword">' + (vk === 'VERIFIED' ? TICK : '') + esc(spot ? 'SPOT-CHECK' : word) + '</div>' +
      '<div class="ac-meta"><b>' + esc(c.exchange_id) + '</b> · ' + esc(S.meta.channel) + (c._at != null ? ' · T+' + clock(c._at) : '') +
      (LANG[c.lang] ? ' · <span class="langc">' + esc(LANG[c.lang]) + '</span>' : '') + '</div></header>';
    if (opts.timing) h += opts.timing;
    // the one-line verdict
    if (v && v.diffs && v.diffs.length) {
      h += '<div class="diffline">' + v.diffs.map(function (d) {
        return '<span class="d">' + esc(d.text) + '</span>' + (d.note && d.field !== 'command' ? '<span class="note">' + esc(d.note) + ' (registry)</span>' : '');
      }).join('') + '</div>';
    } else if (v && v.reason && !opts.timing) {
      h += '<div class="diffline"><span class="r">' + esc(v.reason) + '</span><span class="note">' + esc(reasonNote(v.reason, c)) + '</span></div>';
    } else if (vk === 'VERIFIED') {
      h += '<div class="diffline quiet"><span class="d">' + TICK + '</span><span class="note">Every field of the read-back matches the instruction, and the instruction matches the plant registry.' +
        (spot ? ' Drawn as a spot-check: a human re-checks it anyway.' : '') + '</span></div>';
    } else if (!v) {
      h += '<div class="diffline quiet"><span class="note">No instruction on the channel: no exchange was opened, so there is nothing to judge.</span></div>';
    }
    if (vk === 'ABSTAIN' && !opts.timing) h += '<p class="listen">' + (v.reason === 'audio' ? 'Cannot hear: listen yourself.' : 'Cannot clear this exchange: a human must check it.') + '</p>';
    if (vk === 'NOT_AUDITED') h += '<p class="listen na">Not audited: a model call failed, so the referee did not judge this exchange. It is listed, never passed.</p>';
    // the read-back strip
    var ins = c.strip && turnById(c, c.strip.instruction), rb = c.strip && c.strip.readback && turnById(c, c.strip.readback);
    if (ins) {
      h += '<div class="strip' + (vk === 'ABSTAIN' ? ' ab' : '') + '" id="strip">' + stripLine(c, ins, 'ins') + (rb ? '<div class="gap"></div>' + stripLine(c, rb, 'rb') : '') +
        '<svg class="links" aria-hidden="true"></svg></div>';
    }
    // the supervisor's decision comes right after the evidence, before the detail
    h += reviewHTML(c);
    // every turn on the channel for this exchange
    if (c.turns.length > 2 || !ins) {
      h += '<div class="sec"><span class="lab">On the channel</span><ol class="chan">' + c.turns.map(function (t) {
        var on = c.strip && (t.id === c.strip.instruction || t.id === c.strip.readback);
        return '<li class="' + (on ? 'on' : '') + '"><span>+' + t.at.toFixed(1) + ' s</span><span class="rl">' + (t.role === 'operator' ? 'Operator' : 'Field') + '</span><span class="tx">' + esc(t.text) + '</span></li>';
      }).join('') + '</ol></div>';
    }
    if (c.caught && c.caught.length) {
      h += '<div class="sec"><span class="lab">Caught on channel</span><ul class="trace">' + c.caught.map(function (k) {
        return '<li>' + esc(k.readback) + ': ' + esc(k.label) + ', corrected by the operator on air</li>';
      }).join('') + '</ul></div>';
    }
    // provenance
    var refWhere = S.mode === 'live' ? 'in this server' : 'at export, ' + (S.meta.referee_measured_on || '');
    h += '<div class="badges">' +
      '<span class="bdg">speaker <b>' + esc(c.provenance.speaker) + '</b></span>' +
      '<span class="bdg">radio <b>' + esc(c.provenance.radio) + '</b></span>' +
      (c.turns.some(function (t) { return t.receipt && t.receipt.on_amd; }) ? '<span class="bdg amd">extraction <b>' + esc(S.rung) + '</b> · <span class="nw">AMD-hosted API</span></span>'
        : '<span class="bdg">extraction <b>' + esc(S.rung === 'regex' ? 'regex' : S.rung) + '</b>' + (S.rung === 'regex' ? ', no model' : ', no successful call') + '</span>') +
      '<span class="bdg">referee <b>local</b> · <b>' + esc(c.referee_ms != null ? c.referee_ms.toFixed(2) + ' ms' : '') + '</b></span>' +
      '</div>';
    // receipts
    var nCalls = {};
    c.turns.forEach(function (t) { if (t.receipt && t.receipt.request_id) nCalls[t.receipt.request_id] = 1; });
    var nc = Object.keys(nCalls).length, anyLive = c.turns.some(function (t) { return t.receipt && t.receipt.live; });
    h += '<details class="sec"><summary><span class="lab">Receipts · ' + (S.rung === 'regex' ? 'no model call (regex rung 0)' : nc + ' model call' + (nc === 1 ? '' : 's') +
      ' behind these turns') + '</span></summary><ul class="rcpt">' + c.turns.map(receiptLine).join('') + '</ul>' +
      (nc && !anyLive && !c.not_audited_turns ? '<p class="evalrow">Identical wording shares one stored call: the run extracted each distinct transcript once.</p>' : '') +
      '<p class="evalrow">Referee time ' + esc(c.referee_ms != null ? c.referee_ms.toFixed(2) + ' ms' : 'n/a') + ', measured ' + esc(refWhere) + '.</p></details>';
    // the baseline
    if (c.baseline && c.baseline.verdict) {
      var b = c.baseline, exp = c.expected || '';
      var passedWrong = b.verdict === 'VERIFIED' && /^(MISMATCH|NO-READBACK)/.test(exp);
      var passedUnknown = b.verdict === 'VERIFIED' && /unknown tag/.test(exp);
      h += '<div class="sec"><span class="lab">Same exchange, LLM-as-judge baseline</span><div class="base"><span class="bv' + (passedWrong || passedUnknown ? ' bad' : '') + '">' +
        esc(VERDICT_WORD[b.verdict] || b.label || b.verdict) + '</span><span>' + esc(b.model || 'the judge model') + ', given the whole exchange and no plant registry' +
        (passedWrong ? ': <strong>it passed a read-back the card script seeded as wrong.</strong>'
          : passedUnknown ? ': <strong>it passed an order for equipment the plant registry does not have.</strong>' : '.') + '</span></div></div>';
    }
    // trace + eval label
    h += '<div class="sec"><span class="lab">Verdict trace</span><ol class="trace">' + (c.trace || []).map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ol></div>';
    h += '<div class="sec evalrow">Card script seeded as <b>' + esc(c.expected || 'no card') + '</b> (' + esc(c.error_type) + ', ' + esc(c.row) + ' row, language ' + esc(c.lang) + ')' +
      (c.matches_stored === false ? ' · <b>differs from the stored run</b>' : c.matches_stored ? ' · same label as the stored run' : '') + '</div>';
    h += '</article>';
    $('card-body').innerHTML = h;
    bindReview(c);
    requestAnimationFrame(drawLinks);
  }

  function reasonNote(r, c) {
    if (r === 'unknown tag' && c) {
      var ins = c.strip && turnById(c, c.strip.instruction), off = ins && ins.commands.filter(function (k) { return k.tag_status === 'ok' && k.line && k.registry_line && k.line !== k.registry_line; })[0];
      if (off) return off.tag + ' is on line ' + off.registry_line + ' in the plant registry, not line ' + off.line + '. The referee will not guess which the operator meant: a human checks the order itself.';
    }
    var m = {
      'unknown tag': 'The instruction names equipment the plant registry does not have, or a line it is not on. The referee will not guess: a human checks the order itself.',
      'grounding': 'The model pointed at words that are not in the transcript. The referee does not trust it: a human checks the exchange.',
      'coverage': 'A safety word in the transcript was not accounted for by any command. A human checks the exchange.',
      'retraction cue missing': 'Two values for one field and no correction word between them. The referee will not pick one for the crew.',
      'ambiguous tag': 'A bare number that more than one tag in the registry ends with.',
      'audio': 'The speech-to-text was not sure of the words that matter. Listen yourself.',
    };
    if (/^unmapped/.test(r)) return 'A word in a span maps to no action, tag or line in the alias tables.';
    return m[r] || 'A human must check this exchange.';
  }

  // ------------------------------------------------------------------------------------------- review
  function reviewHTML(c) {
    var r = S.reviews[c.exchange_id];
    if (r) {
      return '<div class="decided' + (r._fresh ? ' enter' : '') + '"><span class="w">' + esc(r.decision) + '</span><span class="m">by the supervisor · ' + esc(r.at) +
        (r.reason ? ' · reason: ' + esc(r.reason) : '') + (S.mode === 'replay' ? ' · kept in this browser only' : ' · stored by the server') + '</span></div>';
    }
    if (!c.live && !isSpot(c.exchange_id) && vkey(c) !== 'NOT_AUDITED') {
      return '<div class="review"><p class="hint">No card: this exchange was logged, not sent to the supervisor.</p></div>';
    }
    var opts = DISMISS_REASONS.map(function (t, i) {
      return '<label><input type="radio" name="why" value="' + esc(t) + '" id="why' + i + '">' + esc(t) + '</label>';
    }).join('') + '<label><input type="radio" name="why" value="other" id="why-o">Other</label><input type="text" id="why-txt" placeholder="Why is the referee wrong here?" hidden aria-label="Dismissal reason">';
    return '<div class="review"><button type="button" class="btn btn-confirm" id="rv-confirm">' + (vkey(c) === 'ABSTAIN' ? 'Confirm: needs follow-up' : 'Confirm error') + '</button>' +
      '<button type="button" class="btn" id="rv-dismiss" aria-expanded="false" aria-controls="rv-form">Dismiss…</button>' +
      '<form class="dismiss" id="rv-form" hidden><fieldset><legend class="lab">Why dismiss? A reason is required</legend>' + opts +
      '</fieldset><div class="acts"><button type="submit" class="btn" id="rv-send" disabled>Dismiss with this reason</button></div></form>' +
      '<p class="hint">The card leaves the queue only through one of these two buttons. Nothing is sent to the plant.</p></div>';
  }
  function bindReview(c) {
    var conf = $('rv-confirm');
    if (!conf) return;
    conf.addEventListener('click', function () { review(c, 'confirmed', null); });
    var form = $('rv-form'), dis = $('rv-dismiss'), send = $('rv-send'), txt = $('why-txt');
    dis.addEventListener('click', function () {
      form.hidden = !form.hidden;
      dis.setAttribute('aria-expanded', String(!form.hidden));
      if (!form.hidden) form.querySelector('input').focus();
    });
    function reason() {
      var r = form.querySelector('input[name=why]:checked');
      if (!r) return '';
      return r.value === 'other' ? txt.value.trim() : r.value;
    }
    form.addEventListener('change', function () {
      var r = form.querySelector('input[name=why]:checked');
      txt.hidden = !(r && r.value === 'other');
      send.disabled = !reason();
    });
    txt.addEventListener('input', function () { send.disabled = !reason(); });
    form.addEventListener('submit', function (e) { e.preventDefault(); if (reason()) review(c, 'dismissed', reason()); });
  }
  function review(c, decision, reason) {
    var body = { run: S.run, rung: S.rung, exchange_id: c.exchange_id, decision: decision, reason: reason };
    var p = S.mode === 'live'
      ? fetch('api/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(function (r) { if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || r.status); }); return r.json(); })
      : Promise.resolve(Object.assign({}, body, { at: new Date().toISOString().slice(0, 19) + 'Z' }));
    p.then(function (row) {
      row._fresh = true;
      S.reviews[c.exchange_id] = row;
      if (S.mode === 'replay') {
        var all = Object.keys(S.reviews).map(function (k) { var x = Object.assign({}, S.reviews[k]); delete x._fresh; return x; });
        try { localStorage.setItem(reviewKey(), JSON.stringify(all)); } catch (e) { /* private mode: keep in memory */ }
      }
      renderCard(c, {});
      delete row._fresh;
      if (S.tab === 'eos') renderEos(); else renderList();
      renderSub();
    }).catch(function (e) {
      var hint = document.querySelector('.review .hint');
      if (hint) hint.textContent = 'Not saved: ' + (e.message || e);
    });
  }

  // ------------------------------------------------------------------------------------------- the refused one
  function tryRefused() {
    var rf = S.meta && S.meta.refused;
    if (!rf) return;
    if (S.mode === 'replay' && !S.data) return;
    setFollow(false);
    var gen = S.gen, btn = $('refused');
    btn.disabled = true;
    var p;
    if (S.mode === 'live') {
      p = fetch('api/try-refused', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run: S.run, rung: S.rung, extract: S.extract }) })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); });
    } else {
      var ev = S.data.events.filter(function (e) { return e.type === 'judged' && e.card.exchange_id === rf.exchange_id; })[0];
      p = ev ? Promise.resolve({ card: JSON.parse(JSON.stringify(ev.card)), extract: 'stored', measured_on: S.meta.referee_measured_on, replay: true })
        : Promise.reject(new Error('the refused exchange is not in this run'));
    }
    p.then(function (j) {
      if (gen !== S.gen) return;
      var c = j.card;
      c._at = null;
      S.tried = c;
      Array.prototype.forEach.call(document.querySelectorAll('.row[aria-current]'), function (b) { b.removeAttribute('aria-current'); });
      S.selected = null;
      var turns = c.turns.slice(), k = 0;
      var wait = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 700;
      (function next() {
        if (gen !== S.gen) return;
        if (k < turns.length) {  // play the turns on the channel strip, then show the card
          var t = turns[k++];
          $('ch-who').textContent = t.role === 'operator' ? 'Operator' : 'Field tech';
          var tx = $('ch-txt'); tx.textContent = t.text; tx.classList.remove('in'); void tx.offsetWidth; tx.classList.add('in');
          $('ch-line').classList.add('tx');
          S.timers.push(setTimeout(next, wait));
          return;
        }
        $('ch-line').classList.remove('tx');
        var ext = c.turns.map(function (t) { return t.receipt && typeof t.receipt.latency_s === 'number' ? t.receipt.latency_s.toFixed(2) + ' s' : null; }).filter(Boolean);
        var live = c.turns.some(function (t) { return t.receipt && t.receipt.live; });
        var ex = S.meta.stages.extract;
        var exTxt = S.rung === 'regex' ? 'extraction: regex rung 0, no model call'
          : (live ? 'extraction called live on the ' : 'extraction replayed from its receipts, ') + esc(ex.where) + ': <b>' + esc(ext.join(' + ')) + '</b>' +
            (ex.latency_s_p50 != null ? ' (p50 ' + esc(ex.latency_s_p50.toFixed(2)) + ' s over the ' + esc(ex.calls) + ' calls behind this run' + (ex.calls_reused ? ', ' + esc(ex.calls_reused) + ' of them reused from the first pass' : '') + ')' : '');
        var refTxt = 'referee <b>' + c.referee_ms.toFixed(2) + ' ms</b> ' + (j.replay ? 'when this replay was exported' : 'just now in the server') + ' (' + esc(j.measured_on || '') + ')' +
          (live ? ' · each live receipt below carries the time of its call (a turn already called this session is reused)' : '');
        var why = c.verdict && c.verdict.reason === 'unknown tag'
          ? esc(rf.tag) + ' is not in the plant registry, so the referee refuses to clear the order: <b>' + esc(c.verdict.label) + '</b>.'
          : 'This rung’s output made the referee answer <b>' + esc(c.verdict ? c.verdict.label : 'no verdict') + '</b> before the registry check.';
        var timing = '<div class="timing"><p><b>You pressed \u201cTry the refused one\u201d.</b> ' + why + '</p>' +
          '<details><summary>How this was judged</summary><p>' + exTxt + ' · ' + refTxt + '</p></details></div>';
        renderCard(c, { enter: true, timing: timing });
        $('card').scrollTop = 0;
        if (matchMedia('(max-width: 767px)').matches) openCard();
        btn.disabled = false;
      })();
    }).catch(function (e) {
      btn.disabled = false;
      notice('Could not judge the refused exchange: ' + (e.message || e));
    });
  }

  // ------------------------------------------------------------------------------------------- connectors
  function drawLinks() {
    var strip = document.getElementById('strip');
    if (!strip) return;
    var svg = strip.querySelector('svg.links');
    var box = strip.getBoundingClientRect();
    var ins = strip.querySelectorAll('[data-side="ins"]'), rb = strip.querySelectorAll('[data-side="rb"]');
    var out = '';
    Array.prototype.forEach.call(rb, function (r) {
      var f = r.getAttribute('data-f'), cmd = r.getAttribute('data-c');
      var same = Array.prototype.filter.call(ins, function (i) { return i.getAttribute('data-f') === f; });
      var match = same.filter(function (i) { return i.getAttribute('data-c') === cmd; })[0] || same[0];
      if (!match) return;
      var a = match.getBoundingClientRect(), b = r.getBoundingClientRect();
      if (!a.width || !b.width) return;
      var x1 = a.left + a.width / 2 - box.left, y1 = a.bottom - box.top + 2;
      var x2 = b.left + b.width / 2 - box.left, y2 = b.top - box.top - 2;
      var bad = r.getAttribute('data-x') === '1', cls = bad ? ' class="x"' : '';
      out += '<line' + cls + ' x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '"/>' +
        '<circle' + cls + ' cx="' + x1.toFixed(1) + '" cy="' + y1.toFixed(1) + '" r="2.2"/><circle' + cls + ' cx="' + x2.toFixed(1) + '" cy="' + y2.toFixed(1) + '" r="2.2"/>';
    });
    svg.innerHTML = out;
  }

  // ------------------------------------------------------------------------------------------- chime
  function chime(force) {
    if (!S.chime && !force) return;
    try {
      S.audio = S.audio || new (window.AudioContext || window.webkitAudioContext)();
      var ctx = S.audio, t = ctx.currentTime;
      [660, 880].forEach(function (f, i) {
        var o = ctx.createOscillator(), g = ctx.createGain();
        o.type = 'sine'; o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, t + i * 0.12);
        g.gain.exponentialRampToValueAtTime(0.06, t + i * 0.12 + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.12 + 0.35);
        o.connect(g); g.connect(ctx.destination);
        o.start(t + i * 0.12); o.stop(t + i * 0.12 + 0.4);
      });
    } catch (e) { /* no audio device: silent */ }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
