/* ATCO site - the signal-processing demonstration.

   An educational browser model of what a radio channel does to a signal. It is NOT
   a live test of the recognition models, and not a calibrated recreation of real
   air traffic radio. It approximates
   experiments/asr/04_augmentation_and_synthetic_speech/simulate_radio_channel.py;
   every place the approximation differs is described on the page.

   No corpus audio is used, loaded, recorded, stored or transmitted. */
(function () {
  'use strict';

  var demo = document.getElementById('demo');
  if (!demo) return;
  var AC = window.AudioContext || window.webkitAudioContext;

  var playBtn   = document.getElementById('play');
  var squelchBt = document.getElementById('squelch');
  var hint      = document.getElementById('hint');
  var scopefall = document.getElementById('scopefall');
  var canvas    = document.getElementById('canvas');
  var micBtn    = document.getElementById('micbtn');
  var micStopBt = document.getElementById('micstop');
  var micState  = document.getElementById('micstate');
  var presetState = document.getElementById('presetstate');
  var srcState  = document.getElementById('srcstate');
  var phraseEl  = document.getElementById('phrase');
  var monWrap   = document.getElementById('monwrap');
  var monitor   = document.getElementById('monitor');
  var srcRadios = Array.prototype.slice.call(demo.querySelectorAll('input[name="src"]'));
  var preRadios = Array.prototype.slice.call(demo.querySelectorAll('input[name="preset"]'));
  var sw = {
    band:  document.getElementById('s-band'),
    brown: document.getElementById('s-brown'),
    white: document.getElementById('s-white'),
    mulaw: document.getElementById('s-mulaw'),
    bp:    document.getElementById('s-bp'),
    clip:  document.getElementById('s-clip'),
    sq:    document.getElementById('s-sq')
  };

  function disableAll(msg) {
    [playBtn, squelchBt, micBtn, micStopBt].forEach(function (b) { if (b) b.disabled = true; });
    srcRadios.concat(preRadios).forEach(function (r) { r.disabled = true; });
    if (hint) hint.textContent = msg;
  }
  if (!AC) {
    disableAll('This browser does not support the Web Audio API, so the demonstration cannot run. ' +
               'The seven stages below describe the simulated channel in full.');
    return;
  }

  var reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- constants ---------- */
  var NYQ_OFF = 21000;      // nominal "filter bypassed" corner; clamped to Nyquist at use
  var CURVE_N = 16384;
  var VOWELS = [            // F1, F2, F3 in Hz
    [730, 1090, 2440], [270, 2290, 3010], [300, 870, 2240], [530, 1840, 2480],
    [400, 1990, 2550], [570, 840, 2410], [490, 1350, 1690]
  ];


  /* ---------- helpers ---------- */
  function gauss() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function identityCurve() {
    var c = new Float32Array(CURVE_N);
    for (var i = 0; i < CURVE_N; i++) c[i] = (i / (CURVE_N - 1)) * 2 - 1;
    return c;
  }

  /* mu-law: compress, quantise, expand. The grit comes from quantisation error,
     which is proportional to the signal. NOTE: round(y*128)/128 snaps to steps of
     1/128 across a signed domain, so it yields 257 possible output values, not 128
     levels. The page says 257; this comment used to say 128. The historical Python
     experiment is left exactly as it was. */
  function muLawCurve() {
    var c = new Float32Array(CURVE_N), mu = 255, i, x, y;
    for (i = 0; i < CURVE_N; i++) {
      x = (i / (CURVE_N - 1)) * 2 - 1;
      y = Math.sign(x) * Math.log(1 + mu * Math.abs(x)) / Math.log(1 + mu);
      y = Math.round(y * 128) / 128;
      c[i] = Math.sign(y) * (1 / mu) * (Math.pow(1 + mu, Math.abs(y)) - 1);
    }
    return c;
  }

  function clipCurve() {
    var c = new Float32Array(CURVE_N), i, x;
    for (i = 0; i < CURVE_N; i++) {
      x = (i / (CURVE_N - 1)) * 2 - 1;
      c[i] = Math.max(-0.95, Math.min(0.95, x));
    }
    return c;
  }

  /* brown noise = running sum of white noise. Integration divides each component's
     amplitude by its frequency, so power falls as 1/f^2. Detrended so the loop
     point does not click. */
  function brownBuffer(ac) {
    var n = ac.sampleRate * 4, b = ac.createBuffer(1, n, ac.sampleRate);
    var d = b.getChannelData(0), run = 0, i, peak = 0;
    for (i = 0; i < n; i++) { run += gauss(); d[i] = run; }
    var drift = d[n - 1] / (n - 1);
    for (i = 0; i < n; i++) { d[i] -= drift * i; if (Math.abs(d[i]) > peak) peak = Math.abs(d[i]); }
    if (peak > 0) for (i = 0; i < n; i++) d[i] /= peak;
    return b;
  }

  function whiteBuffer(ac) {
    var n = ac.sampleRate * 2, b = ac.createBuffer(1, n, ac.sampleRate);
    var d = b.getChannelData(0);
    for (var i = 0; i < n; i++) d[i] = gauss() * 0.3;
    return b;
  }

  function lowpass(ac, f) { var n = ac.createBiquadFilter(); n.type = 'lowpass';  n.frequency.value = f; n.Q.value = 0.7071; return n; }
  function highpass(ac, f){ var n = ac.createBiquadFilter(); n.type = 'highpass'; n.frequency.value = f; n.Q.value = 0.7071; return n; }

  /* ---------- the graph ---------- */
  function build() {
    var ac = ctx, g = {};

    g.srcSum  = ac.createGain();
    g.voice   = ac.createGain(); g.voice.gain.value = 0;
    g.sweep   = ac.createGain(); g.sweep.gain.value = 0;
    g.mic     = ac.createGain(); g.mic.gain.value = 0;
    g.phrase  = ac.createGain(); g.phrase.gain.value = 0;
    g.voice.connect(g.srcSum); g.sweep.connect(g.srcSum);
    g.mic.connect(g.srcSum);   g.phrase.connect(g.srcSum);

    /* the spoken call: one sentence from the project's own generator, read by a
       text-to-speech voice. createMediaElementSource may only be called once per
       element, so it is created here with the graph and kept. */
    if (phraseEl) {
      try {
        g.phraseSrc = ac.createMediaElementSource(phraseEl);
        g.phraseSrc.connect(g.phrase);
      } catch (e) { g.phraseSrc = null; }
    }

    /* fake voice: a buzz through three resonances, the way a vowel is made */
    g.osc = ac.createOscillator(); g.osc.type = 'sawtooth'; g.osc.frequency.value = 112;
    g.env = ac.createGain(); g.env.gain.value = 0.0001;
    g.osc.connect(g.env);
    g.fmt = [];
    [[1.0, 9], [0.55, 11], [0.3, 13]].forEach(function (spec, i) {
      var f = ac.createBiquadFilter();
      f.type = 'bandpass'; f.frequency.value = VOWELS[0][i]; f.Q.value = spec[1];
      var a = ac.createGain(); a.gain.value = spec[0];
      g.env.connect(f); f.connect(a); a.connect(g.voice);
      g.fmt.push(f);
    });

    /* sweep: 80 Hz to 7.5 kHz, so the filters are impossible to miss */
    g.sosc = ac.createOscillator(); g.sosc.type = 'sine'; g.sosc.frequency.value = 80;
    g.sgain = ac.createGain(); g.sgain.gain.value = 0.42;
    g.sosc.connect(g.sgain); g.sgain.connect(g.sweep);

    /* 1. band-limiting — two cascaded low-pass sections stand in for the
          16 kHz -> 8 kHz -> 16 kHz resample (nothing survives above 4 kHz) */
    g.lpA = lowpass(ac, 4000); g.lpB = lowpass(ac, 4000);
    g.srcSum.connect(g.lpA); g.lpA.connect(g.lpB);

    /* 2 + 3. engine rumble and static, summed in */
    g.noiseMix = ac.createGain(); g.lpB.connect(g.noiseMix);
    g.brownSrc = ac.createBufferSource(); g.brownSrc.buffer = brownBuffer(ac); g.brownSrc.loop = true;
    g.brownGain = ac.createGain(); g.brownGain.gain.value = 0;
    g.brownSrc.connect(g.brownGain); g.brownGain.connect(g.noiseMix);
    g.whiteSrc = ac.createBufferSource(); g.whiteSrc.buffer = whiteBuffer(ac); g.whiteSrc.loop = true;
    g.whiteGain = ac.createGain(); g.whiteGain.gain.value = 0;
    g.whiteSrc.connect(g.whiteGain); g.whiteGain.connect(g.noiseMix);

    /* 4. mu-law companding */
    g.mu = ac.createWaveShaper(); g.mu.oversample = 'none';
    g.muCurve = muLawCurve(); g.idCurve = identityCurve();
    g.mu.curve = g.muCurve;
    g.noiseMix.connect(g.mu);

    /* 5. 300-3400 Hz Butterworth band-pass, 4th order each side */
    g.hpA = highpass(ac, 300); g.hpB = highpass(ac, 300);
    g.lpC = lowpass(ac, 3400); g.lpD = lowpass(ac, 3400);
    g.mu.connect(g.hpA); g.hpA.connect(g.hpB); g.hpB.connect(g.lpC); g.lpC.connect(g.lpD);

    /* 6. overdrive: gain into a hard clip at +/-0.95 */
    g.drive = ac.createGain(); g.drive.gain.value = 1;
    g.clip = ac.createWaveShaper(); g.clip.oversample = '2x'; g.clipCurve = clipCurve();
    g.clip.curve = g.clipCurve;
    g.lpD.connect(g.drive); g.drive.connect(g.clip);

    /* 7. squelch bursts land here, after everything else */
    g.wetSum = ac.createGain(); g.clip.connect(g.wetSum);

    g.dry = ac.createAnalyser(); g.wet = ac.createAnalyser();
    [g.dry, g.wet].forEach(function (a) {
      a.fftSize = 2048; a.smoothingTimeConstant = 0.45;
      a.minDecibels = -92; a.maxDecibels = -20;
    });
    g.srcSum.connect(g.dry);
    g.wetSum.connect(g.wet);

    /* a gentle limiter so the clipping stage cannot produce a harsh output level */
    g.limiter = ac.createDynamicsCompressor();
    g.limiter.threshold.value = -12;
    g.limiter.knee.value = 12;
    g.limiter.ratio.value = 8;
    g.limiter.attack.value = 0.004;
    g.limiter.release.value = 0.12;

    g.out = ac.createGain(); g.out.gain.value = 0;
    g.wetSum.connect(g.limiter); g.limiter.connect(g.out); g.out.connect(ac.destination);

    g.osc.start(); g.sosc.start(); g.brownSrc.start(); g.whiteSrc.start();
    return g;
  }

  var ctx = null, graph = null;
  var dryData = null, wetData = null;


  /* ---------- explicit lifecycle: idle | starting | running | stopped ---------- */
  var phase = 'idle';
  var rafId = null;          /* exactly one animation-frame loop, ever */
  /* read the initial source from the markup rather than hard-coding it, so the
     two can never disagree */
  var srcKind = (function () {
    var r = document.querySelector('input[name="src"]:checked');
    return r ? r.value : 'voice';
  })();
  var DEFAULT_SRC = srcKind;          /* where leaving the microphone returns to */
  var startToken = 0;        /* guards a resume() that settles after Stop */

  function setPhase(next) {
    phase = next;
    var running = (phase === 'running');
    playBtn.textContent = (phase === 'starting') ? 'Starting...' : (running ? 'Stop' : 'Start');
    playBtn.disabled = (phase === 'starting');
    if (squelchBt) squelchBt.disabled = !running || !sw.sq.checked;
    if (scopefall) scopefall.hidden = running;
  }

  /* Rendering and audio scheduling are deliberately separate. Reduced motion
     suppresses the scrolling animation; it must never starve the source. */
  var schedTimer = null;

  function stopLoop() {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  }
  function startLoop() {
    stopLoop();
    if (reducedMotion) { drawOnce(); return; }
    rafId = requestAnimationFrame(frame);
  }

  function stopScheduler() {
    if (schedTimer !== null) { clearInterval(schedTimer); schedTimer = null; }
  }
  function startScheduler() {
    stopScheduler();
    pumpScheduler();
    schedTimer = setInterval(pumpScheduler, 250);   /* look-ahead, independent of rAF */
  }
  function pumpScheduler() {
    if (phase !== 'running' || !ctx) return;
    if (srcKind === 'voice' && ctx.currentTime > nextVoiceT - 0.6) {
      scheduleVoice(Math.max(nextVoiceT, ctx.currentTime + 0.05));
    }
    if (srcKind === 'sweep' && ctx.currentTime > nextSweepT - 0.5) {
      scheduleSweep(Math.max(nextSweepT, ctx.currentTime + 0.05));
    }
  }

  /* ---------- stage switches ---------- */
  function applyStages() {
    if (!graph) return;
    var t = ctx.currentTime, r = 0.02;
    /* a biquad corner above Nyquist is undefined, so clamp it */
    var OFF = Math.min(NYQ_OFF, ctx.sampleRate / 2 - 100);
    graph.lpA.frequency.setTargetAtTime(sw.band.checked ? 4000 : OFF, t, r);
    graph.lpB.frequency.setTargetAtTime(sw.band.checked ? 4000 : OFF, t, r);
    graph.brownGain.gain.setTargetAtTime(sw.brown.checked ? 0.022 : 0, t, r);
    graph.whiteGain.gain.setTargetAtTime(sw.white.checked ? 0.008 : 0, t, r);
    graph.mu.curve = sw.mulaw.checked ? graph.muCurve : graph.idCurve;
    graph.hpA.frequency.setTargetAtTime(sw.bp.checked ? 300 : 12, t, r);
    graph.hpB.frequency.setTargetAtTime(sw.bp.checked ? 300 : 12, t, r);
    graph.lpC.frequency.setTargetAtTime(sw.bp.checked ? 3400 : OFF, t, r);
    graph.lpD.frequency.setTargetAtTime(sw.bp.checked ? 3400 : OFF, t, r);
    graph.drive.gain.setTargetAtTime(sw.clip.checked ? 1.6 : 1, t, r);
    graph.clip.curve = sw.clip.checked ? graph.clipCurve : graph.idCurve;
    if (squelchBt) squelchBt.disabled = (phase !== 'running') || !sw.sq.checked;
  }
  Object.keys(sw).forEach(function (k) {
    sw[k].addEventListener('change', function () {
      syncPresetFromStages();      /* label first: correct even before Start */
      applyStages();               /* no-op until the graph exists */
    });
  });

  var STAGE_KEYS = ['band', 'brown', 'white', 'mulaw', 'bp', 'clip', 'sq'];
  function applyPreset(name) {
    var on = (name === 'radio');
    STAGE_KEYS.forEach(function (k) { sw[k].checked = on; });
    syncPresetFromStages();   /* the label must follow the preset, not lag it */
    applyStages();
  }
  /* Runs whether or not an audio graph exists, so the label is never stale. */
  function syncPresetFromStages() {
    var all = STAGE_KEYS.every(function (k) { return sw[k].checked; });
    var none = STAGE_KEYS.every(function (k) { return !sw[k].checked; });
    preRadios.forEach(function (r) {
      r.checked = (r.value === 'radio' && all) || (r.value === 'clean' && none);
    });
    if (presetState) {
      presetState.textContent = all ? 'Radio' : (none ? 'Clean' : 'Custom');
      presetState.dataset.state = all ? 'radio' : (none ? 'clean' : 'custom');
    }
  }
  preRadios.forEach(function (r) {
    r.addEventListener('change', function () { if (r.checked) applyPreset(r.value); });
  });

  function fireSquelch() {
    if (phase !== 'running' || !sw.sq.checked) return;
    var n = Math.floor(ctx.sampleRate * 0.15);
    var b = ctx.createBuffer(1, n, ctx.sampleRate), d = b.getChannelData(0), i, e;
    for (i = 0; i < n; i++) { e = 1 - i / n; d[i] = gauss() * 0.5 * e * e; }
    var s = ctx.createBufferSource(); s.buffer = b;
    var gg = ctx.createGain(); gg.gain.value = 0.1 + Math.random() * 0.2;
    s.connect(gg); gg.connect(graph.wetSum); s.start();
  }
  if (squelchBt) squelchBt.addEventListener('click', fireSquelch);

  /* ---------- source scheduling ---------- */
  function scheduleVoice(from) {
    var t = from, end = from + 2.5, g = graph;
    while (t < end) {
      var syllables = 1 + Math.floor(Math.random() * 4);
      for (var i = 0; i < syllables; i++) {
        var dur = 0.085 + Math.random() * 0.13;
        var v = VOWELS[(Math.random() * VOWELS.length) | 0];
        for (var k = 0; k < 3; k++) g.fmt[k].frequency.setTargetAtTime(v[k], t, 0.018);
        var pitch = 104 + Math.random() * 26;
        g.osc.frequency.setValueAtTime(pitch, t);
        g.osc.frequency.linearRampToValueAtTime(pitch * (0.93 + Math.random() * 0.12), t + dur);
        g.env.gain.setValueAtTime(0.0001, t);
        g.env.gain.exponentialRampToValueAtTime(0.85, t + 0.022);
        g.env.gain.setValueAtTime(0.85, t + Math.max(0.03, dur - 0.03));
        g.env.gain.exponentialRampToValueAtTime(0.0001, t + dur);
        t += dur + 0.012 + Math.random() * 0.03;
      }
      t += 0.16 + Math.random() * 0.26;
    }
    nextVoiceT = t;
  }
  function scheduleSweep(from) {
    graph.sosc.frequency.cancelScheduledValues(from);
    graph.sosc.frequency.setValueAtTime(80, from);
    graph.sosc.frequency.exponentialRampToValueAtTime(7500, from + 5.5);
    nextSweepT = from + 5.8;
  }
  var nextVoiceT = 0, nextSweepT = 0;

  function gateSources() {
    if (!graph) return;
    var t = ctx.currentTime;
    graph.voice.gain.setTargetAtTime(srcKind === 'voice' ? 0.8 : 0, t, 0.02);
    graph.sweep.gain.setTargetAtTime(srcKind === 'sweep' ? 0.85 : 0, t, 0.02);
    graph.phrase.gain.setTargetAtTime(srcKind === 'phrase' ? 1 : 0, t, 0.02);
    graph.mic.gain.setTargetAtTime(micNode ? 1.4 : 0, t, 0.02);
    syncPhrasePlayback();
    var audible = (srcKind !== 'mic') || (monitor && monitor.checked);
    graph.out.gain.setTargetAtTime(audible ? 0.5 : 0, t, 0.03);
    if (srcKind === 'voice' && nextVoiceT < ctx.currentTime) scheduleVoice(ctx.currentTime + 0.06);
    if (srcKind === 'sweep' && nextSweepT < ctx.currentTime) scheduleSweep(ctx.currentTime + 0.06);
  }
  /* the clip plays only while it is the selected source and the demo is running.
     Never on load, never after Stop. */
  function wantsPhrase() { return phase === 'running' && srcKind === 'phrase'; }

  function syncPhrasePlayback() {
    if (!phraseEl) return;
    if (wantsPhrase()) {
      if (phraseEl.paused) {
        var p = phraseEl.play();
        /* a play() still pending when Stop arrives would otherwise resume the clip
           after the pause, so re-check the intent once it settles */
        if (p && p.then) p.then(function () { if (!wantsPhrase()) haltPhrase(); },
                                function () { /* blocked or interrupted: nothing to undo */ });
      }
    } else {
      haltPhrase();
    }
  }

  function haltPhrase() {
    if (!phraseEl) return;
    if (!phraseEl.paused) phraseEl.pause();
    try { phraseEl.currentTime = 0; } catch (e) {}   /* Clean and Radio start level */
  }

  if (monitor) monitor.addEventListener('change', gateSources);

  srcRadios.forEach(function (r) {
    r.addEventListener('change', function () {
      if (!r.checked) return;
      releaseMic('switched');            /* leaving microphone mode must stop capture */
      srcKind = r.value;
      if (phase === 'running') gateSources();
    });
  });

  /* ---------- microphone: one request at a time, always released ---------- */
  var micStream = null, micNode = null, micGen = 0, micRequestPending = false;

  function setMicState(text, sticky) {
    if (!micState) return;
    micState.textContent = text;
    micState.dataset.sticky = sticky ? '1' : '';
  }
  function clearMicState() {
    if (micState && !micState.dataset.sticky) micState.textContent = '';
  }

  function releaseMic(reason) {
    micGen++;                                   /* invalidates any in-flight request */
    micRequestPending = false;
    if (micBtn) micBtn.disabled = false;
    if (micStream) {
      micStream.getTracks().forEach(function (tr) { tr.stop(); });
      micStream = null;
    }
    if (micNode) { try { micNode.disconnect(); } catch (e) {} micNode = null; }
    if (monitor) monitor.checked = false;
    if (monWrap) monWrap.hidden = true;
    if (micStopBt) micStopBt.hidden = true;
    if (micBtn) micBtn.hidden = false;
    if (srcKind === 'mic') {
      srcKind = DEFAULT_SRC;
      srcRadios.forEach(function (r) { r.checked = (r.value === DEFAULT_SRC); });
    }
    if (srcState) { srcState.hidden = true; srcState.textContent = ''; }
    if (reason === 'stopped' || reason === 'switched') clearMicState();
    if (graph && ctx) gateSources();
  }

  function requestMic() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMicState('This browser does not offer microphone access.', true);
      return;
    }
    if (micStream || micNode || micRequestPending) return;   /* exactly one in flight */
    micRequestPending = true;
    if (micBtn) micBtn.disabled = true;
    var gen = ++micGen;
    setMicState('Asking for permission...', false);
    navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    }).then(function (stream) {
      /* the request may settle after Stop or after the source changed */
      micRequestPending = false;
      if (micBtn) micBtn.disabled = false;
      /* stale: Stop or a source change happened while this was pending */
      if (gen !== micGen || phase !== 'running') {
        stream.getTracks().forEach(function (tr) { tr.stop(); });
        return;
      }
      micStream = stream;
      micNode = ctx.createMediaStreamSource(stream);
      micNode.connect(graph.mic);
      srcKind = 'mic';
      /* clear the competing selection: the interface must never show a synthetic
         source selected while the microphone is the live input */
      srcRadios.forEach(function (r) { r.checked = false; });
      if (srcState) { srcState.hidden = false; srcState.textContent = 'Source: microphone'; }
      if (monitor) monitor.checked = false;         /* monitoring off on every entry */
      if (monWrap) monWrap.hidden = false;
      if (micBtn) micBtn.hidden = true;
      if (micStopBt) micStopBt.hidden = false;
      setMicState('Microphone active locally. Nothing is recorded or sent.', false);
      gateSources();
    }).catch(function (err) {
      micRequestPending = false;
      if (micBtn) micBtn.disabled = false;
      if (gen !== micGen) return;
      var denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
      /* sticky: a later source change must not overwrite this */
      setMicState(denied
        ? 'Microphone permission was refused, so nothing changed: the demo is still playing the source you had selected. Nothing was recorded.'
        : 'No microphone was available, so nothing changed: the demo is still playing the source you had selected. Nothing was recorded.',
        true);
      releaseMic('failed');
    });
  }

  if (micBtn) micBtn.addEventListener('click', function () {
    if (phase !== 'running') { setMicState('Press Start first.', false); return; }
    requestMic();
  });
  if (micStopBt) micStopBt.addEventListener('click', function () { releaseMic('stopped'); });
  window.addEventListener('pagehide', function () { cleanup('stopped'); });
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) return;                    /* returning never auto-starts */
    var hadMic = !!micStream;
    if (phase === 'running') cleanup('stopped');
    if (hadMic) setMicState('Microphone released when the tab was hidden.', true);
  });

  /* ---------- the two spectrograms ---------- */
  var W = canvas.width, H = canvas.height;
  var GUT = 74, COL = 2, PANEL = 166, TOP_Y = 4, BOT_Y = 190;
  var c2d = canvas.getContext('2d');
  var buf = document.createElement('canvas'); buf.width = W; buf.height = H;
  var b2d = buf.getContext('2d');
  var ramp = null, rowBin = null;
  var col = 0;

  function buildRamp() {
    // light paper -> deep slate blue, in four stops
    var stops = [[0, 244, 246, 247], [0.35, 150, 175, 190], [0.72, 56, 95, 118], [1, 21, 40, 54]];
    ramp = new Uint8Array(256 * 3);
    for (var i = 0; i < 256; i++) {
      var v = i / 255, a = stops[0], b = stops[stops.length - 1], j;
      for (j = 0; j < stops.length - 1; j++) {
        if (v >= stops[j][0] && v <= stops[j + 1][0]) { a = stops[j]; b = stops[j + 1]; break; }
      }
      var f = (v - a[0]) / Math.max(1e-6, b[0] - a[0]);
      ramp[i * 3]     = a[1] + (b[1] - a[1]) * f;
      ramp[i * 3 + 1] = a[2] + (b[2] - a[2]) * f;
      ramp[i * 3 + 2] = a[3] + (b[3] - a[3]) * f;
    }
  }

  function rowForFreq(f) {
    return Math.round((PANEL - 1) * (1 - Math.log(f / 80) / Math.log(8000 / 80)));
  }

  function buildRowMap(fftSize, sampleRate) {
    rowBin = new Int32Array(PANEL);
    var bins = fftSize / 2;
    for (var i = 0; i < PANEL; i++) {
      var t = 1 - i / (PANEL - 1);
      var f = 80 * Math.pow(8000 / 80, t);
      rowBin[i] = Math.max(0, Math.min(bins - 1, Math.round(f * fftSize / sampleRate)));
    }
  }

  function paintColumn(data, y0, dashOn) {
    var img = c2d.createImageData(COL, PANEL), p = img.data, i, x, o, v, r, g, b;
    var gl = [rowForFreq(3400), rowForFreq(300)];
    for (i = 0; i < PANEL; i++) {
      v = data[rowBin[i]];
      r = ramp[v * 3]; g = ramp[v * 3 + 1]; b = ramp[v * 3 + 2];
      if (dashOn && (i === gl[0] || i === gl[1])) { r = 180; g = 95; b = 60; }
      for (x = 0; x < COL; x++) {
        o = (i * COL + x) * 4;
        p[o] = r; p[o + 1] = g; p[o + 2] = b; p[o + 3] = 255;
      }
    }
    c2d.putImageData(img, W - COL, y0);
  }

  function drawGutter() {
    c2d.fillStyle = '#F4F6F7';
    c2d.fillRect(0, 0, GUT, H);
    [[TOP_Y, 'Input'], [BOT_Y, 'Processed']].forEach(function (panel) {
      var y0 = panel[0];
      c2d.textAlign = 'left';
      c2d.fillStyle = '#475866';
      c2d.font = '600 10.5px "Source Sans 3", sans-serif';
      c2d.fillText(panel[1], 6, y0 + 12);
      c2d.textAlign = 'right';
      c2d.font = '500 9.5px "IBM Plex Mono", monospace';
      [[8000, '8k', 10], [3400, '3.4k', null], [300, '300', null], [80, '80', 163]].forEach(function (f) {
        var y = f[2] === null ? rowForFreq(f[0]) + 3 : f[2];
        c2d.fillStyle = (f[0] === 3400 || f[0] === 300) ? '#8F452C' : '#7A8790';
        c2d.fillText(f[1], GUT - 7, y0 + y);
      });
    });
    c2d.textAlign = 'left';
  }


  function frame() {
    if (phase !== 'running') { rafId = null; return; }
    rafId = requestAnimationFrame(frame);
    renderColumn();
  }

  var staticTimer = null;
  function drawOnce() {
    if (phase !== 'running') return;
    renderColumn();
    /* a slow, non-animated refresh so the view is not frozen at one instant */
    if (staticTimer === null) {
      staticTimer = setInterval(function () {
        if (phase !== 'running') { clearInterval(staticTimer); staticTimer = null; return; }
        renderColumn();
      }, 1000);
    }
  }

  function renderColumn() {
    graph.dry.getByteFrequencyData(dryData);
    graph.wet.getByteFrequencyData(wetData);
    b2d.clearRect(0, 0, W, H);
    b2d.drawImage(canvas, 0, 0);
    c2d.clearRect(0, 0, W, H);
    c2d.fillStyle = '#F4F6F7'; c2d.fillRect(0, 0, W, H);
    c2d.drawImage(buf, GUT, 0, W - GUT, H, GUT - COL, 0, W - GUT, H);
    col++;
    var dash = (col % 8) < 5;
    paintColumn(dryData, TOP_Y, dash);
    paintColumn(wetData, BOT_Y, dash);
    drawGutter();
  }

  /* ---------- start / stop ---------- */
  function start() {
    if (phase === 'starting' || phase === 'running') return;   /* idempotent */
    setPhase('starting');
    var token = ++startToken;
    try {
      if (!ctx) {
        ctx = new AC();
        graph = build();
        dryData = new Uint8Array(graph.dry.frequencyBinCount);
        wetData = new Uint8Array(graph.wet.frequencyBinCount);
        buildRamp();
        buildRowMap(graph.dry.fftSize, ctx.sampleRate);
      }
    } catch (e) {
      setPhase('idle');
      if (hint) hint.textContent = 'This browser could not create the audio graph, so the demonstration is unavailable.';
      return;
    }
    ctx.resume().then(function () {
      /* a graph must never be built twice by repeated Start */
      if (token !== startToken) return;        /* Stop was pressed while resuming */
      setPhase('running');
      nextVoiceT = 0; nextSweepT = 0;
      applyStages();
      gateSources();
      startScheduler();     /* audio scheduling, independent of rendering */
      startLoop();          /* rendering, suppressed under reduced motion */
    }).catch(function () {
      if (token !== startToken) return;
      setPhase('idle');
      if (hint) hint.textContent = 'The browser would not start audio. Press Start to try again.';
    });
  }

  /* One idempotent teardown. Safe to call repeatedly and from any state.
     It never starts a sound: returning to the tab requires Start again. */
  function cleanup(nextPhase) {
    startToken++;                              /* any pending resume is now stale */
    stopLoop();
    stopScheduler();
    if (staticTimer !== null) { clearInterval(staticTimer); staticTimer = null; }
    releaseMic('stopped');
    if (graph && ctx) {
      var t = ctx.currentTime;
      ['out', 'voice', 'sweep', 'mic'].forEach(function (k) {
        try { graph[k].gain.cancelScheduledValues(t); graph[k].gain.setTargetAtTime(0, t, 0.02); } catch (e) {}
      });
      if (ctx.suspend) { try { ctx.suspend(); } catch (e) {} }
    }
    nextVoiceT = 0; nextSweepT = 0;
    setPhase(nextPhase || 'stopped');
    syncPhrasePlayback();          /* phase is set first, so this halts the clip */
  }

  function stop() { cleanup('stopped'); }

  playBtn.addEventListener('click', function () {
    if (phase === 'running') stop(); else start();
  });

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (micBtn) { micBtn.disabled = true; micBtn.title = 'No microphone access in this browser'; }
  }

  /* Before the first Start the panels would otherwise be an empty box, which reads
     as broken. Say what will appear there instead. */
  function drawIdle() {
    if (phase !== 'idle') return;
    c2d.fillStyle = '#F4F6F7'; c2d.fillRect(0, 0, W, H);
    drawGutter();
    var cx = GUT + (W - GUT) / 2;
    c2d.textAlign = 'center';
    c2d.fillStyle = '#475866';
    c2d.font = '600 22px "Source Sans 3", system-ui, sans-serif';
    c2d.fillText('Press Start to hear the call', cx, TOP_Y + PANEL / 2 + 7);
    c2d.fillStyle = '#5C6A75';
    c2d.font = '400 18px "Source Sans 3", system-ui, sans-serif';
    c2d.fillText('the same call, after the simulated radio, appears here', cx, BOT_Y + PANEL / 2 + 6);
    c2d.textAlign = 'left';
  }

  setPhase('idle');
  syncPresetFromStages();
  drawIdle();
  /* the web font may arrive after this script runs; redraw once it has */
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawIdle);
})();
