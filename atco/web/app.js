const $ = id => document.getElementById(id);
let recording = null;
let clip = null;
let previewURL = null;
let busy = false;
let microphoneRequest = 0;
let waitingForMicrophone = false;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}

function controls() {
  $('transcribe').disabled = busy || !!recording || !clip;
  $('file').disabled = busy || !!recording;
  $('parse').disabled = busy || !!recording;
  $('speech').disabled = busy || !!recording;
  $('record').disabled = busy && !waitingForMicrophone;
}

function recordButton() {
  $('record').replaceChildren();
  const dot = document.createElement('span'); dot.className = 'record-dot';
  $('record').append(dot, 'Record voice');
  $('record').classList.remove('recording');
}

function wav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((value, i) => {
    const s = Math.max(-1, Math.min(1, value));
    view.setInt16(44 + i * 2, Math.round(s * (s < 0 ? 32768 : 32767)), true);
  });
  return new Blob([buffer], { type: 'audio/wav' });
}

function setClip(blob, name, duration) {
  if (previewURL) URL.revokeObjectURL(previewURL);
  clip = blob;
  previewURL = URL.createObjectURL(blob);
  $('preview').src = previewURL;
  $('preview').hidden = false;
  $('source').textContent = `${name} · ${duration.toFixed(1)} seconds`;
  controls();
}

async function stopRecording() {
  const current = recording;
  if (!current) return;
  recording = null;
  current.node.port.onmessage = null;
  current.source.disconnect();
  current.node.disconnect();
  current.mute.disconnect();
  current.stream.getTracks().forEach(track => track.stop());
  await current.context.close();
  const samples = new Float32Array(current.length);
  let offset = 0;
  for (const chunk of current.chunks) { samples.set(chunk, offset); offset += chunk.length; }
  setClip(wav(samples, current.rate), 'Microphone recording', samples.length / current.rate);
  recordButton();
  status('Recording ready. Choose Transcribe recording.');
  controls();
}

$('record').addEventListener('click', async () => {
  if (recording) return stopRecording();
  if (waitingForMicrophone) {
    microphoneRequest++;
    waitingForMicrophone = false; busy = false;
    recordButton(); controls(); status('Microphone request cancelled.');
    return;
  }
  const request = ++microphoneRequest;
  waitingForMicrophone = true; busy = true;
  $('record').textContent = 'Cancel microphone request';
  status('Allow microphone access in your browser, or cancel to choose a file.');
  controls();
  let stream, context;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone access is unavailable. Open this page at localhost, or choose an audio file.');
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (request !== microphoneRequest) { stream.getTracks().forEach(track => track.stop()); return; }
    waitingForMicrophone = false;
    controls();
    context = new AudioContext({ sampleRate: 16000 });
    await context.audioWorklet.addModule('/recorder.js');
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, 'atco-recorder');
    const mute = context.createGain(); mute.gain.value = 0;
    const current = { context, stream, source, node, mute, chunks: [], length: 0, rate: context.sampleRate };
    recording = current;
    node.port.onmessage = event => {
      if (recording !== current) return;
      const remaining = current.rate * 30 - current.length;
      const chunk = event.data.subarray(0, remaining);
      current.chunks.push(chunk); current.length += chunk.length;
      const seconds = Math.floor(current.length / current.rate);
      $('timer').textContent = `0:${String(seconds).padStart(2, '0')}`;
      if (current.length >= current.rate * 30) stopRecording();
    };
    source.connect(node); node.connect(mute); mute.connect(context.destination);
    $('timer').textContent = '0:00';
    $('record').textContent = 'Stop recording';
    $('record').classList.add('recording');
    status('Recording… speak one short instruction.');
  } catch (error) {
    stream?.getTracks().forEach(track => track.stop());
    if (context && context.state !== 'closed') await context.close();
    if (request === microphoneRequest) {
      recordButton();
      status(error.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow it in your browser or choose an audio file.' : error.message, true);
    }
  } finally {
    if (request === microphoneRequest) { waitingForMicrophone = false; busy = false; controls(); }
  }
});

$('file').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  clip = null; $('preview').hidden = true; $('source').textContent = 'Reading audio…';
  busy = true; controls();
  let context;
  try {
    if (file.size > 32 * 1024 * 1024) throw new Error('Choose a smaller audio file (32 MB maximum).');
    context = new AudioContext({ sampleRate: 16000 });
    const bytes = await file.arrayBuffer();
    const decoded = await context.decodeAudioData(bytes.slice(0));
    if (!(decoded.duration > 0 && decoded.duration <= 30)) throw new Error('Choose a recording between 0 and 30 seconds long.');
    const mono = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const samples = decoded.getChannelData(channel);
      for (let i = 0; i < mono.length; i++) mono[i] += samples[i] / decoded.numberOfChannels;
    }
    const header = new TextDecoder().decode(bytes.slice(0, 12));
    const isWav = header.startsWith('RIFF') && header.slice(8) === 'WAVE';
    // Keep WAV input unchanged; only convert formats the Python loader cannot read.
    const blob = isWav ? new Blob([bytes], { type: 'audio/wav' }) : wav(mono, decoded.sampleRate);
    setClip(blob, file.name, decoded.duration);
    status('Recording ready.');
  } catch (error) {
    $('source').textContent = 'No recording selected.';
    status(error.name === 'EncodingError' ? 'This file could not be decoded. Try a WAV or MP3 recording.' : error.message, true);
  } finally { if (context) await context.close(); busy = false; controls(); }
});

function showResult(data) {
  $('empty').hidden = true; $('result').hidden = false;
  $('transcript').textContent = data.transcript;
  $('copy').textContent = 'Copy text';
  $('download').href = data.transcript_url;
  $('details').href = data.details_url;
  $('entities').replaceChildren();
  for (const entity of data.entities) {
    const box = document.createElement('div'); box.className = 'entity';
    const label = document.createElement('span'); label.textContent = entity.type;
    box.append(label, entity.text); $('entities').append(box);
  }
  if (!data.entities.length) $('entities').textContent = 'No entities found.';
  $('warnings').replaceChildren();
  for (const warning of data.rule_warnings) {
    const item = document.createElement('li'); item.textContent = warning; $('warnings').append(item);
  }
  $('warnings').hidden = !data.rule_warnings.length;
  $('readback-section').hidden = !data.readback_text;
  $('readback').textContent = data.readback_text;
  $('output-audio').hidden = !data.audio_url;
  if (data.audio_url) $('output-audio').src = data.audio_url;
  else { $('output-audio').removeAttribute('src'); $('output-audio').load(); }
  $('speech-error').hidden = data.tts_status !== 'failed';
  $('speech-error').textContent = data.tts_status === 'failed' ? `Text is ready, but speech generation failed: ${data.tts_error}` : '';
}

async function submit(path, body, contentType) {
  busy = true; controls();
  $('result').hidden = true; $('empty').hidden = true;
  $('output-audio').pause();
  status('Loading the saved models and processing… this may take a minute.');
  try {
    const response = await fetch(path, { method: 'POST', body, headers: {
      'Content-Type': contentType, 'X-ATCO-Speech': $('speech').checked ? 'yes' : 'no'
    }});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Processing failed.');
    showResult(data);
    status(data.tts_status === 'failed' ? 'Text ready. Speech needs attention.' : 'Done. Check the transcript and extracted words.');
  } catch (error) { status(error.message, true); $('empty').hidden = false; }
  finally { busy = false; controls(); }
}

$('transcribe').addEventListener('click', () => submit('/api/transcribe', clip, 'audio/wav'));
$('parse').addEventListener('click', () => {
  const text = $('typed').value.trim();
  if (!text) return status('Enter an instruction first.', true);
  submit('/api/text', JSON.stringify({ text }), 'application/json');
});
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('transcript').textContent); $('copy').textContent = 'Copied'; }
  catch { status('Copy was blocked by the browser. Use Save .txt instead.', true); }
});
window.addEventListener('pagehide', () => recording?.stream.getTracks().forEach(track => track.stop()));
