// Audio contract checks without using a real microphone or browser permissions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    textContent: '', checked: false, listeners: {}, className: '',
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    replaceChildren() { this.textContent = ''; },
    append(...items) { this.textContent += items.filter(x => typeof x === 'string').join(''); },
  });
  return elements.get(id);
}
let permission, captured, stopped = 0, lastNode;
class FakeAudioContext {
  constructor(options) { this.sampleRate = options.sampleRate; this.state = 'running'; }
  audioWorklet = { addModule: async () => {} };
  async resume() {}
  async close() { this.state = 'closed'; }
  createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
  createGain() { return { gain: {}, connect() {}, disconnect() {} }; }
}
class FakeWorklet {
  constructor() { this.port = {}; lastNode = this; }
  connect() {}
  disconnect() {}
}
const context = vm.createContext({
  document: { getElementById: element, createElement: () => element(Symbol()) },
  window: { addEventListener() {} },
  navigator: { mediaDevices: { getUserMedia: () => new Promise(resolve => { permission = resolve; }) } },
  AudioContext: FakeAudioContext, AudioWorkletNode: FakeWorklet,
  Blob, ArrayBuffer, DataView, Float32Array, TextDecoder, Uint8Array,
  URL: { createObjectURL(blob) { captured = blob; return 'blob:test'; }, revokeObjectURL() {} },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../atco/web/app.js'), 'utf8'), context);
const stream = () => ({ getTracks: () => [{ stop() { stopped++; } }] });
const flush = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  // Signed PCM endpoints and RIFF sizes must be readable by the Python loader.
  const blob = vm.runInContext('wav(new Float32Array([-1, 0, 1]), 16000)', context);
  const bytes = Buffer.from(await blob.arrayBuffer());
  assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
  assert.equal(bytes.readUInt32LE(24), 16000);
  assert.equal(bytes.readUInt32LE(40), 6);
  assert.deepEqual([44, 46, 48].map(i => bytes.readInt16LE(i)), [-32768, 0, 32767]);

  // Cancelling a pending permission request also stops a stream granted later.
  const first = element('record').listeners.click();
  assert.equal(element('record').textContent, 'Cancel microphone request');
  await element('record').listeners.click();
  permission(stream()); await first;
  assert.equal(stopped, 1);
  assert.equal(element('record').textContent, 'Record voice');

  // AudioWorklet chunks become a WAV and stop exactly at the 30-second limit.
  const second = element('record').listeners.click();
  permission(stream()); await second;
  assert.equal(element('record').textContent, 'Stop recording');
  lastNode.port.onmessage({ data: new Float32Array(31 * 16000).fill(0.25) });
  await flush();
  const recording = Buffer.from(await captured.arrayBuffer());
  assert.equal(recording.readUInt32LE(40), 30 * 16000 * 2);
  assert.equal(stopped, 2);
  assert.equal(element('transcribe').disabled, false);
  assert.equal(element('record').textContent, 'Record voice');

  // Verify that the worklet averages channels, rather than doubling their level.
  let Processor;
  const worklet = vm.createContext({ AudioWorkletProcessor: class {}, Float32Array,
    registerProcessor(name, implementation) { Processor = implementation; } });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../atco/web/recorder.js'), 'utf8'), worklet);
  const processor = new Processor();
  let mono;
  processor.port = { postMessage(samples) { mono = samples; } };
  processor.process([[new Float32Array([1, -1]), new Float32Array([-1, 1])]]);
  assert.deepEqual([...mono], [0, 0]);
  console.log('Audio checks passed: PCM, permission cancellation, 30-second cutoff, channel mixing.');
})().catch(error => { console.error(error); process.exitCode = 1; });
