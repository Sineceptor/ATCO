# Where `atco-demo.mp4` comes from

| | |
| --- | --- |
| What it shows | The real ATCO app (`app.py`) running on my laptop in a browser, recorded on 1 October 2026 |
| Speech model | `models/whisper-medium-2026`, the 12-epoch Whisper-medium soup (SHA-256 of `model.safetensors`: c3f8eef99fba103b021f65928663061ff0afb1cdecbc692370f450a64fab612d) |
| First input | `site/audio/phrase.m4a` converted to WAV: a sentence from my own generator read by the project's SpeechT5 voice (see `site/audio/PROVENANCE.md`). Not ATCO2 audio |
| Second input | The typed text "jetstar 1 squawk 4582" |
| How it was made | A Playwright script (`site/tools/record_demo.py`) opened the app in headless Chromium, uploaded the file, pressed the buttons and added the captions at the bottom; ffmpeg converted the recording to H.264 MP4. Nothing was edited or cut |
| Sound | None: browser recordings carry no audio, so the captions describe each step |
| What it is not | Benchmark evidence. The measured results are on the page and in `results/` |
