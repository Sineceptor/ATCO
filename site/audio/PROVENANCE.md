# Where `phrase.m4a` comes from

| | |
| --- | --- |
| Words | "Turn right, heading one seven two, Speedbird one eight zero four." |
| Where the words come from | One output of my own sentence generator, `generate_sentence()` in `experiments/asr/04_augmentation_and_synthetic_speech/synthesize_atc_sentences_with_tts.py`, run with seed 11 |
| Voice | The project's own readback voice: `AtcSpeaker` in `atco/speech_synthesis.py`, which uses Microsoft's SpeechT5 text-to-speech model and HiFi-GAN vocoder (MIT licence) with the app's fixed speaker setting |
| Made | September 2026, on my own computer, with no online service |
| Check | Unmodified Whisper-small transcribes it as "Turn right heading 172 Speedbird 1804." |
| Format | AAC, mono, 16 kHz (the rate the project's models use), 3.10 seconds, 23,038 bytes |
| What it is not | Not a recording of a real controller or pilot, and not ATCO2 audio. No model runs on it on the page |

To replace it, put a new mono file at `site/audio/phrase.m4a` and update this
note. No code changes are needed. If the file is missing, the demo still works
with its two browser-generated sources, the test vowel and the sweep.
