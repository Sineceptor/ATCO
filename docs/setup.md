# Running ATCO

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. On an external Mac drive,
keep the environment on the internal disk if filesystem sidecar files interfere
with Python packages. The existing Mac installation already uses a local environment.

## Model files

Provide complete local model directories with these names:

| Directory | Required for |
| --- | --- |
| `models/whisper-medium-2026/` | The app's speech model: the 2026 Whisper-medium soup (the average of two Whisper-medium models), the most accurate model; runs on an Apple or NVIDIA GPU if there is one |
| `models/whisper-small-2026/` | The 2026 retrained Whisper-small (the average of two fine-tuned checkpoints): three times smaller and faster, used if the medium folder is missing |
| `models/whisper-small/` | The 2025 fine-tuned Whisper-small checkpoint, which the re-scoring in `evaluation/` uses |
| `models/distilbert/` | Extracting entities with the saved token-classification checkpoint |
| `models/speecht5/` | Optional readback using the pretrained SpeechT5 TTS model and processor |
| `models/hifigan/` | Optional readback using the pretrained SpeechT5 HiFi-GAN vocoder |
| `models/bert-reference/` | Reproducing the report's separate TensorFlow BERT evaluation |
| `models/bert-pretrained/` | Starting the retained TensorFlow BERT fine-tuning script |

These names are local folder names, not model download identifiers. The project
checkpoints are not included in GitHub. The configured working copy uses links
to the existing local files. Replacing a checkpoint with a generic pretrained
model will not reproduce the reported results.

Use `--models-dir /path/to/models` to keep the same layout elsewhere. The app and
command line also accept `--asr-model`, `--ner-model`, `--tts-model` and
`--vocoder-model` to override individual directories. The browser app loads local
files only. The command-line `--allow-downloads` option is explicit and requires
appropriate model identifiers or paths.

## Browser

```bash
python app.py
```

Open **http://127.0.0.1:8765**. If that port is occupied, use
`python app.py --port 8767` and open the corresponding address. The server listens
only on the local machine. Stop it with Ctrl+C in its terminal.

Select an audio file or allow microphone access to record. Recordings are limited
to 30 seconds. The app processes one request at a time and loads models for each
request. Text, JSON and optional WAV outputs are saved in `outputs/browser/`.
WAV uploads retain their bytes; other browser-supported formats are converted
into mono WAV.

## Command line

```bash
python transcribe.py --audio /path/to/recording.wav --tts none
python transcribe.py --audio /path/to/recording.wav --tts speecht5
python transcribe.py --text "jetstar one squawk four five eight two" --tts none
```

Outputs go to `outputs/latest/`. Use `--output-dir /path/to/run` to keep a separate
run. Typed text skips speech recognition, and `--tts none` skips speech synthesis.
The readback uses extracted words in their original order. Recognition or labelling
errors can change its meaning, and the simple command checks do not catch every error.
