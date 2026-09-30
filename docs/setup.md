# Running ATCO

You need Python 3.12. From the top of the repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate it with `.venv\Scripts\activate`. If the repository is on
an external drive on a Mac, put the environment on the internal disk instead:
macOS writes small hidden files next to everything on some external drives, and
they can break Python packages. That is what I do.

## Model files

The models aren't in this repository, so the app looks for them in these
folders:

| Folder | What it is for |
| --- | --- |
| `models/whisper-medium-2026/` | The app's speech model: my best 2026 model, the average of two Whisper-medium models. Runs on an Apple or NVIDIA GPU if there is one |
| `models/whisper-small-2026/` | My best 2026 Whisper-small, the average of two fine-tuned models. About three times smaller and faster; the app uses it if the medium folder is missing |
| `models/whisper-small/` | My 2025 Whisper-small, which the re-scoring in `evaluation/` uses |
| `models/distilbert/` | The word tagger |
| `models/speecht5/` | The optional spoken readback: the pretrained SpeechT5 model |
| `models/hifigan/` | The optional spoken readback: SpeechT5's HiFi-GAN vocoder |
| `models/bert-reference/` | Repeating the separate TensorFlow BERT evaluation |
| `models/bert-pretrained/` | Starting the old TensorFlow BERT training script |

These are just folder names on my computer, not names you can download. On my
machine they are links to the model files on an external drive. Putting a
different model in one of them will run, but won't reproduce my results.

### Getting the models

My fine-tuned checkpoints aren't downloadable. They are large (about 3 GB for
the app's model), and they are trained on the ATCO2 set, released for research
use, and on UWB-ATCC, whose licence is non-commercial and share-alike; I haven't
yet worked out whether that lets me publish the weights. What you can do:

- **Try the speech step with a public model.** The command line can use
  unmodified Whisper instead of mine:
  `python transcribe.py --audio clip.wav --asr-model openai/whisper-small --allow-downloads --tts none`.
  The word tagger is my own checkpoint too, though, so the rest of the pipeline
  needs `models/distilbert/`. The tests run without any models.
- **Recreate them.** Get the ATCO2 one-hour set and UWB-ATCC ([data](data.md)),
  then follow [experiment_manifest.json](../results/experiment_manifest.json):
  it has the exact command, seed and base-model revision for every run, the
  ingredients of each averaged model, and SHA-256 fingerprints so you can
  confirm your data and checkpoints match mine. The word tagger comes from
  `training/train_distilbert.py`, trained on the labels `training/entity_labels.py`
  makes.
- **Check the scores without running anything.** The per-clip error counts in
  [results/clips/](../results/clips/) reproduce every headline number.

To keep the same layout somewhere else, pass `--models-dir /path/to/models`. The
app and the command line also take `--asr-model`, `--ner-model`, `--tts-model`
and `--vocoder-model` to point at one folder at a time. The browser app only ever
loads local files. The command line will download a model only if you pass
`--allow-downloads` with a real model name or path.

## Browser

```bash
python app.py
```

Then open **http://127.0.0.1:8765**. If that port is taken, run
`python app.py --port 8767` and open that address instead. The app only listens
on your own computer. Stop it with Ctrl+C in its terminal.

Choose an audio file, or allow the microphone and record up to 30 seconds. The
app handles one request at a time and loads the models for each one, so it is
slow to start. It saves the text, a JSON file and the optional WAV readback in
`outputs/browser/`. WAV uploads are kept exactly as they are; other formats the
browser can play are converted to mono WAV first.

## Command line

```bash
python transcribe.py --audio /path/to/recording.wav --tts none
python transcribe.py --audio /path/to/recording.wav --tts speecht5
python transcribe.py --text "jetstar one squawk four five eight two" --tts none
```

Results go to `outputs/latest/`, or to another folder with
`--output-dir /path/to/run`. Typing text with `--text` skips speech recognition,
and `--tts none` skips the spoken readback. The readback says the extracted words
in the order they were heard, so a recognition or labelling mistake can change
what it says, and the simple range checks won't catch every mistake.
