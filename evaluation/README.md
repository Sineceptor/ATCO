# Evaluation

Every number in [results/](../results/) comes from a script in this folder.
From the repository root, with Python 3.12:

```bash
python -m pip install -r requirements.txt -r evaluation/requirements.txt
```

Model loads are local; the folder names are in [setup](../docs/setup.md). Scores
are written to `outputs/evaluation/`, and the summaries that matter are copied
into `results/`.

## The 2025 checkpoints, re-scored

```bash
python -m evaluation.evaluate_models asr          # results/rescoring_2025/speech_recognition.json
python -m evaluation.evaluate_models distilbert   # results/rescoring_2025/entity_extraction.json
python -m evaluation.evaluate_models bert         # results/rescoring_2025/bert_reference.json
python -m evaluation.check_data_splits            # results/rescoring_2025/data_splits.json
python -m evaluation.check_archived_scores        # results/rescoring_2025/archived_speech_scores.json
```

`asr` transcribes each clip three ways: greedy, beam search with the 2025
repetition guards (`beam`), and plain beam search (`beamplain`). `--asr-model`
scores another checkpoint, `--manifest` another set of clips.

## The 2026 retraining

```bash
python -m evaluation.session_split --from-original   # train / validation / unseen test
python -m training.train_whisper --output outputs/training/fixed
python -m evaluation.evaluate_models asr --asr-model outputs/training/fixed/best \
    --manifest datasets/speech_split/test.jsonl --output outputs/evaluation/fixed_test
python -m evaluation.summarise_retraining             # results/retraining.json
```

The split keeps the 2025 training clips and divides the old test clips by
recording: 101 clips that share a recording with training become validation
data, and 74 from recordings no model has heard become the test set. Every
choice (checkpoint, decoding, which run is best) is made on validation.
`summarise_retraining.py` reports each model on the unseen clips with a 95%
interval from resampling clips, and the paired difference from the 2025 model.

| Script | What it measures |
| --- | --- |
| `compare_models.py` | Word error rate of several models on the same clips, with bootstrap intervals |
| `noise_sweep.py` | Word error rate as white noise is added, optionally after a 300 to 3,400 Hz filter |
| `extraction_agreement.py` | How often a transcription error changes the callsign, command or value the tagger extracts |
| `session_split.py --folds 5` | Five-fold cross-validation by recording, so every clip is tested by a model that never heard its recording |
| `hand_labels.py` | Exports words for labelling by hand, then scores the tagger against them (not done yet) |

## Limits

The tagger's reference labels come from word-list rules, not from people, so its
scores measure agreement with those rules. The 74 unseen clips are a small test
set, which is why every comparison carries an interval. Read
[evaluation notes](../docs/evaluation.md) before comparing numbers across files.

## Application checks

```bash
python -m unittest discover -s tests -v
node tests/test_audio.cjs
```
