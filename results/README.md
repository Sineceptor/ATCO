# Results

Every number in the paper, the README and the project page comes from a file
here. Each file records how it was produced; the scripts are in
[evaluation/](../evaluation/README.md).

## 2026: the retraining

| File | What it holds |
| --- | --- |
| [retraining.json](retraining.json) | Every model on the 101 validation clips and the 74 unseen-recording test clips: word error rate for each decoding setting, the setting chosen on validation, 95% intervals, the paired change from the 2025 model, training details, and the noise sweeps |
| [extraction_agreement.json](extraction_agreement.json) | For each model, how often the callsign, commands and numbers extracted from its transcript match those from the correct transcript, and which missed callsigns start with an unfamiliar word |
| [letters_digits.json](letters_digits.json) | Spelled phonetic-alphabet letters and digit words transcribed exactly, per model, on validation and test, with the commonest confusions and the rejected alphabet-prompt trial |
| [cross_validation.json](cross_validation.json) | Five-fold cross-validation by recording over all 874 clips: the retrained Whisper-small and the unmodified model, pooled and per fold |

## 2025 models, re-scored in 2026

In [rescoring_2025/](rescoring_2025/): the 2025 speech model on the original 175
test clips (`speech_recognition.json`), unmodified Whisper-small on the same clips
(`speech_recognition_zero_shot.json`), the two word taggers
(`entity_extraction.json`, `bert_reference.json`), the overlap between training
and test recordings (`data_splits.json`), and the reconstruction of two archived
scores (`archived_speech_scores.json`).

## Original 2025 logs

[original_2025/](original_2025/) keeps result files exactly as the 2025 scripts
wrote them.
