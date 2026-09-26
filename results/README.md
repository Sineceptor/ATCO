# Results

Every number in the paper, the README and the project page comes from a file
here. Each file records how it was produced; the scripts are in
[evaluation/](../evaluation/README.md).

## 2026: the retraining

| File | What it holds |
| --- | --- |
| [retraining.json](retraining.json) | Every model on the 101 validation clips and the 74 unseen-recording test clips: word error rate for each decoding setting, the setting chosen on validation, 95% intervals, the paired change from the 2025 model, training details (epoch budget, when and why each run stopped), the runs never scored on test (`not_scored_on_test`), and the noise sweeps, including two more models and the first sweep without repetition guards |
| [extraction_agreement.json](extraction_agreement.json) | For each model, how often the callsign, commands, numbers and waypoints extracted from its transcript match those from the correct transcript, with whole-clip counts with and without waypoints, and which missed callsigns start with an unfamiliar word |
| [letters_digits.json](letters_digits.json) | Spelled phonetic-alphabet letters and digit words transcribed exactly, per model, on validation and test, with the commonest confusions and the two rejected trials (alphabet prompt, letter-weighted loss) |
| [callsign_errors.json](callsign_errors.json) | Each callsign the best and 2025 models miss, sorted by cause (letters or digits, tagger, airline word, mostly lost); counts only |
| [radio_chain_noise.json](radio_chain_noise.json) | A simulation of the 2025 radio script's noise stages: how much of the engine noise and static survive its own 300-3,400 Hz filter |
| [band_pass_power.json](band_pass_power.json) | How much power the 300-3,400 Hz filter removes from each of the 74 test clips, which sets how much less noise the filtered noise sweep added |
| [clips/](clips/) | The split clip by clip (`split.csv`: all 874 clip IDs with recording, airport and split) and every model's word errors on each validation and test clip (`errors_*.csv`), so the intervals can be checked. IDs and counts only, no transcripts |
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
