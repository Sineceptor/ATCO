# Evaluation methods and results

The first part of this page re-scores the 2025 checkpoints without retraining.
The last part, [the 2026 retraining](#the-2026-retraining), describes the models
trained after the review. The [evaluation instructions](../evaluation/README.md)
show how to repeat each check.

## Speech recognition

| Method | Clips | WER | CER |
| --- | ---: | ---: | ---: |
| Greedy decoding | 175 | 20.88% | 13.14% |
| Beam search | 175 | 20.31% | 12.85% |
| Beam search with historical transcript cleanup | 175 | 19.09% | 12.87% |
| Archived quick-test predictions | 176 | 27.19% | 18.23% |
| Archived beam-cleanup predictions | 175 inferred | 19.24% | 12.92% |

Ordinary normalisation uppercases text, removes punctuation and collapses spaces.
Greedy decoding gives 435 word edits over 2,083 reference words; beam decoding
gives 423/2,083. All 175 recordings completed without failure. The evaluator uses
128 new tokens. Beam search uses five beams, repetition penalty 1.2, length
penalty 1.0 and a three-token repetition block; since 2026 the application
decodes the same way.

Historical cleanup changes both predictions and references using phrase replacements
chosen from observed errors. It removes substrings such as `NE` and `JA` even within
words, and joins spoken digits. `ONE NINE JAPAN` becomes `O NI PAN`. The resulting
386/2,022 word-edit score describes a different scoring target.

The quick-test log uses `datasets/archived/quick_test_manifest.jsonl`, with
566/2,082 word edits. The beam log uses `datasets/speech/test.jsonl`, with
389/2,022 edits. Both log reconstructions assume unlogged rows were exact matches.
The quick-test header confirms 176 clips; the beam log omits its total, so its
175-row reconstruction remains an assumption. These saved predictions differ
from the evaluated checkpoint's predictions. The old logs lack a complete runtime
and checkpoint hash, preventing a controlled explanation of the difference.

The evaluated runtime uses PyTorch 2.8.0 and Transformers 4.57.6. The saved model
configuration names Transformers 4.39.3. Current summaries record model and
manifest hashes.

The speech split has 699 training and 175 evaluation clips. Filenames are disjoint,
but 92 recording sessions occur in both sets, covering 101 evaluation clips
(57.7%). Session IDs are inferred by removing each filename's final clip suffix.
The scores therefore do not establish generalisation to wholly new sessions.

## Entity extraction

| Checkpoint and target version | Correct words | Accuracy | Weighted F1 |
| --- | ---: | ---: | ---: |
| Application DistilBERT, stored labels | 871 / 1,001 | 87.01% | 0.8657 |
| TensorFlow BERT reference, regenerated labels | 955 / 1,001 | 95.40% | 0.9544 |

Both runs score 89 sequences without truncating words. They use the first subword
prediction, collapse BIO prefixes and include the non-entity label `O`. These are
word-label scores rather than exact entity-span scores. The TensorFlow result
matches [the original report](../results/original_2025/ner_bert_word_level.txt) at its
printed precision.

The word sequences match, but the two rule-generated target versions disagree on
25 words. This prevents a controlled comparison between the models. The app uses
DistilBERT; its grouping, repair and command checks are not included in these scores.
The labelling rules in `training/entity_labels.py` reproduce the stored labels.
`evaluation/reference_labels.py` retains the different rules needed for the BERT report.

The raw entity split contains 504 training and 56 evaluation sessions with no
shared session IDs. Filtering turns without a labelled entity leaves 89 of 90
evaluation turns. Both model families used the named test set during development:
DistilBERT selects its best epoch by its F1 score, and TensorFlow uses it as validation
data. These are development evaluation scores with rule-generated references.

## Known weaknesses of the entity labels

The rules label a word WAYPOINT when it has three or more letters and is in no
other word list. The intended test for an all-capitals waypoint name is applied
to text that has already been uppercased, so it is always true. In the README
example, "portion", "startup" and "position" are tagged WAYPOINT, and the
callsign "eurotrans one three juliett" is split into WAYPOINT, VALUE, WAYPOINT
because Eurotrans is not in the (mostly Australian) airline list. About a
quarter of the evaluation words carry the WAYPOINT label. The taggers therefore
learn a word-list lookup, and the F1 scores measure agreement with that lookup.
The rules are left unchanged so that the stored labels and scores stay
reproducible; [next steps](next_steps.md) describes the hand-labelled
evaluation that would replace them.

## Unmodified Whisper-small

On the same 175 clips, with digits spelled out on both sides because it writes
"4402" where the references say "four four zero two", unmodified Whisper-small
scores 60.68% with beam search and 62.46% greedy
([speech_recognition_zero_shot.json](../results/rescoring_2025/speech_recognition_zero_shot.json)).
Other formatting differences remain, such as "X-ray" against "x ray", so part of
the gap is formatting, but most of it is real: callsigns such as "Jetstar seven
sixty seven" come out as unrelated words.

## The 2026 retraining

The retraining keeps the 699 training clips and splits the 175 old test clips by
recording: 101 that share a recording with training become validation clips, and
the 74 from recordings no model trained on become the test set. Checkpoints,
decoding settings and the choice between models were all made on validation.

| Model | Validation WER | Test WER (74 clips) | 95% range |
| --- | ---: | ---: | --- |
| Whisper-small, not fine-tuned | 63.81% | 56.87% | 49.53 to 64.81 |
| 2025 checkpoint | 21.68% | 18.64% | 14.44 to 23.19 |
| Retrained with the fixes | 20.02% | 19.28% | 15.37 to 23.35 |
| + synthetic radio speech | 23.08% | 20.34% | 16.05 to 25.00 |
| + 10.5 hours of UWB-ATCC | 21.68% | 18.21% | 13.78 to 23.48 |
| Weight average of the fixes-only and UWB-ATCC models | 17.66% | 16.72% | 13.00 to 20.68 |
| Whisper-medium with LoRA (rank 32), + UWB-ATCC | 16.61% | 15.23% | 11.69 to 19.13 |
| Weight average of that and the ATCO2-only Whisper-medium | 14.86% | 14.38% | 10.95 to 18.17 |

Each row uses the decoding setting that did best on validation; the full
breakdown, with intervals for the difference from the 2025 checkpoint, is in
[retraining.json](../results/retraining.json). The fixes are the start and task
tokens described in [training](../training/README.md), SpecAugment, and choosing
checkpoints on validation. Continuing to train the 2025 checkpoint with the fixes
never beat it on validation, so no such checkpoint was kept.

Adding white noise to the 74 test clips (greedy decoding with repetition
guards) raises the selected model's error from 15.44% as recorded to 20.87% at
10 dB and 40.26% at 0 dB, against 19.81%, 28.01% and 48.03% for the 2025
checkpoint; unmodified Whisper-small rises from 58.79% to 93.08%. The smaller
2026 soup follows the 2025 curve closely (45.79% at 0 dB; `noise_sweep.other_models`
in [retraining.json](../results/retraining.json)). The selected model loses fewer
points than the 2025 checkpoint, but as a multiple of its starting error it does
not degrade more slowly (2.61 times at 0 dB against 2.42). The signal-to-noise
ratio is measured over the whole clip, pauses included, and after the optional
300 to 3,400 Hz filter, so a filtered clip receives less noise. Without the
repetition guards, greedy decoding sometimes falls into loops on noisy clips,
which makes the curve jump about.

Five-fold cross-validation by recording over all 874 clips, with a fresh
Whisper-small trained for each fold and checkpoints chosen on a held-out tenth of
each fold's training recordings, gives a pooled WER of 19.16% (95% interval
17.90 to 20.45) against 53.44% for the unmodified model; the folds range from
17.74% to 21.01% ([cross_validation.json](../results/cross_validation.json)).

Scored on their own, spelled phonetic-alphabet letters and digit words are
mostly right: 94.41% for the selected model and 93.24% for the 2025 checkpoint on
the test clips, counting spelling variants of the same letter or digit as the
same word ([letters_digits.json](../results/letters_digits.json)). A prompt
listing the phonetic alphabet, under its best decoding on validation (plain
beam), made no measurable difference to WER (171 word errors against 170) and got
slightly fewer letters right (94.27% against 95.54%), so it was not adopted. Under
guarded beam it did much worse, possibly because the repetition guards also count
the prompt's words. Training further with the loss weighted three times on letter
and digit words gained four of 479 of them on validation but made overall WER
worse (15.38% against 14.86%), so it was not adopted either; restarting training
with fresh adapters and no weighting (`medium_longer`) also made WER worse, so the
weighting itself may not be the cause. Every experiment is listed in the
[experiment log](experiment_log.md).

Two cautions. The 2025 checkpoint was chosen in 2025 by looking at the old test
set, which contains these 74 clips, so its score may be slightly optimistic. And
74 clips is a small test: the soup's 1.92-point gain over the 2025 checkpoint has
a 95% interval that crosses zero. Whisper-medium's 3.41-point gain does not
(1.08 to 5.77 points), and it was the best model on validation before the test
clips were scored. An ablation declared in advance, Whisper-medium trained the
same way on the ATCO2 clips alone, scores 17.66% on validation and 15.65% on
test, so most of the gain comes from the larger model
([retraining.json](../results/retraining.json), `ablations`). Averaging the two
Whisper-medium models gave the lowest validation error of all (14.86%), so the
average is the selected model: 14.38% on test, 4.26 points better than the 2025
checkpoint (95% interval 1.96 to 6.61). Two things about that choice should be
said plainly: the ATCO2-only model had been declared an ablation that could not
itself be chosen, and I built the soup after both Whisper-medium models had been
scored on the test clips. The soup was chosen on validation, but the idea was not
blind to the test scores.

Every run that was kept reached its planned number of epochs while validation
WER was still falling (run E at epoch 8 of 8, `medium_real` at 6 of 6, the
ATCO2-only Whisper-medium at 5 of 5), and the learning rate decays to zero at that
point, so these models may be undertrained. Only runs that never beat their
starting point stopped early. Restarting `medium_real` with fresh adapters and a
new warm-up for up to three more epochs (`medium_longer`) never improved on it on
validation; that is not the same as training the original run for longer.

Validation clips share recordings with training clips, so choosing on them can
reward memorising those recordings. Run E is worse than run A on validation but
better on test, which is a sign of this. Validation ties between decodings go to
guarded beam, the first one tried.

Whisper-medium has three times as many weights as Whisper-small. Its original
weights were frozen and rank-32 LoRA adapters were trained on every attention
and feed-forward layer (34.6 million trainable numbers, 4.3% of the model), so
training fitted in 4.9 GB of GPU memory, at about 3 seconds per clip.

## Connected application

A real 6.12-second file passes through the 2025 Whisper-small model, DistilBERT and SpeechT5/HiFi-GAN.
It produces the transcript, extracted words and a 4.96-second mono WAV at 16 kHz.
The browser upload and command-line routes give matching outputs for this file.
The example misrecognises “push and start approved” as “portion startup approved”
(the 2026 Whisper-medium soup hears “proceed and start approved”),
showing that transcription errors can reach extraction and readback.

Since 2026 the application also removes waypoint labels from English function
words such as "and" and "the". The tagger learned those labels from the
word-list rules, which call any unfamiliar word of three or more letters a
waypoint; the scores above describe the tagger's own output, before this step.

This verifies the connection and output files. Physical microphone capture and human
speech intelligibility have not been verified. SpeechT5 retains the original random
speaker embedding. There is no end-to-end instruction accuracy or operational
validation result. A stronger evaluation needs independent human labels, new recording
sessions and direct tests of whole instructions and speech intelligibility.
