# Evaluation methods and results

The first half of this page is what I found when I re-scored my 2025 models in
2026, without training anything. The second half,
[the 2026 retraining](#the-2026-retraining), covers the models I trained after
that. The [evaluation instructions](../evaluation/README.md) say how to repeat
each check.

## Speech recognition

| Method | Clips | WER | CER |
| --- | ---: | ---: | ---: |
| Greedy decoding | 175 | 20.88% | 13.14% |
| Beam search | 175 | 20.31% | 12.85% |
| Beam search with historical transcript cleanup | 175 | 19.09% | 12.87% |
| Archived quick-test predictions | 176 | 27.19% | 18.23% |
| Archived beam-cleanup predictions | 175 inferred | 19.24% | 12.92% |

For the plain scores I only upper-case the text, remove punctuation and
collapse spaces. Greedy decoding makes 435 word edits over the 2,083 reference
words, and beam search 423. Every one of the 175 clips was transcribed. Beam
search here uses five beams, a repetition penalty of 1.2 and no repeated
three-word sequences, and the app has decoded the same way since 2026.

The "historical cleanup" row is the 2025 scorer I no longer count. It rewrote
both the model's output and the correct transcript using replacements I had
picked by looking at mistakes, and it cut strings like `NE` and `JA` out of the
middle of words, so `ONE NINE JAPAN` became `O NI PAN`. Its 386 edits over 2,022
words measure a different target, not the same one more kindly.

The two "archived" rows come from old log files, not from re-running a model.
The quick-test log covers 176 clips (566 edits over 2,082 words); the beam log
(389 edits over 2,022 words) never says how many clips it covered, so 175 is my
guess. To rebuild both I had to assume that any clip the logs left out was
transcribed perfectly. Their predictions also differ from what the saved
checkpoint produces now, and the logs don't record which checkpoint or software
made them, so I can't say why.

I re-scored with PyTorch 2.8.0 and Transformers 4.57.6; the saved model says it
was made with Transformers 4.39.3. Every summary I write now records the model
and manifest hashes, so this can't happen again.

The 2025 split had 699 training and 175 test clips. No file appears in both,
but 92 recordings do: I had split clips, not recordings, so 101 of the 175 test
clips (57.7%) came from a recording the model had also trained on. I worked out
which recording each clip came from by removing the clip number from the end of
its file name. So these scores don't show how the model does on new recordings;
the 2026 section below does.

## Entity extraction

| Checkpoint and target version | Correct words | Accuracy | Weighted F1 |
| --- | ---: | ---: | ---: |
| Application DistilBERT, stored labels | 871 / 1,001 | 87.01% | 0.8657 |
| TensorFlow BERT reference, regenerated labels | 955 / 1,001 | 95.40% | 0.9544 |

Both taggers are scored on the same 89 sentences, with no words cut off. I use
each word's first sub-word, merge the begin and inside tags, and count the
"other" label too, so these are word-label scores, not whole-entity scores. The
BERT number matches [the original report](../results/original_2025/ner_bert_word_level.txt)
to the precision it printed.

The two taggers can't be compared, though. They were trained and scored against
two versions of my labelling rules, which disagree on 25 of the words. The app
uses DistilBERT, and these scores leave out the grouping, repair and command
checks the app runs afterwards. `training/entity_labels.py` reproduces the
labels DistilBERT was scored on; `evaluation/reference_labels.py` keeps the other
version, for the BERT report.

The sentences come from 504 training and 56 test recordings, with none shared.
Dropping the one sentence with no labelled word leaves 89 of 90. Both taggers
also used these test sentences while training: DistilBERT kept whichever epoch
scored best on them, and the BERT script used them as its validation set. So
these are development scores against labels made by rules, not a clean test.

Until I label the words myself, I had an AI model label the same 1,001 words
following my [labelling guide](hand_labelling_guide.md), reading only the
sentences. Against those labels, which are not a person's, the tagger agrees on
57.33% of the 982 words it was sure about (weighted F1 0.6233) and the word-list
rules on 65.07%. Most of the gap is ordinary words called waypoints (173 of 247)
and callsign digits called values (71)
([tagger_vs_ai_labels.json](../results/tagger_vs_ai_labels.json)).

## Known weaknesses of the entity labels

My rules call a word a WAYPOINT if it has three or more letters and isn't in any
other word list. The check that was meant to catch waypoint names written in
capitals runs on text that has already been upper-cased, so it always passes.
That is why, in the README example, "portion", "startup" and "position" are
tagged WAYPOINT, and the callsign "eurotrans one three juliett" is split into
WAYPOINT, VALUE, WAYPOINT: Eurotrans isn't in my mostly Australian airline list.
About a quarter of the test words carry the WAYPOINT label. The taggers have
learnt to copy a word-list lookup, and their F1 scores measure how well they copy
it. I've left the rules as they were so the stored labels and scores can still be
reproduced; [next steps](next_steps.md) describes the hand-labelled test that
would replace them.

## Unmodified Whisper-small

On the same 175 clips, unmodified Whisper-small scores 60.68% with beam search
and 62.46% greedy
([speech_recognition_zero_shot.json](../results/rescoring_2025/speech_recognition_zero_shot.json)).
I spelled out digits on both sides first, because it writes "4402" where the
transcripts say "four four zero two". Other formatting differences remain, such
as "X-ray" against "x ray", so part of the gap is formatting, but most of it is
real: callsigns such as "Jetstar seven sixty seven" come out as unrelated words.

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
| The same average, with the first model trained for 12 epochs | 14.42% | 14.27% | 10.93 to 17.99 |

Each row uses the decoding setting that did best on validation; the full
breakdown, with intervals for the difference from the 2025 checkpoint, is in
[retraining.json](../results/retraining.json). The fixes are the start and task
tokens described in [training](../training/README.md), SpecAugment, and choosing
checkpoints on validation. Continuing to train the 2025 checkpoint with the fixes
never beat it on validation, so no such checkpoint was kept.

Adding white noise to the 74 test clips (greedy decoding with repetition
guards) raises the selected model's error from 15.97% as recorded to 19.06% at
10 dB and 39.51% at 0 dB, against 19.81%, 28.01% and 48.03% for the 2025
checkpoint; unmodified Whisper-small rises from 58.79% to 93.08%. The smaller
2026 soup follows the 2025 curve closely (45.79% at 0 dB; `noise_sweep.other_models`
in [retraining.json](../results/retraining.json)). The selected model loses fewer
points than the 2025 checkpoint, but as a multiple of its starting error it does
not degrade more slowly (2.47 times at 0 dB against 2.42). The signal-to-noise
ratio is measured over the whole clip, pauses included, and after the optional
300 to 3,400 Hz filter, so a filtered clip receives less noise. Without the
repetition guards, greedy decoding sometimes falls into loops on noisy clips,
which makes the curve jump about.

Five-fold cross-validation by recording over all 874 clips, with a fresh
Whisper-small trained for each fold and checkpoints chosen on a held-out tenth of
each fold's training recordings, gives a pooled WER of 19.16% (95% interval
17.90 to 20.45) against 53.44% for the unmodified model; the folds range from
17.74% to 21.01% ([cross_validation.json](../results/cross_validation.json)).
Leaving out one whole airport at a time instead, seven times, gives
26.48% (95% interval 24.96 to 27.99) on the same 874 clips,
7.32 points worse than by recording (paired interval 6.32 to 8.36); Sydney,
the only airport outside Europe, goes from 18.80% to 41.44%
([leave_one_airport_out.json](../results/leave_one_airport_out.json)).

Scored on their own, spelled phonetic-alphabet letters and digit words are
mostly right: 94.64% for the selected model and 93.24% for the 2025 checkpoint on
the test clips, counting spelling variants of the same letter or digit as the
same word ([letters_digits.json](../results/letters_digits.json)). A prompt
listing the phonetic alphabet, under its best decoding on validation (plain
beam), made no measurable difference to WER (171 word errors against 170) and got
slightly fewer letters right (94.27% against 95.54%) than the model it was tried
on (the best model at the time, the 6-epoch soup), so it was not adopted. Under
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
Whisper-medium models gave the lowest validation error at the time (14.86%) and
14.38% on test. Two things about that choice should be said plainly: the
ATCO2-only model had been declared an ablation that could not itself be chosen,
and I built the soup after both Whisper-medium models had been scored on the test
clips. The soup was chosen on validation, but the idea was not blind to the test
scores.

The same average, with its first model trained for 12 epochs instead of 6
(`medium_long`, rule written before the run), made 165 word errors on
validation against 170 and was adopted: 14.27% on test, 4.37 points better than
the 2025 checkpoint (95% interval 2.07 to 6.76). Against the 6-epoch soup it is
one word better on test (−0.11 points, 95% interval −1.56 to +1.50), so the two
are level. Trained alone, the 12-epoch model scores 15.65% on validation against
16.61% for the 6-epoch one, and its best epoch was 9 of 12, so the 6-epoch run was
somewhat undertrained.

Run E and both Whisper-medium runs reached their planned number of epochs while
validation WER was still falling (run E at epoch 8 of 8, `medium_real` at 6 of 6,
the ATCO2-only Whisper-medium at 5 of 5; run A peaked earlier, at 9 of 12), and the learning rate decays to zero at that
point, so these models may be undertrained; for Whisper-medium, the 12-epoch
run above confirmed it. Only runs that never beat their starting point stopped
early. Restarting `medium_real` with fresh adapters and a
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

## The app, end to end

I checked the whole app on one real 6.12-second recording, which says "push and
start approved". It goes through the speech model, the word tagger and the
SpeechT5 readback, and comes out as a transcript, the tagged words and a
4.96-second spoken readback. The browser and the command line give the same
result. My 2025 model hears "portion startup approved", and the current model
hears "provision start approved", so the mistake travels into the tagged words
and the readback either way.

Since 2026 the app also removes waypoint labels from small English words such
as "and" and "the". The tagger learnt to call them waypoints from my word-list
rules, which call any unfamiliar word of three or more letters a waypoint; the
tagger scores above are from before this step.

How far does a transcription error travel? On the 74 unseen clips, the tagger
finds the same callsign, command and numbers in the best model's transcript as
in the correct one for 38 clips, and for 25 when waypoints are counted too
([extraction_agreement.json](../results/extraction_agreement.json)). That is
agreement with the tagger's reading of the correct transcript, not a check by a
person that the instruction is right. I haven't tested a real microphone, how
understandable the spoken readback is, or anything in live operation; the
readback also uses SpeechT5's default speaker setting.
