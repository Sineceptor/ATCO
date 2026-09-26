# Experiment log, September 2026

Every experiment from the 2026 review and retraining, in the order it was run,
including the ones that failed or were never finished. Word error rates (WER)
are on the 74 test clips from recordings no model trained on, unless marked
validation (the 101 clips used for every choice). Numbers come from the files
named in each section; nothing here is rounded.

**The rule for every choice.** Checkpoints, decoding settings and the choice
between models were made on the validation clips only. A model was scored on
the test clips only after it was built; the test clips never decided anything.

## 1. Re-scoring the 2025 work

| What | Result | Source |
| --- | --- | --- |
| 2025 model on the old 175 test clips, beam search | 20.31% (423 edits / 2,083 words) | [rescoring_2025/speech_recognition.json](../results/rescoring_2025/speech_recognition.json) |
| Same, greedy decoding | 20.88% | same file |
| Same, with the 2025 "clean-up" scorer that rewrote mistakes on both sides | 19.09%, not counted | same file |
| Unmodified Whisper-small on the same 175 clips (digits spelled out) | 60.68% beam, 62.46% greedy | [rescoring_2025/speech_recognition_zero_shot.json](../results/rescoring_2025/speech_recognition_zero_shot.json) |
| Test clips sharing a recording with training clips | 101 of 175 (92 recordings) | [rescoring_2025/data_splits.json](../results/rescoring_2025/data_splits.json) |

Findings: the test was not unseen; the test clips had been used to tune the
model; the training labels repeated the start token and lacked the language
and task tokens used at inference; one scorer changed its own targets; the
word tagger was scored against rules, not people.

## 2. A clean test

| What | Result | Adopted |
| --- | --- | --- |
| Split the old test clips by recording: 101 validation, 74 test | Written by `session_split.py --from-original` | Yes, used for everything below |
| 2025 model on the 74 unseen-recording clips | 18.64% (better than on all 175) | Baseline for every comparison |
| 2025 model on the 101 validation clips | 21.68% | |
| Unmodified Whisper-small on the 74 clips | 56.87% | |

## 3. Making training work on the laptop

| Problem | Fix |
| --- | --- |
| The first training run used about 20 GB and pushed the Mac into swap | Stopped it; one clip per step with gradients summed over 16, gradient checkpointing |
| Still too much memory | A cap on GPU memory, so a run fails cleanly instead of swapping |
| Memory crept up every step | Apple GPUs compile a new graph for every tensor shape; labels are now padded to one fixed length. Training then held at about 6 GB (Whisper-small) and 4.9 GB (Whisper-medium) |
| One validation clip looped ("ne sly ne sly ...") and added 68 errors, so checkpoint choice was driven by a single clip | Validation during training uses the same repetition guards as the 2025 decoder. The first run of the corrected model (validation 38.72%, 38.29%, 33.13%, 36.28% over four epochs) was restarted with the guards |

## 4. Whisper-small experiments

All from [retraining.json](../results/retraining.json), `runs`, except where noted.

| Run | What changed | Validation | Test | Change from 2025 (95% range) | Adopted |
| --- | --- | ---: | ---: | --- | --- |
| A, `fixed` | Corrected labels, SpecAugment, checkpoints chosen on validation | 20.02% | 19.28% | +0.64 (−2.13 to +3.39) | No: no measurable change |
| B, `synthetic` | A plus 700 of my synthetic radio clips per epoch | 23.08% | 20.34% | +1.70 (−1.29 to +4.87) | No: synthetic speech did not help |
| C, `augmented` | A plus noise and speed copies of the training clips | not run | not run | | Dropped for time |
| D, `continued` | The 2025 model trained further with the corrected labels | never beat its start (22.55% → best 22.90%) | | | No: nothing saved |
| E, `real_data` | A plus 1,400 UWB-ATCC clips per epoch (10.5 hours of real speech) | 21.68% | 18.21% | −0.43 (−3.13 to +2.42) | No, but far less looping: 19.38% greedy against 26.52% for A |
| Soup of A and E | Weight average | 17.66% | 16.72% | −1.92 (−4.54 to +0.75) | Best Whisper-small, but its gain over 2025 is not certain |
| Soup of A, B and E | Weight average | 19.49% | not scored | | No: worse than the soup of A and E on validation |

## 5. Every recording: cross-validation

Five folds by recording over all 874 clips; each fold trained a fresh copy of
recipe A and was scored on the recordings it never heard
([cross_validation.json](../results/cross_validation.json)).

| Model | WER over all 874 clips | 95% range | Folds |
| --- | ---: | --- | --- |
| Recipe A | 19.16% | 17.90 to 20.45 | 17.74% to 21.01% |
| Unmodified Whisper-small | 53.44% | 51.13 to 55.81 | |

## 6. Whisper-medium experiments

Whisper-medium with its weights frozen and rank-32 LoRA adapters trained
(4.3% of the model), because full fine-tuning does not fit in 18 GB.

| Run | What changed | Validation | Test | Change from 2025 (95% range) | Adopted |
| --- | --- | ---: | ---: | --- | --- |
| `medium_real` | ATCO2 clips plus 1,000 UWB-ATCC clips per epoch, 6 epochs | 16.61% | 15.23% | −3.41 (−5.77 to −1.08) | Was the best until the soup below |
| `medium_atco2_only` | Ablation, declared in advance and not eligible to be chosen: ATCO2 clips only | 17.66% | 15.65% | −2.98 (−5.26 to −0.67) | Ablation. Shows most of the gain is from model size |
| `medium_longer` | `medium_real` trained up to 3 more epochs with fresh adapters | never beat its start (17.92% → 18.09%, 18.44%) | | | No: nothing saved |
| **`soup_medium`** | **Weight average of the two Whisper-medium models** | **14.86%** | **14.38%** | **−4.26 (−6.61 to −1.96)** | **Yes: the best model, used by the app** |

## 7. Noise

White noise added to the 74 test clips at a fixed signal-to-noise ratio,
greedy decoding with repetition guards ([retraining.json](../results/retraining.json),
`noise_sweep`).

| Added noise | Best model | 2025 model | Unmodified |
| --- | ---: | ---: | ---: |
| none | 15.44% | 19.81% | 58.79% |
| 10 dB | 20.87% | 28.01% | 70.50% |
| 0 dB | 40.26% | 48.03% | 93.08% |

The first noise sweep used plain greedy decoding and jumped about because a few
noisy clips looped; it was redone with the guards. With a 300 to 3,400 Hz
filter applied first, the best model scores 16.93% with no added noise and
36.42% at 0 dB.

## 8. From words to instructions

The app's tagger run on the correct transcript and on each model's transcript
([extraction_agreement.json](../results/extraction_agreement.json)).

| Field matches the one from the correct transcript | Unmodified | 2025 model | Best model |
| --- | ---: | ---: | ---: |
| Callsign (59 clips) | 3 | 33 | 35 |
| Command (56 clips) | 19 | 37 | 44 |
| Numbers (58 clips) | 11 | 35 | 41 |
| Every field (74 clips) | 0 | 20 | 22 |

None of the best model's 27 missed callsigns starts with a word absent from
the training transcripts: the errors are misheard letters and digits, not
unknown airlines.

## 9. Callsign letters and digits

Only the phonetic-alphabet letters and digit words, which callsigns are built
from, counting spelling variants of the same letter or digit (alfa and alpha,
niner and nine) as the same ([letters_digits.json](../results/letters_digits.json)).

| Letters and digits right | Validation | Test |
| --- | ---: | ---: |
| Unmodified Whisper-small | 60.54% | 66.36% |
| 2025 model | 91.23% | 93.22% |
| Best model | 96.03% | 94.39% |

| Trial | Result on validation | Adopted |
| --- | --- | --- |
| Give Whisper the phonetic alphabet as a prompt | Worse: WER 16.96% against 14.86%, letters 87.26% against 95.54% | No; never scored on test |
| Train the best model 2 more epochs with the loss on letter and digit words weighted three times (`--letter-weight 3`) | Epoch 1: letters and digits 96.87% (4 more of 479 right) but WER 15.38%. Epoch 2: 96.45%, WER 16.00% | No: the rule set before the run needed both better; never scored on test |

## 10. The app

| Change | Why |
| --- | --- |
| Uses the best model (`models/whisper-medium-2026`, the Whisper-medium soup), on an Apple or NVIDIA GPU when there is one | Most accurate model |
| Beam search with repetition guards instead of greedy decoding | Greedy decoding looped on noisy clips; the guards did best on validation |
| Drops waypoint labels on English function words such as "and" | The word-list rules, and so the tagger, called them waypoints |
| A missing package is reported as a server problem | It used to be reported as a bad upload |

## Not done yet

- The word tagger has never been scored against labels written by a person:
  the sheet is ready (`python -m evaluation.hand_labels export`).
- More test recordings from new airports, to separate models a point apart.
- Training audio of spelled-out registrations, the other way to work on callsign letters.
