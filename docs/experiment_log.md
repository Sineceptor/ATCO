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
| B, `synthetic` | A plus 700 of my synthetic radio clips per epoch | 23.08% | 20.34% | +1.70 (−1.29 to +4.87) | No: synthetic speech did not help. Its airline list included callsigns picked from old test errors (favours B), and its transcripts say "decimal" and "x-ray" where ATCO2 says "point" and "x ray" (B wrote "decimal" once on test) |
| C, `augmented` | A plus noise and speed copies of the training clips | not run | not run | | Dropped for time |
| D, `continued` | The 2025 model trained further with the corrected labels | never beat its start (22.55% → best 22.90%) | | | No: nothing saved |
| E, `real_data` | A plus 1,400 UWB-ATCC clips per epoch (10.5 hours of real speech) | 21.68% | 18.21% | −0.43 (−3.13 to +2.42) | No, but far less looping: 19.38% greedy against 26.52% for A |
| Soup of A and E | Weight average | 17.66% | 16.72% | −1.92 (−4.54 to +0.75) | Best Whisper-small, but its gain over 2025 is not certain |
| Soup of A, B and E | Weight average | 19.23% (plain beam, its best) | not scored | | No: worse than the soup of A and E on validation |

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
| `medium_longer` | `medium_real` restarted with fresh adapters and a new warm-up, up to 3 more epochs | never beat its start (17.92% → 18.09%, 18.44%) | | | No: nothing saved. Not the same as training the original run for longer |
| `soup_medium` | Weight average of the two Whisper-medium models | 14.86% | 14.38% | −4.26 (−6.61 to −1.96) | Was the best model until the 12-epoch run. Built after both ingredients had been scored on test, and one of them was the ablation |
| `medium_long` | `medium_real`'s settings with 12 planned epochs (section 13); best epoch 9 | 15.65% | not scored | | Candidate only; its soup did better on validation |
| **`soup_medium_long`** | **Weight average of `medium_long` and the ATCO2-only Whisper-medium** | **14.42%** | **14.27%** | **−4.37 (−6.76 to −2.07)** | **Yes, under the rule written first (165 validation errors against 170): the best model, used by the app. One word better than `soup_medium` on test (−0.11, −1.56 to +1.50), so the two are level** |

## 7. Noise

White noise added to the 74 test clips at a fixed signal-to-noise ratio,
greedy decoding with repetition guards ([retraining.json](../results/retraining.json),
`noise_sweep`).

| Added noise | Best model (12-epoch soup) | 6-epoch soup | 2025 model | Unmodified |
| --- | ---: | ---: | ---: | ---: |
| none | 15.97% | 15.44% | 19.81% | 58.79% |
| 10 dB | 19.06% | 20.87% | 28.01% | 70.50% |
| 0 dB | 39.51% | 40.26% | 48.03% | 93.08% |

The Whisper-small soup reaches 45.79% at 0 dB and the single Whisper-medium
model 42.81% (`noise_sweep.other_models`). 0 dB means the noise has the same
average power as the whole clip, pauses included. The first noise sweep used
plain greedy decoding and jumped about because a few noisy clips looped; it was
redone with the guards (both are in `retraining.json`). With a 300 to 3,400 Hz
filter applied first, the best model scores 15.55% with no added noise and
33.87% at 0 dB (the 6-epoch soup: 16.93% and 36.42%), but the filter removes a median 2.05 dB of each clip's power, so
filtered clips got about 2 dB less noise; that alone could explain the gap
([band_pass_power.json](../results/band_pass_power.json)).

A simulation of the 2025 radio script's noise stages found that its "engine"
noise was about 99% below 20 Hz and that its own 300 to 3,400 Hz filter kept a
median 0.012% of it, leaving it 34 dB below the static
([radio_chain_noise.json](../results/radio_chain_noise.json)).

## 8. From words to instructions

The app's tagger run on the correct transcript and on each model's transcript
([extraction_agreement.json](../results/extraction_agreement.json)).

| Field matches the one from the correct transcript | Unmodified | 2025 model | 6-epoch soup | Best model (12-epoch soup) |
| --- | ---: | ---: | ---: | ---: |
| Callsign (59 clips) | 4 | 35 | 37 | 38 |
| Command (56 clips) | 19 | 37 | 44 | 44 |
| Numbers (58 clips) | 11 | 35 | 41 | 46 |
| Callsign, command and numbers (74 clips) | 3 | 29 | 34 | 38 |
| Every field, including waypoints (74 clips) | 0 | 20 | 23 | 25 |

Waypoint labels come from a rule known to be too loose, so the first of those
two rows is the headline. On it the best model is only just ahead of the other
2026 models (Whisper-small soup and single Whisper-medium both 37), and a few
clips out of 74 is within noise. Spelling variants (alfa and alpha, oskar and
oscar) count as the same word, as in section 9.

None of the best model's 24 missed callsigns starts with a word absent from
the training transcripts. Sorted by cause
([callsign_errors.json](../results/callsign_errors.json)): 14 are letters or
digits heard wrong, missed or added; 7 were transcribed word for word but the
tagger labelled them differently; 1 is a misheard airline name; 2 are mostly
lost. For the 6-epoch soup's 24: 15, 7, 1 and 1; for the 2025 model's 25: 15,
6, 4 and 0.

## 9. Callsign letters and digits

Only the phonetic-alphabet letters and digit words, which callsigns are built
from, counting spelling variants of the same letter or digit (alfa and alpha,
niner and nine) as the same ([letters_digits.json](../results/letters_digits.json)).

| Letters and digits right | Validation | Test |
| --- | ---: | ---: |
| Unmodified Whisper-small | 60.96% | 66.20% |
| 2025 model | 91.23% | 93.24% |
| 6-epoch soup (the model both trials below started from) | 96.03% | 94.41% |
| Best model (12-epoch soup) | 96.24% | 94.64% |

| Trial | Result on validation | Adopted |
| --- | --- | --- |
| Give Whisper the phonetic alphabet as a prompt | Under its best decoding (plain beam): WER 14.95% against 14.86% (171 errors against 170), letters 94.27% against 95.54%. Under guarded beam it was much worse (16.96%), possibly because the guards also count the prompt's words | No: no measurable gain; never scored on test |
| Train the best model 2 more epochs with the loss on letter and digit words weighted three times (`--letter-weight 3`) | Epoch 1: letters and digits 96.87% (4 more of 479 right) but WER 15.38%. Epoch 2: 96.45%, WER 16.00% | No: the rule set before the run needed both better; never scored on test. `medium_longer` also got worse after a restart with no weighting, so the restart, not the weighting, may explain the WER rise |

The best model and the single Whisper-medium model have identical rows here
(460 of 479, same confusions). Re-scored from their own prediction files on
27 September 2026: their transcripts differ on 39 of the 101 clips, but they make
the same 19 letter and digit errors.

## 10. The app

| Change | Why |
| --- | --- |
| Uses the best model (`models/whisper-medium-2026`, the Whisper-medium soup), on an Apple or NVIDIA GPU when there is one | Most accurate model |
| Beam search with repetition guards instead of greedy decoding | Greedy decoding looped on noisy clips; the guards did best on validation |
| Drops waypoint labels on English function words such as "and" | The word-list rules, and so the tagger, called them waypoints |
| A missing package is reported as a server problem | It used to be reported as a bad upload |

## 11. How long each run trained

Every run had a planned number of epochs and would stop early if validation did
not improve for two to four epochs (`training` in
[retraining.json](../results/retraining.json)).

| Run | Planned | Run | Best epoch | Stopped |
| --- | ---: | ---: | ---: | --- |
| A, `fixed` | 12 | 12 | 9 | at the plan |
| B, `synthetic` | 8 | 8 | 7 | at the plan |
| E, `real_data` | 8 | 8 | 8 | at the plan, still improving |
| `medium_real` | 6 | 6 | 6 | at the plan, still improving |
| `medium_atco2_only` | 5 | 5 | 5 | at the plan, still improving |
| `medium_long` | 12 | 12 | 9 | at the plan; peaked at epoch 9 |
| D, `continued` | 6 | 4 | none | early: never beat its start |
| `medium_longer` | 3 | 2 | none | early: never beat its start |

The learning rate falls to zero at the end of the plan, so the runs that were
still improving may be undertrained. The 12-epoch repeat of `medium_real`
(section 13) tested this: it peaked at epoch 9 and did better on validation.

## 12. Corrections after a review, 27 September 2026

Before freezing the numbers, I had the whole repo and site reviewed with an AI
assistant, working from my own planning notes. Everything below was checked
against the code and saved outputs before it was changed. Figures in this
section describe the best model at the time, the 6-epoch soup.

| Finding | What changed |
| --- | --- |
| The README and site described the 2025 Whisper-small app | Now describe the current app; the traced call is labelled as the 2025 model |
| "Every field right in 22 of 74" included waypoint labels | Recounted without waypoints and both counts published |
| Spelling variants (alfa and alpha, oskar and oscar) counted as missed callsigns | Treated as the same word in both the letters score and the instruction counts: then-best model (6-epoch soup) 34 of 74 without waypoints, 23 with them; letters and digits 94.41% |
| "Mostly misheard letters and digits" was not counted | Sorted by cause: 15 of the 24 missed callsigns |
| "Training stops early" and "training longer did not help" | Epoch budgets published (section 11); wording now says what was actually run |
| 45.79% and the `continued`, `medium_longer` and three-model soup numbers were not in `results/` | Added to `retraining.json` by the summary script |
| The alphabet prompt was quoted under guarded beam, not its best decoding | Corrected (section 9) |
| The best model and the single Whisper-medium model had identical letter rows | Checked: a real coincidence, not a copy (section 9) |
| The filter explanation in the noise section was wrong | Corrected: the noise level follows the filtered clip's lower power |
| "Degrades more slowly" | Now "loses fewer points"; as a multiple of the starting error it does not |
| Hardware | 2025 on a Windows laptop with an RTX 3070, 2026 on an M3 Pro |
| Citation [8] and the ATCO2 author list | Corrected |
| Rounded large-v3 numbers, "mostly silence", generator ranges, "clean clips", "different airspace", missing credits | Corrected |
| The soup was built after both ingredients were scored on test | Now said wherever the soup is described |

## 13. Decided before running, 27 September 2026

Written and published before either run started.

**Leave one airport out.** The ATCO2 clips come from seven airports (Prague,
Brno, Sion, Bern, Zurich, Bratislava and Sydney). Recipe A is trained seven
times, each time without one airport, and scored on every clip from the airport
it never heard (`python -m evaluation.session_split --by-airport`). This is a
measurement, not a choice: no model from it can be adopted. It will be reported
per airport and pooled over all 874 clips, next to the five-fold
cross-validation by recording (19.16%) on the same clips.

*Result (27 September, 09:48):* 26.48% pooled (95% range 24.96 to 27.99),
7.32 points worse than by recording (paired range 6.32 to 8.36); unmodified
53.44% ([leave_one_airport_out.json](../results/leave_one_airport_out.json)).

| Airport left out | Clips | Never heard the airport | Heard other recordings | Not fine-tuned |
| --- | ---: | ---: | ---: | ---: |
| Prague (LKPR) | 104 | 30.24% | 24.85% | 60.70% |
| Brno (LKTB) | 32 | 22.57% | 23.67% | 57.74% |
| Sion (LSGS) | 258 | 31.81% | 20.00% | 62.08% |
| Bern (LSZB) | 173 | 17.69% | 16.29% | 44.03% |
| Zurich (LSZH) | 126 | 20.74% | 16.54% | 48.67% |
| Bratislava (LZIB) | 79 | 23.85% | 19.08% | 46.53% |
| Sydney (YSSY) | 102 | 41.44% | 18.80% | 56.13% |

Six of the seven models used all 8 planned epochs; Zurich's stopped at 6 (best
epoch 3).

**A longer Whisper-medium run (`medium_long`).** The same settings as
`medium_real`, but with 12 planned epochs instead of 6 (patience 3), so the
learning-rate schedule spans the longer run. Two candidates: `medium_long` on
its own, and its average with the ATCO2-only Whisper-medium. The one with the
lower validation WER (best decoding) is adopted only if it makes fewer than 170
word errors on the 101 validation clips, the current best (14.86%). Only an
adopted model is scored on the test clips.

*Result (28 September, 18:18):* the first two attempts were lost to the external
drive being unplugged (in epoch 2, and in epoch 3 after it was moved to the
internal disk but still read the ATCO2 clips from the drive). The third ran
entirely from the internal disk, with pauses when the Mac was needed, and
reproduced the lost attempts' first epochs exactly. Validation WER during
training (greedy with guards): 29.11%, 26.22%, 25.17%, 21.94%, 20.72%, 20.98%,
18.09%, 20.80%, 17.31%, 18.79%, 18.62%, 18.01%, so the best epoch was 9.

| Candidate | Validation (best decoding) | Word errors |
| --- | ---: | ---: |
| `medium_long` alone | 15.65% | 179 |
| `soup_medium_long` (with the ATCO2-only model) | 14.42% | 165 |

165 is fewer than 170, so the soup was adopted and scored once on test: 14.27%
(134 errors), 4.37 points better than the 2025 model (95% range 2.07 to 6.76).
Against the previous best it is one word better on test (−0.11 points, 95% range
−1.56 to +1.50): the two are level, and the new one is used because the rule
chose it on validation. Its noise sweep, instruction counts, letters score and
callsign causes were re-run, and every page updated.

## 14. The rest of the review, 27 September 2026

| What | Result |
| --- | --- |
| Engine noise in the 2025 radio script | About 99% of its power below 20 Hz; the script's own filter kept a median 0.012%, leaving it 34 dB below the static ([radio_chain_noise.json](../results/radio_chain_noise.json)) |
| Why the filtered noise sweep scored better | The filter removes a median 2.05 dB of each test clip's power, so filtered clips got about 2 dB less noise ([band_pass_power.json](../results/band_pass_power.json)) |
| The audio band | 300 to 3,400 Hz is the telephone band; ICAO's guidance for 8.33 kHz channels assumes about 2,500 Hz of audio |
| Missed callsigns sorted by cause | Then-best model (6-epoch soup): 15 of 24 letters or digits, 7 the tagger's, 1 airline word, 1 mostly lost; the current best: 14, 7, 1 and 2 ([callsign_errors.json](../results/callsign_errors.json)) |
| Spelling variants | Now one word in every measure; changes the instruction counts and letters score slightly (section 12) |
| Squawk check | Accepted codes with a leading zero wrongly; fixed, with tests |
| ATCO2's own word tags as a human reference | Not usable: only 2 of 877 segments are marked as checked |
| Hand labels for the tagger | Sheet exported and the rules for awkward words written first ([guide](hand_labelling_guide.md)); the labelling itself is still to do |
| Clip IDs and per-clip errors | Published in [results/clips/](../results/clips/) |

## 15. The tagger against AI-written labels, 28 September 2026

The tagger has still not been scored against a person. As a stand-in, an AI
model labelled the 1,001 words of its 89 test sentences, following the
[labelling guide](hand_labelling_guide.md) written beforehand and reading only
the sentences, in a separate sheet (`outputs/evaluation/ai_labels.tsv`), so my
own sheet stays blank ([tagger_vs_ai_labels.json](../results/tagger_vs_ai_labels.json)).

| Against the AI labels (982 words; 19 marked unsure) | Words agreeing | Weighted F1 |
| --- | ---: | ---: |
| The tagger (DistilBERT) | 57.33% | 0.6233 |
| The word-list rules | 65.07% | 0.7011 |
| For comparison: the tagger against the rules | 87.01% | 0.8657 |

Most of the disagreement is ordinary words tagged as waypoints (173 of 247) and
callsign words, mostly flight-number digits, tagged as values (71). Some is the
guide's conventions differing from the rules', such as "runway" belonging to the
number after it (27 words). This is not a human evaluation, and it doesn't
replace my own labels.

## 16. Tidying after a second review, 28 September 2026

A second AI-assisted review of the public repo found a few leftovers, all
fixed: one page still quoted the old best model's transcript of the example
call, some experiment-log lines still described the 6-epoch soup as the best
model, one site sentence said every kept run was still improving (run A peaked
at epoch 9 of 12), two places still said "clean" clips, the README promised the
whole page in two minutes, the site's top line didn't say what word error rate
means, and the history didn't say why the public Git history starts on
26 September.

## Not done yet

- The word tagger has never been scored against labels written by a person:
  the sheet and the [labelling guide](hand_labelling_guide.md) are ready. The
  AI-labelled stand-in (section 15) suggests the real score is well below 0.8657.
- More test recordings from new airports, to separate models a point apart.
- Training audio of spelled-out registrations, the other way to work on callsign letters.
