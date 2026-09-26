# Where every number on the project page comes from

Every figure on the [project page](https://sineceptor.github.io/ATCO/) is read
from a file in [results/](../results/). `site/tools/check_evidence.py` recomputes
them from those files and fails if the page shows anything different, and the
charts are drawn from the same files by `site/tools/build_charts.py`. No model
runs in the browser.

## The test that everything is scored on

| Claim | Source |
| --- | --- |
| 699 training clips, 175 old test clips | [`data_splits.json`](../results/rescoring_2025/data_splits.json) (`asr_train_rows`, `asr_test_rows`) |
| 101 of the 175 test clips share a recording with training clips (92 shared recordings) | same file (`asr_test_clips_in_shared_sessions`, `asr_shared_sessions`) |
| The 74 "clean" test clips come from recordings no model trained on; the other 101 are used only to make choices | `split_original_test` in [`evaluation/session_split.py`](../evaluation/session_split.py), tested in [`tests/test_retraining.py`](../tests/test_retraining.py) |

With 74 clips, a difference of one or two points can be chance, so every
comparison in [`retraining.json`](../results/retraining.json) carries a 95%
interval from resampling whole clips, and differences are paired.

## Speech recognition on the 74 clean clips

All from [`retraining.json`](../results/retraining.json), `runs.<name>.headline_test`.
Each model's decoding setting was chosen on the validation clips.

| On the page | Run | What it does not show |
| --- | --- | --- |
| 56.87% | `zero_shot`: Whisper-small, not fine-tuned | It writes "4402" where the references spell digits out; digits are spelled out before scoring, but other formatting differences remain |
| 18.64% | `checkpoint_2025`: the 2025 model the app used | Its checkpoint was picked in 2025 by looking at the old test set, which included these clips |
| 19.28% | `fixed`: retrained with the start-token fix and validation-based choices | |
| 20.34% | `synthetic`: as fixed, plus my synthetic radio clips | |
| 18.21% | `real_data`: as fixed, plus 10.5 hours of UWB-ATCC speech | |
| 16.72% | `soup_fixed_real`: the average of the fixed and real_data weights | Its 1.92-point gain over 2025 has a 95% interval that crosses zero |
| 15.23% | `medium_real`: Whisper-medium with rank-32 LoRA adapters, real clips plus UWB-ATCC | |
| 15.65% | `ablations.medium_atco2_only`: the same, on the ATCO2 clips only; declared in advance as not eligible for selection | |
| 14.38%, 4.26 points better than 2025, at least 1.96 at the edge of the 95% range | `soup_medium`: the average of the two Whisper-medium models; the best model on validation, so the chosen one | One test set of 74 clips; each model trained once |

`continued` (the 2025 model trained further with the fix) never beat its
starting point on validation, so no checkpoint was saved.

## From words to instructions

From [`extraction_agreement.json`](../results/extraction_agreement.json): the
DistilBERT tagger is run on the correct transcript and on each model's
transcript, and the extracted fields are compared clip by clip. The tagger's
labels on the correct transcript are the target, and they come from rules, so
this measures how much transcription error reaches the extracted fields, not
the tagger's own accuracy.

## Spelled letters and digits

From [`letters_digits.json`](../results/letters_digits.json): 94.39% for the best
model and 93.22% for the 2025 model on the test clips, counting spelling
variants of the same letter or digit as the same word. The alphabet-prompt trial
is under `rejected_trials`.

## The older numbers

| On the page | Source | Why it is not the headline any more |
| --- | --- | --- |
| 20.31%: 423 word edits across 2,083 reference words | [`speech_recognition.json`](../results/rescoring_2025/speech_recognition.json), `metrics.beam` | Mixes the clean clips with 101 from training recordings, and the settings were tuned on the same clips |
| About 19%, not counted | same file, `metrics.beam_cleaned` | That scorer rewrote known mistakes into the right answer on both sides |
| 76.18, 76.23, 76.67 | a 2025 comment, line 22 of [`eval_vocab_prompt.py`](../experiments/asr/06_decoding_and_prompts/eval_vocab_prompt.py) | Labelled `acc-`; the scorer changed the text first and the test file cannot be confirmed |
| 27.86%, 34.42%, 28.78% | [`asr_large_lora_robustness.txt`](../results/original_2025/asr_large_lora_robustness.txt) | A different training method, compression and decoding, on the old test clips |

## The tagger

| On the page | Source |
| --- | --- |
| 0.8657 weighted F1, per-label F1, the confusion matrix, 66 of 140 "ordinary" words tagged as waypoints | [`entity_extraction.json`](../results/rescoring_2025/entity_extraction.json) |
| 0.9544 for a separate BERT tagger, against labels that differ on 25 words | [`bert_reference.json`](../results/rescoring_2025/bert_reference.json), [`data_splits.json`](../results/rescoring_2025/data_splits.json) |

Both scores measure agreement with labels made by my word-list rules, not with a
person.

## Data and the traced call

| On the page | Source |
| --- | --- |
| 4,808 generated clips, 31 voices, the generator's ranges | the synthetic training list (4,808 lines; the generator aimed for 5,000) and [`synthesize_atc_sentences_with_tts.py`](../experiments/asr/04_augmentation_and_synthetic_speech/synthesize_atc_sentences_with_tts.py) |
| The seven radio stages, and µ-law giving 257 levels | [`simulate_radio_channel.py`](../experiments/asr/04_augmentation_and_synthetic_speech/simulate_radio_channel.py) |
| 10.5 hours of UWB-ATCC speech | [`training/prepare_uwb_atcc.py`](../training/prepare_uwb_atcc.py) output: 11,291 clips |
| The traced call: transcript, labels, 6.12 seconds | [`docs/evaluation.md`](../docs/evaluation.md) and [`docs/images/app_output.png`](../docs/images/app_output.png) |
| "proceed and start approved" from the best 2026 model, "sion start approved" from the soup | their saved transcripts of that clip, one of the 74 clean clips |
| The squawk check | `validate_physics` in [`atco/entity_extraction.py`](../atco/entity_extraction.py) |

No ATCO2 or UWB-ATCC audio is published on the page or in this repository.

## Outside sources

| On the page | Source |
| --- | --- |
| Whisper trained on 680,000 hours, 438,000 of them English | the [Whisper-small model card](https://huggingface.co/openai/whisper-small) |
| Model soups | Wortsman et al., [Model soups](https://arxiv.org/abs/2203.05482), ICML 2022 |
| LoRA | Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021 |
| The Haneda collision | the Japan Transport Safety Board [interim report](https://jtsb.mlit.go.jp/eng-air_report/interim20241225-JA722A_JA13XJ.pdf); the page makes no claim about its cause |
| Speech recognition filling in radar labels | Helmke et al., [Aerospace 10(6), 538, 2023](https://www.mdpi.com/2226-4310/10/6/538) |
| ATCO2 and UWB-ATCC | Zuluaga-Gomez et al., [arXiv:2211.04054](https://arxiv.org/abs/2211.04054); Šmídl et al., Language Resources and Evaluation 53, 2019 |

## The demo audio

The spoken call in the radio demo is one sentence from my own generator, read by
a text-to-speech voice ([provenance](audio/PROVENANCE.md)). It is not a recording
of a real controller or pilot.
