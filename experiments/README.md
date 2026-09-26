# My 2025 experiments

These are the scripts from the original project. The originals had names like
`1.py`, `make_it_real.py` and `train_finale_v2_round3_gan.py`, so I renamed each
one by what it actually does and grouped them in the order the work went. They
are a record, not a package you can install: paths, model folder names and
dependencies are as they were on my 2025 training laptop (Windows, NVIDIA RTX
3070). The code I still
maintain is in [atco/](../atco/), [training/](../training/) and
[evaluation/](../evaluation/).

Every file starts with a comment block giving its original name, what it does
and what is wrong with it. The code itself is unchanged apart from removing
personal file paths. [source_map.json](../docs/source_map.json) lists original names
and hashes.

## What was tried, in order

The order below is inferred from version numbers, comments and which script
loads which model. No file dates or Git history survive from 2025.

1. **Baseline.** Whisper-small fully fine-tuned on 699 ATCO2 clips (about 80% of the one-hour set).
2. **Bigger model.** Whisper-large-v3 in 4-bit with LoRA, 18 numbered versions,
   plus a noise and speed robustness test. It did not beat the small model.
3. **More data.** Noise and speed copies, then 4,808 synthetic sentences spoken
   by TTS and passed through a simulated radio channel, then TTS sentences built
   around words the model kept getting wrong.
4. **Three-round training** on raw, augmented and synthetic data.
5. **Decoding.** Vocabulary prompts, beam sizes and penalties (64+ numbered
   variants), voice-activity trimming, and an LLM correction pass.
6. **Restart** with cleaned lowercase transcripts. This produced the 2025
   checkpoint, which the app used until the 2026 retraining replaced it with a
   Whisper-medium model ([training/train_whisper.py](../training/train_whisper.py)).
7. **Text side.** Rule-generated entity labels, DistilBERT and BERT taggers,
   masked-language-model pretraining, a comparison with Phi-3 and Qwen, and a
   speaker-role side experiment.

## What I found when I went back over it

I found these in 2026, not at the time. I think they matter more than any
single score.

- **My synthetic data barely helped.** The scores I recorded for the raw,
  augmented and TTS-mixed models were within half a point of each other, on a
  2025 scorer I no longer fully trust. Likely reasons: TTS voices do not
  sound like stressed controllers on AM radio, round 2 probably trained partly
  on silence because of a path bug, and the test clips share recordings with
  training, which rewards memorising the real data. A controlled test in 2026,
  on recordings no model had heard, confirmed it: the synthetic clips did not
  help ([results](../results/retraining.json)).
- **The test set leaked into development.** It selected checkpoints, supplied
  the 'hard word' lists for synthetic data and prompts, and was used to tune
  decoding over dozens of runs. Scores from stages 3 to 5 are optimistic.
- **Some scorers changed the question.** Deleting tags by substring cuts into
  real words (in one scorer, ONE becomes O); one scorer removed the words the
  model got wrong; another mapped specific wrong outputs to the right answer.
  Everything in [results/](../results/) is scored plainly for this reason.
- **A start-token bug runs through the training scripts.** Labels already begin
  with Whisper's start token and the collator adds another, and the labels
  lack the language and task tokens used at inference, so training and
  inference prompts do not match. Fixed in
  [training/train_whisper.py](../training/train_whisper.py) in 2026; retraining
  with the fix did not change accuracy measurably.
- **The entity labels are a lexicon, not annotation.** Any unknown word of three
  or more letters becomes WAYPOINT, so the taggers learn to imitate a word list.

## Index

### ASR 1. Data preparation

| Script | Original name | What it does |
| --- | --- | --- |
| [prepare_atco2_segments.py](asr/01_data_preparation/prepare_atco2_segments.py) | `ASR/Data_reduc.py` | Parses the ATCO2 1-hour corpus (XML + WAV), keeps English segments marked correct, cuts the audio into clips and writes one label JSON per clip. |
| [split_train_test.py](asr/01_data_preparation/split_train_test.py) | `ASR/划分训练集.py` | Shuffles all clips (seed 42) and writes train.jsonl / test.jsonl. Original filename was Chinese for 'split training set'. |
| [extract_lid_english.py](asr/01_data_preparation/extract_lid_english.py) | `extract_english.py` | Sorts the separate ATCO2 language-ID dataset into English / Others folders. Nothing else in the project uses its output. |

### ASR 2. Whisper-small, full fine-tuning

| Script | Original name | What it does |
| --- | --- | --- |
| [train_small_500_steps.py](asr/02_whisper_small_full_finetune/train_small_500_steps.py) | `ASR/train_atc.py` | First attempt: full fine-tune of Whisper-small on the raw training clips, lr 1e-5, 500 steps. |
| [train_small_2500_steps_cosine.py](asr/02_whisper_small_full_finetune/train_small_2500_steps_cosine.py) | `ASR/train_atc_v4_test.py` | Full fine-tune of Whisper-small with cached features, lr 5e-5 cosine, 2500 steps. Direct ancestor of the final training script in training/train_whisper.py. |
| [train_small_raw_plus_augmented_3_epochs.py](asr/02_whisper_small_full_finetune/train_small_raw_plus_augmented_3_epochs.py) | `ASR/train_asr_mixed_v5.py` | Full fine-tune of Whisper-small on raw + augmented clips for 3 epochs. |
| [eval_greedy_plain_normalisation.py](asr/02_whisper_small_full_finetune/eval_greedy_plain_normalisation.py) | `ASR2/quick_test.py` | Greedy decoding on the test set with plain scoring (uppercase, no punctuation). This is the honest evaluator; its archived log reconstructs to 27.19% WER. |

### ASR 3. Whisper-large-v3 with LoRA, and a robustness test

Saved result ([asr_large_lora_robustness.txt](../results/original_2025/asr_large_lora_robustness.txt)): 27.86% WER clean, 34.42% with added noise, 28.78% at 1.1x speed. The bigger model with LoRA did not beat the fully fine-tuned small model, so this line was dropped.

| Script | Original name | What it does |
| --- | --- | --- |
| [train_large_v3_lora_v1.py](asr/03_whisper_large_lora/train_large_v3_lora_v1.py) | `ASR/train_atc_improve.py` | Whisper-large-v3 in 4-bit with LoRA (r=32, q/v), lr 1e-3, 1000 steps, random Gaussian noise added on the fly. |
| [train_large_v3_lora_v17_cleaned_labels.py](asr/03_whisper_large_lora/train_large_v3_lora_v17_cleaned_labels.py) | `ASR/train_atc_v17.py` | Same line, 'v17': label cleaning, LoRA r=64 on q/k/v/o, lr lowered to 5e-4 because 1e-3 was too aggressive, 800 steps. |
| [train_large_v3_lora_v18_augmented.py](asr/03_whisper_large_lora/train_large_v3_lora_v18_augmented.py) | `ASR/train_atc_augment.py` | 'v18': the large-v3 LoRA trained on the noise/speed augmented set, 1500 steps. This is the model behind results/original_2025/asr_large_lora_robustness.txt. |
| [train_small_lora_8bit.py](asr/03_whisper_large_lora/train_small_lora_8bit.py) | `ASR/80%train.py` | Side branch: Whisper-small in 8-bit with LoRA (r=32), 3 epochs, best checkpoint by WER. |
| [make_noisy_and_fast_test_sets.py](asr/03_whisper_large_lora/make_noisy_and_fast_test_sets.py) | `ASR/augment_test_robustness.py` | Builds two stress-test copies of the test set: added Gaussian noise, and 1.1x speed. |
| [eval_large_lora_clean_noisy_fast.py](asr/03_whisper_large_lora/eval_large_lora_clean_noisy_fast.py) | `ASR/evaluate_all.py` | Scores the v18 large-v3 LoRA on the clean, noisy and fast test sets. Saved result: 27.86% / 34.42% / 28.78% WER. |

### ASR 4. Augmentation and synthetic radio speech

One hour of audio is very little, so this stage makes more: template sentences spoken by 31 TTS accents, then degraded by a simulated radio channel. Some scripts and comments call this 'GAN' data. It is not a GAN (no generator or discriminator is trained); it is text-to-speech followed by signal processing. The full story is in [limited data and radio physics](../docs/limited_data_and_radio_physics.md).

| Script | Original name | What it does |
| --- | --- | --- |
| [augment_noise_and_speed.py](asr/04_augmentation_and_synthetic_speech/augment_noise_and_speed.py) | `ASR/augment_data.py` | Triples the training set: original + Gaussian-noise copy + 0.9x/1.1x speed copy. |
| [synthesize_atc_sentences_with_tts.py](asr/04_augmentation_and_synthetic_speech/synthesize_atc_sentences_with_tts.py) | `ASR/augment_grammar.py` | Generates 5,000 ATC sentences from templates (callsign, runway, heading, flight level, frequency, wind, squawk) and speaks them with Edge TTS in 31 English accents at varied speaking rates. |
| [simulate_radio_channel.py](asr/04_augmentation_and_synthetic_speech/simulate_radio_channel.py) | `ASR/make_it_real.py` | Makes the clean TTS speech sound like VHF radio: 8 kHz band-limiting, brown (integrated white) engine noise, white static, mu-law companding, 300-3400 Hz Butterworth band-pass, gain + clipping, and a squelch burst on half the clips. |
| [train_frozen_encoder_real_aug_tts.py](asr/04_augmentation_and_synthetic_speech/train_frozen_encoder_real_aug_tts.py) | `ASR/train_atc_augment_v2.py` | Whisper-small with the encoder frozen, trained on real + augmented + radio-simulated TTS data (Adafactor, lr 1e-4, 3 epochs). |
| [train_frozen_encoder_plus_hard_word_tts.py](asr/04_augmentation_and_synthetic_speech/train_frozen_encoder_plus_hard_word_tts.py) | `ASR/train_atc_augment_v3.py` | Adds 2,000 extra TTS sentences built around words the model kept getting wrong (STEFANIK, SION, NAV CHECKER...), weighted 2x. |
| [train_frozen_encoder_plus_vocab_tts.py](asr/04_augmentation_and_synthetic_speech/train_frozen_encoder_plus_vocab_tts.py) | `ASR/train_atc_with_vocab.py` | Generates TTS clips for every 'critical' location and callsign in vocab_master.json and trains on them at 2x weight. |

### ASR 5. Three-round training (raw, augmented, synthetic)

Curriculum idea: learn the real data first, then augmented copies, then synthetic radio speech mixed with real clips.

| Script | Original name | What it does |
| --- | --- | --- |
| [train_round1_raw.py](asr/05_three_round_training/train_round1_raw.py) | `ASR/train_finale_v2.py` | Round 1: full fine-tune of Whisper-small on raw clips, lr 2e-5, 3500 steps. |
| [train_round2_augmented.py](asr/05_three_round_training/train_round2_augmented.py) | `ASR/train_finale_v2_round2.py` | Round 2: continues from round 1 on the augmented set, lr 1e-5, 3000 steps. |
| [train_round3_tts_radio_mixed.py](asr/05_three_round_training/train_round3_tts_radio_mixed.py) | `ASR/train_finale_v2_round3_gan.py` | Round 3: continues from round 2 on radio-simulated TTS speech mixed with 20% real clips, lr 5e-6, 2500 steps. |

### ASR 6. Decoding settings, vocabulary prompts and LLM correction

The comment in `eval_vocab_prompt.py` is the most useful record in this folder: raw 76.18, augmented 76.23, TTS-mixed 76.67, labelled "acc". My 2025 scripts computed "acc" as 100 minus WER, but after a clean-up step that changed the text, and I cannot confirm which test file each number used, so I do not claim a gain from them. They only show that two extra rounds of training on manufactured data did not move the recorded score by more than half a point.

| Script | Original name | What it does |
| --- | --- | --- |
| [build_vocab_master.py](asr/06_decoding_and_prompts/build_vocab_master.py) | `ASR/build_master_vocab.py` | Counts words over all data files and sorts them into hand-written ATC categories to build decoding prompts. |
| [eval_vocab_prompt.py](asr/06_decoding_and_prompts/eval_vocab_prompt.py) | `ASR/eval_vocab_constrained.py` | Evaluates a checkpoint with a vocabulary prompt and beam 5. A comment records "acc" for three models: raw 76.18, augmented 76.23, TTS-mixed 76.67. The recorded scores moved by less than half a point; how they were measured cannot now be confirmed. |
| [eval_structured_prompt.py](asr/06_decoding_and_prompts/eval_structured_prompt.py) | `ASR/v2_raw_inference_v2.py` | Same evaluation with a structured prompt (format hint + longest locations/callsigns), temperature 0 and no conditioning on previous tokens. |
| [eval_round2_prompt_beam15.py](asr/06_decoding_and_prompts/eval_round2_prompt_beam15.py) | `ASR/1.py` | 'V37': evaluates the round-2 model with beam 15, length penalty 0.1 and a 50-word prompt. |
| [eval_small_lora_tuned_decoding.py](asr/06_decoding_and_prompts/eval_small_lora_tuned_decoding.py) | `ASR/eval_vocab_optimized.py` | 'V64': merges the small LoRA adapter and evaluates with beam 20, length penalty 0.42, repetition penalty 1.08. |
| [eval_vad_trim_and_prompt.py](asr/06_decoding_and_prompts/eval_vad_trim_and_prompt.py) | `ASR/eval_advanced.py` | Trims silence with Silero VAD before decoding and uses a hand-written prompt of difficult names. |
| [eval_llm_correction_deepseek.py](asr/06_decoding_and_prompts/eval_llm_correction_deepseek.py) | `ASR/v2_raw_inference_with_deepseek.py` | Whisper output is post-corrected by DeepSeek-chat with guard rails (reject digit changes, large length changes); raw and corrected WER are compared. |

### ASR 7. Scoring variants (why the headline number is the plain one)

| Script | Original name | What it does |
| --- | --- | --- |
| [eval_raw_vs_cleaned_wer.py](asr/07_scoring_variants/eval_raw_vs_cleaned_wer.py) | `ASR/eval_normalizer.py` | Reports WER twice: plainly, and after removing noise tags and foreign greetings from both sides. |
| [eval_excluding_hard_words.py](asr/07_scoring_variants/eval_excluding_hard_words.py) | `ASR/test_exclusion.py` | Reports a 'net WER' after deleting 19 names the model gets wrong. |
| [eval_beam_with_error_mappings.py](asr/07_scoring_variants/eval_beam_with_error_mappings.py) | `ASR2/inference.py` | Beam-5 evaluation of the final checkpoint with a 30-entry mapping table applied to both sides. The rules survive in evaluation/historical_normalization.py, and docs/evaluation.md explains why this 19% figure is not the headline result. |

### NLP 1. Sessions and rule-generated labels

| Script | Original name | What it does |
| --- | --- | --- |
| [merge_clips_into_sessions.py](nlp/01_data_and_rule_labels/merge_clips_into_sessions.py) | `NLP/merge.py` | Groups the per-clip label files by recording into one session JSON with context and an ordered dialogue. |
| [split_sessions_train_test.py](nlp/01_data_and_rule_labels/split_sessions_train_test.py) | `NLP/split_dataset_raw.py` | 90/10 split at session level, seed 42. |
| [build_mlm_corpus.py](nlp/01_data_and_rule_labels/build_mlm_corpus.py) | `NLP/atc_corpus.py` | Dumps every utterance into one text file for masked-language-model pretraining. |
| [label_entities_rules_v1.py](nlp/01_data_and_rule_labels/label_entities_rules_v1.py) | `NLP/prepare_ner_data_strict.py` | First rule labeller: airline list, NATO alphabet, command verbs, number words, session callsigns/waypoints -> BIO tags. Version 2 (with stopwords) is training/entity_labels.py. |

### NLP 2. Tagger training

| Script | Original name | What it does |
| --- | --- | --- |
| [train_distilbert_ner_tensorflow.py](nlp/02_ner_training/train_distilbert_ner_tensorflow.py) | `NLP/train_bert_ner_tf.py` | TensorFlow/Keras version of the DistilBERT tagger (3 epochs, lr 2e-5). The PyTorch version used by the app is training/train_distilbert.py. |
| [probe_mlm_fill_mask.py](nlp/02_ner_training/probe_mlm_fill_mask.py) | `NLP/test_bert_mlm.py` | Prints top-5 fill-mask predictions for seven ATC sentences to see whether MLM pretraining learned phraseology. No metric. |

### NLP 3. Tagger evaluation and demos

| Script | Original name | What it does |
| --- | --- | --- |
| [eval_bert_word_level.py](nlp/03_ner_evaluation/eval_bert_word_level.py) | `NLP/evaluate bert finetuned.py` | Evaluates the MLM-pretrained BERT tagger. Produced the 0.9544 weighted F1 in results/original_2025/. |
| [eval_bert_token_level_first_attempt.py](nlp/03_ner_evaluation/eval_bert_token_level_first_attempt.py) | `NLP/atc inference.py` | Older evaluator of the same model (accuracy 0.5321 in results/original_2025/). |
| [eval_distilbert_with_rule_patch.py](nlp/03_ner_evaluation/eval_distilbert_with_rule_patch.py) | `NLP/bert_analyzer_testset.py` | Evaluates DistilBERT after patching its 'O' predictions with keyword lists, and splits instructions at each command. |
| [eval_distilbert_with_rule_patch_v3.py](nlp/03_ner_evaluation/eval_distilbert_with_rule_patch_v3.py) | `NLP/bert_analyzer_testset_tf.py` | Same, plus a stop-word list that forces words to 'O'. |
| [eval_distilbert_exact_span.py](nlp/03_ner_evaluation/eval_distilbert_exact_span.py) | `NLP/test_bert_prediction.py` | Strict exact-match span precision/recall/F1 for DistilBERT against the stored rule labels. |
| [smoke_test_five_handwritten_sentences.py](nlp/03_ner_evaluation/smoke_test_five_handwritten_sentences.py) | `NLP/test bert.py` | Five hand-written sentences with hand-written labels, as a quick sanity check. |
| [demo_entities_to_intent.py](nlp/03_ner_evaluation/demo_entities_to_intent.py) | `NLP/test_bert_full.py` | Demo: entities -> intent -> standardised text (airline to ICAO code, spoken digits to numerals, e.g. FL300). |
| [interactive_tagger_with_range_checks.py](nlp/03_ner_evaluation/interactive_tagger_with_range_checks.py) | `NLP/nlp_12_inference_full_guard.py` | Interactive tagger with sub-word repair and range checks (squawk digits, heading 0-360, frequency 118-137 MHz, flight level, speed, runway 1-36). The app's atco/entity_extraction.py comes from this file. |

### NLP 4. Small language models as taggers

Saved confusion matrices: [Phi-3](../results/original_2025/ner_phi3_few_shot.png), [Qwen](../results/original_2025/ner_qwen_few_shot.png). Both models 'miss' most rule-labelled WAYPOINT words (161 and 206 of about 258). Given that the rules label almost any unknown word as a waypoint, the language models are often right and the reference is wrong.

| Script | Original name | What it does |
| --- | --- | --- |
| [eval_phi3_mini_few_shot.py](nlp/04_small_llm_comparison/eval_phi3_mini_few_shot.py) | `NLP/evaluate_phi3.py` | Few-shot entity extraction with Phi-3-mini (4-bit): three examples, JSON output, soft matching by value. |
| [eval_qwen25_3b_few_shot.py](nlp/04_small_llm_comparison/eval_qwen25_3b_few_shot.py) | `NLP/evaluate_qwen.py` | Same experiment with Qwen2.5-3B-Instruct. |

### NLP 5. Speaker-role detection and a wav2vec2 baseline

| Script | Original name | What it does |
| --- | --- | --- |
| [train_bert_speaker_role.py](nlp/05_speaker_role_and_wav2vec2/train_bert_speaker_role.py) | `NLP/comparison_train_bert_srd.py` | Side experiment following a published sequential ASR -> speaker-role baseline: BERT tags each word as PILOT or ATCO. Written with heavy LLM assistance from a paper. |
| [train_wav2vec2_ctc_baseline.py](nlp/05_speaker_role_and_wav2vec2/train_wav2vec2_ctc_baseline.py) | `NLP/comparison_train_pure_asr.py` | wav2vec2-base-960h CTC fine-tune as a non-Whisper ASR baseline (lr 4e-4, 2000 steps). |
| [eval_wav2vec2_wer.py](nlp/05_speaker_role_and_wav2vec2/eval_wav2vec2_wer.py) | `NLP/test_asr_comparison.py` | WER/CER of the wav2vec2 baseline. |
| [eval_asr_then_speaker_role.py](nlp/05_speaker_role_and_wav2vec2/eval_asr_then_speaker_role.py) | `NLP/test_asr_srd.py` | End to end: wav2vec2 transcript -> speaker-role tagger, scored on correctly recognised words. |

## Kept elsewhere in the repository

| Original name | Now | What it is |
| --- | --- | --- |
| `ASR2/train_atc.py` | [training/train_whisper.py](../training/train_whisper.py) | The 2025 final Whisper-small training script (2,800 steps, lr 2e-5, cosine schedule), rewritten in 2026 with the corrected labels, checkpoints chosen on validation and a LoRA option. The 2025 version is not kept. |
| `ASR2/clean_data.py` | [training/prepare_speech.py](../training/prepare_speech.py) | Transcript cleaning: removes [HES]/[NE-..] tags, lowercases. |
| `ASR/inference.py` | [atco/speech_recognition.py](../atco/speech_recognition.py) | Single-file transcription. |
| `NLP/prepare_ner_mlm.py` | [training/entity_labels.py](../training/entity_labels.py) | Rule labeller version 2 (with stopwords). |
| `NLP/nlp_05_train_bert.py` | [training/train_distilbert.py](../training/train_distilbert.py) | DistilBERT tagger used by the app. |
| `NLP/train_mlm.py` | [training/pretrain_bert.py](../training/pretrain_bert.py) | Masked-language-model pretraining of BERT-base on ATC text. |
| `NLP/train_mlm_finetune.py` | [training/train_bert.py](../training/train_bert.py) | BERT-base tagger fine-tuned from the MLM model. |
| `TTS/test.py` | [atco/speech_synthesis.py](../atco/speech_synthesis.py) | SpeechT5 readback and ATC pronunciation rules. |
| `ASR/train_finale.py + ASR2/tchr_inference.py` | [evaluation/historical_normalization.py](../evaluation/historical_normalization.py) | The error-mapping scoring rules. |

## Left out

| Original name | Reason |
| --- | --- |
| `ASR2/tchr_inference.py` | Byte-for-byte copy of ASR/train_finale.py apart from two paths. |
| `ASR/train_finale.py` | Despite the name, an evaluator; identical in method to ASR2/inference.py, which is kept. |
| `ASR/quick_test.py` | Same as ASR2/quick_test.py apart from the model path. |
| `ASR/eval_speed_final.py` | Same script as eval_vocab_constrained.py with another model path. |
| `ASR/v2_raw_inference.py` | Superseded by v2_raw_inference_v2.py, which fixes its rate calculation. |
| `NLP/prepare_raw_text.py` | Identical to atc_corpus.py. |
| `NLP/slm.py` | 24-line debug print, not an experiment. |
| `NLP/bert_command_parser.py` | Loads the tagger as a sequence classifier, so the head is random and its output meaningless. |
| `NLP/test_bert_test.py` | Interactive viewer reading a stale dataset path. |
| `demo.py, TTS/main.py` | Connectors added during the 2026 restoration; replaced by atco/pipeline.py. |
