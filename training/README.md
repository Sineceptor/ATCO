# Training

| Script | What it trains | Output |
| --- | --- | --- |
| `train_whisper.py` | Fine-tunes Whisper-small in full, or Whisper-medium with LoRA adapters, on the speech clips | `outputs/training/<name>/best/` |
| `train_distilbert.py` | The app's word tagger | `outputs/training/distilbert/` |
| `pretrain_bert.py` | Masked-language pretraining for the separate BERT tagger | `outputs/training/bert-pretrained/` |
| `train_bert.py` | Fine-tunes that BERT tagger | `outputs/training/bert-reference/` |
| `prepare_entities.py` | Word labels from the word lists in `entity_labels.py` | `outputs/training/entity_labels/` |
| `prepare_speech.py` | Cleans a speech manifest into a new file | the path you give it |

Run each as a module from the repository root, for example
`python -m training.train_whisper`. Install `requirements.txt` first, plus
`jiwer` for Whisper, and the packages in
[evaluation/requirements.txt](../evaluation/requirements.txt) for the taggers.

## Whisper

```bash
python -m evaluation.session_split --from-original      # writes datasets/speech_split/
python -m training.train_whisper --output outputs/training/fixed
```

The split keeps the 699 training clips the 2025 model used, and divides the 175
old test clips in two: 101 from recordings that also gave training clips become
the validation set, and 74 from recordings no model has trained on become the
test set. Training reads only the training and validation clips. After every
epoch it transcribes the validation clips, keeps the checkpoint with the lowest
word error rate, and stops once four epochs pass without improvement.

`--extra MANIFEST --extra-audio-dir DIR --extra-per-epoch N` mixes N extra clips
into every epoch, for example the synthetic radio speech or the noise and speed
copies of the training clips. All the real clips are used every epoch.

It runs on an Apple silicon GPU, an NVIDIA GPU or the CPU. On a Mac with 18 GB
of memory a run uses about 6 GB: one clip at a time with gradients summed over
16, gradient checkpointing, and labels padded to one fixed length (a new label
length would otherwise make the GPU compile and keep another compute graph).
`--memory-fraction` caps GPU memory so a run stops with an error instead of
pushing the whole machine into swap.

Changes from the 2025 script, which trained the app's original checkpoint:

- **Start and task tokens.** The old labels began `<|startoftranscript|>
  <|notimestamps|>`, and the collator added a second start token, while at run
  time the model is prompted with `<|startoftranscript|><|en|><|transcribe|>
  <|notimestamps|>`. Labels now carry the same prefix as the prompt, and the
  start token appears once.
- **Choosing checkpoints.** The 2025 runs chose checkpoints and decoding settings
  by looking at the test clips. Now the validation clips do that job.
- **SpecAugment.** `--spec-augment` hides random slices of time and frequency
  during training, a standard guard against memorising a small data set.

## Taggers

`prepare_entities.py` applies the word-list rules in `entity_labels.py` to the
saved session split. Those rules label any unfamiliar word of three letters or
more as a waypoint, which is too loose; see [evaluation notes](../docs/evaluation.md).
The DistilBERT and BERT scripts train on those labels and were not retrained in
2026.
