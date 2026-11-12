# Project credits

## My contribution

In 2025 I chose the question and the one-hour ATCO2 dataset, planned the
experiments and ran the training and evaluation on my laptop. I designed the
word-list labelling rules, ATC text rules, range checks, sentence generator and
simulated radio channel. The original split later turned out to share source
recordings between training and test clips.

For the 2026 review and retraining, I set the goals, chose and approved the
experiments, selected between the options offered and decided what to publish.
The aim was a better result on a fair test and a write-up that reflected the
evidence. Hand-labelling the tagger's 1,001 test words remains to be done.

## Models, methods and data

Whisper, BERT, DistilBERT, SpeechT5 and HiFi-GAN, wav2vec2, Phi-3 and Qwen are
other people's models. LoRA, model soups, SpecAugment and the paired bootstrap
are other people's methods. The audio comes from ATCO2 and UWB-ATCC
([data credit](data.md)); the generated clips were spoken by Microsoft Edge
voices. I am not proposing a new model or method: the contribution is the
investigation and what it found.

## Development support

AI coding tools helped with implementation in 2025. In 2026, Claude Code carried
out the audit, wrote most of the new code, proposed most of the new experiments
and ran them on my laptop, and drafted and edited the site, paper and
documentation. AI assistants also supported the later repository reviews.

## Where to see the process

- [Experiment log](experiment_log.md): every run in order, including the ones
  that failed, the two lost to an unplugged drive, and the rules written before
  each run.
- [Experiment manifest](../results/experiment_manifest.json): the exact command,
  seed, data fingerprints and checkpoint hash for every model.
- A failed idea: [synthetic radio speech](limited_data_and_radio_physics.md),
  which didn't help when tested properly, and the measurement of why its engine
  noise vanished inside my own filter.
- A correction: the "every field right in 22 of 74" count that included
  unreliable waypoint labels, and how it was recounted.
