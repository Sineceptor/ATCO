# Who did what

A plain account of my part, the parts I borrowed, and what AI tools did.

## 2025: the original project

- **The question.** After listening to pilots' radio on a seaplane, I chose the
  problem: can small, free models turn ATC radio into text a computer can check?
  I narrowed a team idea for an "AI controller" down to that.
- **The data.** I chose the free one-hour ATCO2 set and split it into training
  and test clips (the split that later turned out to leak recordings).
- **My designs.** The word-list labelling rules, the ATC text rules, the range
  checks on squawk codes, headings, frequencies, levels and runways, the sentence
  generator that follows phraseology rules, and the simulated radio channel.
- **Running it.** I ran all the training and evaluation on my own laptop (an RTX
  3070), about thirty experiments: Whisper-small and Whisper-large-v3 with LoRA,
  noise and speed copies, 4,808 generated clips, decoding prompts, BERT and
  DistilBERT taggers, a small-LLM comparison and a spoken readback.
- **Help.** I used AI coding tools to help write that code.

## 2026: the audit and retraining

I decided to go back over the whole project before showing it to anyone, and
did it with an AI coding assistant (Claude Code), working on my laptop.

- **The assistant** ran the audit that found the four problems (the recording
  overlap, tuning on the test clips, the label mismatch and the scorer that
  changed its own targets). It wrote most of the new code (the evaluation
  scripts, the joined-up app, the training changes and the checker that ties
  every published number to `results/`), proposed most of the experiments and
  ran them, and drafted the site, the paper and these docs. Two later reviews of
  the repo were also done with AI assistants.
- **I** set the goals: a better result on a fair test, and a write-up with no
  overclaiming. I decided which experiments to run and when the laptop was free
  for them, approved each run (several were the assistant's suggestions), chose
  between the options offered at each step (for example a fresh public history,
  the project's own voice for the demo clip, and how to word the AI
  disclosure), and decided what was published.
- **Still to do by me:** labelling the tagger's 1,001 test words by hand, so the
  tagger can be scored against a person (the sheet and the rules for awkward
  words are ready).

## Borrowed

Whisper, BERT, DistilBERT, SpeechT5 and HiFi-GAN, wav2vec2, Phi-3 and Qwen are
other people's models. LoRA, model soups, SpecAugment and the paired bootstrap
are other people's methods. The audio comes from ATCO2 and UWB-ATCC
([data credit](data.md)); the generated clips were spoken by Microsoft Edge
voices. I am not proposing a new model or method: the contribution is the
investigation and what it found.

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
