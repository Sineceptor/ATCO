# Project history

## Timeline

| When | What happened |
| --- | --- |
| 2025 | The idea started with a team: an AI that could act like an air traffic controller. That was too big for the time I had and the tools available then, so I narrowed it to one part of the problem: could software take some load off controllers by turning radio speech into structured text that can be checked? The team did not continue. None of the code, data or writing here is theirs. |
| Late 2025 | All the model training and the experiments in [experiments/](../experiments/): Whisper fine-tuning, LoRA on Whisper-large-v3, augmentation, my synthetic radio speech, decoding experiments, word-list labels, BERT taggers, the small-LLM comparison and the SpeechT5 readback. I started a paper and did not finish it. I did not present or submit the project anywhere. |
| Mid 2026 | A first attempt to rebuild and tidy the project, which I dropped because I ran out of time. |
| September 2026 | I re-scored the saved models with new evaluation code, joined the three models into one app, renamed and documented the old experiments, and published this repository. |
| Late September 2026 | I fixed what the review found and retrained: a test set of recordings no model had heard, the start-token bug, checkpoints chosen on validation clips, controlled tests of my synthetic data and of 10.5 hours of extra real speech, model soups, and Whisper-medium trained with LoRA adapters; the average of two Whisper-medium models became the best model. Results in [results/](../results/README.md) and the [paper](../paper/atco_paper.pdf). |

The Git history starts in 2026 because I did the original work in a plain
folder with no version control. Dates before that are from memory and from
document metadata, not from commits.

## What I did and what I used

I chose the problem and the data, planned the experiments, ran all the training
and evaluation on my own machine, and decided what to try next at each stage.
The labelling word lists, the ATC text rules, the range checks, the sentence
generator and the radio simulation were my designs.

In 2026 I went back and reviewed everything: I separated the original files
from later rewrites, joined the models into `atco/pipeline.py`, wrote new
evaluation scripts in [evaluation/](../evaluation/) and checked every old
result. That review is how I found the recording overlap between training and
test clips, the test-set leakage into my prompts and synthetic data, the
start-token bug and the too-loose waypoint rule. None of those were known to me
in 2025.

Whisper, BERT, DistilBERT, SpeechT5, HiFi-GAN, wav2vec2, Phi-3 and Qwen are
other people's models, used through the Hugging Face libraries. I am not
proposing a new model or training method. I used AI coding tools to help write
the code.

## The name

ATCO stands for Air Traffic Control Officer. The project has nothing to do with
the ATCO2 research project except that it uses their public data
([data credit](data.md)).
