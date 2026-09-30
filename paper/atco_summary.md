---
title: "ATCO: transcribing air traffic control radio. One-page summary"
author: Ethan Zhang, 2025 to 2026
---

**The question.** Can a small, free speech model turn air traffic control (ATC) radio, which is fast, accented and squeezed through a narrow radio channel, into text a computer can check? I started with one hour of real ATC audio (874 clips from the ATCO2 corpus) and later added 10.5 hours from a second corpus (UWB-ATCC).

**What I built.** Whisper fine-tuned on ATC speech, a word tagger that labels callsigns, commands and numbers, range checks on the numbers, and a spoken readback, joined into one app.

**What an audit found.** An audit I ran in 2026 with an AI assistant found my 2025 test was not clean: 101 of its 175 clips came from recordings that also gave training clips, and I had tuned settings on them. The test was rebuilt from the 74 clips whose recordings no model trained on, checkpoints and decoding were chosen on the other 101, and each later run's adoption rule was written down before it started. Some later ideas still came after earlier test results were known, so the test is disjoint from training but not untouched.

| Word error rate (lower is better) | Result | 95% range |
| ---------------------------------------- | ------------: | ------------ |
| Whisper-small, not fine-tuned | 56.87% | 49.53 to 64.81 |
| Whisper-medium, not fine-tuned | 43.88% | 38.09 to 50.11 |
| My 2025 model (Whisper-small) | 18.64% | 14.44 to 23.19 |
| **My best model (two Whisper-medium models averaged)** | **14.27%** | **10.93 to 17.99** |
| Whisper-small recipe, every clip, split by recording | 19.16% | 17.90 to 20.45 |
| Whisper-small recipe, on an airport it never heard | 26.48% | 24.96 to 27.99 |

The first four rows are on the 74 unseen clips; the last two pool all 874 clips. The best model is 4.37 points better than the 2025 model (95% range 2.07 to 6.76); most of the gain comes from the bigger model.

**What didn't work.** My synthetic radio speech (4,808 generated clips) didn't help when tested properly, and a measurement showed its "engine noise" had almost all vanished inside my own filter. Restarting a model for more epochs, a phonetic-alphabet prompt and a letter-weighted loss all failed to beat the model they started from.

**What it can't do.** The tagger extracts the same callsign, command and numbers from the best model's transcript as from the correct one in only 38 of 74 clips, and it has not yet been scored against labels written by a person. It is a research prototype, never to be used for real air traffic communication.

**Who did what.** I chose the problem, designed the word lists, rules, range checks, sentence generator and radio simulation, and ran the 2025 experiments. I used AI coding tools throughout: in 2025 to help write the code, and in 2026 an AI assistant (Claude Code) audited the project with me, wrote most of the new code, ran the retraining experiments and drafted and edited the write-up, while I set the goals and made the final calls on what to run and publish.

Code, results and every experiment: github.com/Sineceptor/ATCO. Full paper and a one-minute demo: sineceptor.github.io/ATCO.
