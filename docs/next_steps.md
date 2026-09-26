# Next steps

## Done in September 2026

| Question | Answer | Where |
| --- | --- | --- |
| Did fine-tuning help at all? | Yes, by far the biggest effect: unmodified Whisper-small scores 56.87% on the 74 clean clips, my 2025 model 18.64% | [retraining.json](../results/retraining.json) |
| How much of the old score was memorised recordings? | None that I can detect: on clips from recordings it never trained on, the 2025 model does slightly better (18.64%) than on all 175 (20.31%) | same file |
| Does fixing the start-token bug help? | Not measurably: 19.28% against 18.64%, well within noise | same file |
| Does my synthetic radio speech help, tested properly? | No: 20.34%, slightly worse than training on the real clips alone | same file |
| Does more real speech help? | A little on its own: 10.5 hours of UWB-ATCC gave 18.21% and made the model far less likely to get stuck repeating itself; averaged with the fixes-only model (a model soup) it gave 16.72% | same file |
| Does a bigger model help, done properly? | Yes: Whisper-medium with LoRA adapters and the same real speech gave 15.23%, and 15.65% on the ATCO2 clips alone, so most of the gain is from size. Averaging those two models gave 14.38%, 4.26 points better than 2025 with a 95% range that excludes zero. Training longer did not help | same file |
| How much of the error reaches the instruction? | Most clips still lose at least one field; callsigns are the weakest part | [extraction_agreement.json](../results/extraction_agreement.json) |
| How well does the recipe do on every recording? | 19.16% pooled over all 874 clips in five-fold cross-validation by recording, against 53.44% unmodified | [cross_validation.json](../results/cross_validation.json) |
| How does accuracy fall as noise is added? | See the noise sweep in [retraining.json](../results/retraining.json) | `noise_sweep` |

## Still open

| Question | How | What it would settle |
| --- | --- | --- |
| Is the tagger any good against a person? | `python -m evaluation.hand_labels export`, label about 1,000 words by hand, then `python -m evaluation.hand_labels score` | Replaces a score that only measures agreement with my own word-list rules |
| Are spelled letters and digits the problem? | Partly: the best model gets 94.39% of them right, but one wrong word breaks a callsign. A phonetic-alphabet prompt made things worse, and weighting the loss on those words gained four of 479 on validation at the cost of overall WER | [letters_digits.json](../results/letters_digits.json) |
| Why are callsigns so hard? | Not unfamiliar airlines: every missed callsign on the clean clips starts with a word from the training data. Most are misheard letters and digits ("hotel delta lima" as "hotel golf lima"). Next: training audio of spelled-out registrations, chosen on validation | Callsigns are the field that decides who an instruction is for |
