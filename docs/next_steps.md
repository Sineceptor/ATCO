# Next steps

## Done in September 2026

| Question | Answer | Where |
| --- | --- | --- |
| Did fine-tuning help at all? | Yes, by far the biggest effect: unmodified Whisper-small scores 56.87% on the 74 unseen clips, my 2025 model 18.64% | [retraining.json](../results/retraining.json) |
| How much of the old score was memorised recordings? | None that I can detect: on clips from recordings it never trained on, the 2025 model does slightly better (18.64%) than on all 175 (20.31%) | same file |
| Does fixing the start-token bug help? | Not measurably: 19.28% against 18.64%, well within noise | same file |
| Does my synthetic radio speech help, tested properly? | No: 20.34%, slightly worse than training on the real clips alone | same file |
| Does more real speech help? | A little on its own: 10.5 hours of UWB-ATCC gave 18.21% and made the model far less likely to get stuck repeating itself; averaged with the fixes-only model (a model soup) it gave 16.72% | same file |
| Does a bigger model help, done properly? | Yes: Whisper-medium with LoRA adapters and the same real speech gave 15.23%, and 15.65% on the ATCO2 clips alone, so most of the gain is from size. Averaging those two models gave 14.38%. Training the first one for 12 epochs instead of 6 helped it on validation, and its soup, adopted under a rule written first, gave 14.27%, 4.37 points better than 2025 with a 95% range that excludes zero, though only one word better than the 6-epoch soup | same file |
| How much of the error reaches the instruction? | Most clips still lose at least one field: callsign, command and numbers all match in 38 of 74 for the best model; callsigns are the weakest of the three | [extraction_agreement.json](../results/extraction_agreement.json) |
| How well does the recipe do on every recording? | 19.16% pooled over all 874 clips in five-fold cross-validation by recording, against 53.44% unmodified | [cross_validation.json](../results/cross_validation.json) |
| Does it work at an airport it has never heard? | Worse: 26.48% pooled when each airport is left out in turn, against 19.16% by recording; Sydney, the only airport outside Europe, 41.44% | [leave_one_airport_out.json](../results/leave_one_airport_out.json) |
| How does accuracy fall as noise is added? | See the noise sweep in [retraining.json](../results/retraining.json) | `noise_sweep` |

## Still open

| Question | How | What it would settle |
| --- | --- | --- |
| Is the tagger any good against a person? | The sheet is exported and the rules for awkward words are written down in [the labelling guide](hand_labelling_guide.md); label about 1,000 words by hand (2 to 4 hours); an AI-labelled stand-in puts the tagger at 57.33% of words, against 87.01% by its own rules ([tagger_vs_ai_labels.json](../results/tagger_vs_ai_labels.json)); then `python -m evaluation.hand_labels score` | Replaces a score that only measures agreement with my own word-list rules |
| Are spelled letters and digits the problem? | Partly: the best model gets 94.64% of them right, but one wrong word breaks a callsign. A phonetic-alphabet prompt gave no measurable gain under its best decoding, and weighting the loss on those words gained four of 479 on validation at the cost of overall WER | [letters_digits.json](../results/letters_digits.json) |
| Why are callsigns so hard? | Not unfamiliar airlines: every missed callsign on the unseen clips starts with a word from the training data. Of the best model's 24, 14 are misheard letters and digits ("hotel delta lima" as "hotel golf lima") and 7 are the tagger's. Next: training audio of spelled-out registrations, chosen on validation | Callsigns are the field that decides who an instruction is for |
