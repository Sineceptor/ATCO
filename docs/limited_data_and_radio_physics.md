# One hour of data: what I tried, and the physics behind the radio simulation

The only free ATC speech data I could find was the one-hour ATCO2 set. After
cleaning and splitting, that left 699 short clips to train on. Whisper-small has
about 240 million parameters. Most of this project was really about that
mismatch, so this page explains what I tried and how well it worked.

## Idea 1: copy the data with changes

The simplest thing was to make altered copies of each clip
([augment_noise_and_speed.py](../experiments/asr/04_augmentation_and_synthetic_speech/augment_noise_and_speed.py)):

- one copy with Gaussian noise added,
- one copy played at 0.9x or 1.1x speed.

This tripled the training set. It is cheap, but the model still only ever hears
the same sentences from the same controllers.

## Idea 2: write my own ATC sentences and have TTS speak them

ATC phraseology is very regular, so I wrote a generator
([synthesize_atc_sentences_with_tts.py](../experiments/asr/04_augmentation_and_synthetic_speech/synthesize_atc_sentences_with_tts.py))
that builds sentences from templates: taxi, line-up, take-off, landing,
go-around, radar vectors, ILS approach, handoff and squawk. I tried to keep the
numbers realistic:

| Item | Rule in the generator |
| --- | --- |
| Heading | 001 to 360, always three digits, spoken digit by digit |
| Flight level | FL060 to FL430, any whole number (so FL347 can appear) |
| Speed | 160 to 310 knots in steps of 10 |
| Frequency | 118 to 136 MHz plus a valid decimal, spoken with "decimal" |
| Squawk | four digits using only 0 to 7, because transponder codes are octal |
| Wind | direction in degrees plus 3 to 25 knots |

Each sentence was spoken by Microsoft Edge TTS using one of 31 English voices
(US, UK, Irish, Australian, Indian, Singaporean, Hong Kong, Filipino, Nigerian,
Kenyan, South African and others) at a random speaking rate, because real
controllers and pilots have every accent. The generator aimed for 5,000 clips;
4,808 of them made it into the training list.

## Idea 3: make the clean TTS audio sound like a radio

TTS audio is studio-clean. Real ATC audio is narrow, noisy and distorted. If I
trained on clean speech the model would learn the wrong thing, so I wrote
[simulate_radio_channel.py](../experiments/asr/04_augmentation_and_synthetic_speech/simulate_radio_channel.py)
to push every TTS clip through a fake radio. At the time I called the output my
"GAN" data. That was the wrong word. A GAN trains two networks against each
other, and nothing here is trained. It is signal processing, and this is where
the project touches physics.

The chain, in order:

1. **Band-limiting by resampling.** The clip is resampled from 16 kHz down to
   8 kHz and back. By the sampling theorem a signal sampled at 8 kHz cannot
   contain anything above 4 kHz, so this throws away the top half of the
   spectrum, like a narrow radio channel does.
2. **Engine noise (brown noise).** I make white noise and take its running sum.
   Summing is a discrete integral, and integrating divides each frequency
   component's amplitude by its frequency, so the power spectrum falls as
   1/f². The result is a low rumble, similar to engine and airframe noise in a
   cockpit.
3. **Static (white noise).** Plain Gaussian noise with a flat spectrum, for
   receiver hiss.
4. **μ-law companding.** The signal is compressed with
   F(x) = sgn(x) · ln(1 + μ|x|) / ln(1 + μ), rounded to a small number of
   levels, and expanded again. The rounding adds quantisation noise that is
   larger for loud samples, which gives the gritty sound.
5. **Voice-band filter.** A 4th-order Butterworth band-pass from 300 to
   3400 Hz. A Butterworth filter is as flat as possible inside the pass band
   and rolls off at about 24 dB per octave outside it.
6. **Overdrive.** The signal is amplified by 1.5 to 3 times and clipped at
   ±0.95. Clipping is non-linear, so it adds harmonics that were not in the
   original, like a microphone held too close.
7. **Squelch tail.** Half the clips get a 150 ms noise burst with a decaying
   envelope at the end, like the click when the transmit button is released.

Things I now know are not quite right. Aircraft voice radio is analogue
amplitude modulation on 118 to 137 MHz, so there is no μ-law stage in a real
aircraft radio; that step is borrowed from telephones. I also never checked my
simulated spectrum against the spectrum of the real ATCO2 clips, which would
have been the obvious way to tune it. And the noise levels were picked by ear,
not from a measured signal-to-noise ratio.

A related piece of radio physics: aviation still uses AM rather than FM on
purpose. With FM the stronger of two overlapping
transmissions captures the receiver and the weaker one disappears. With AM
both are heard, as a squeal, so a controller knows that two aircraft
transmitted at once.

## Idea 4: train in rounds

Instead of mixing everything, I trained in three rounds
([05_three_round_training](../experiments/asr/05_three_round_training/)): real
clips first, then the noise and speed copies at a lower learning rate, then the
simulated radio speech with some real clips mixed back in (one real clip for
every five synthetic ones) at a lower rate again. The idea
was to stop the synthetic data from pulling the model away from real audio.

## Idea 5: target the words it kept getting wrong

The model kept failing on names it had never seen written, such as "Stefanik",
"Sion" and "Eurotrans" (which came out as "year trans"). So I generated extra
TTS sentences around those words and gave them double weight, and I also tried
putting the words into Whisper's decoding prompt.

## Did any of it work?

Not much, as far as I can tell. A comment in one of my 2025 scoring scripts
records 76.18, 76.23 and 76.67 for the model trained on real data only, the
model after the augmented round and the model after the synthetic round. They
are labelled "acc", which in my 2025 scripts meant 100 minus WER, but that
scorer also rewrote parts of the text before comparing, and I can no longer
confirm which test file and settings each number came from. So I do not turn
them into a gain. All they show is that the scores I saw at the time moved by
less than half a point, for a lot of work.

My best explanations, none of them tested yet:

- TTS voices are calm and clear. Controllers are fast, clipped and often
  stressed. My radio filter changed the sound quality but not the way people
  talk.
- The evaluation clips share recordings with the training clips, so a model
  that memorises the real data does well on this test, and extra variety does
  not get rewarded.
- The round-2 script rebuilt every file path to point at the augmented folder.
  The original clips were not in that folder, and the loader returned silence
  for a missing file, so about a third of round 2 was probably silence paired
  with real transcripts.
- Idea 5 was a mistake in method. I picked the "hard words" by looking at
  errors on the test clips, so the test stopped being a fair test.

## The experiment, run in 2026

What I should have measured from the start is how word error rate changes with
signal-to-noise ratio, SNR = 10 log₁₀(P_signal / P_noise) in decibels, for the
fine-tuned model and for plain Whisper-small, with and without the voice-band
filter. I ran it in 2026 with [noise_sweep.py](../evaluation/noise_sweep.py) on
the 74 test clips from recordings no model trained on, adding white noise with a
fixed seed per clip so every model hears the same audio
([results](../results/retraining.json), `noise_sweep`):

| Added noise | Best 2026 model | 2025 model | Unmodified Whisper-small |
| --- | ---: | ---: | ---: |
| none | 15.44% | 19.81% | 58.79% |
| 20 dB | 16.29% | 22.15% | 62.19% |
| 10 dB | 20.87% | 28.01% | 70.50% |
| 0 dB | 40.26% | 48.03% | 93.08% |

The fine-tuned models are better at every noise level, and the best 2026 model,
an average of two Whisper-medium models, also loses fewer points: 5.43 at 10 dB
where the 2025 model loses 8.20, and 24.82 at 0 dB against 28.22. As a multiple
of where each started it is no better (2.61 times at 0 dB against 2.42). The smaller 2026 soup, trained on the same extra real speech,
follows the 2025 curve closely (45.79% at 0 dB), so the robustness seems to come
mostly from the bigger model. Here 0 dB means the noise has the same average power
as the whole clip, pauses included, so while someone is talking the noise is
louder than the speech.

With the 300 to 3,400 Hz filter applied first, the recordings barely change:
16.93% instead of 15.44% for the best 2026 model, 55.80% instead of 58.79% for
the unmodified model. They have already been through a radio, so there is little
outside the voice band to remove. Under heavy noise the filtered versions do
better (36.42% against 40.26% at 0 dB), but that is at least partly bookkeeping: the
noise level is set from the clip's total power, and the filter removes the power
outside the voice band, so a filtered clip gets weaker noise at the same nominal
SNR while its speech in the band hardly changes. The noise is not filtered in
either case, so the same share of it lands in the voice band both times.

It also connects to how Whisper itself works. The model never sees the
waveform. It sees a log-mel spectrogram: a short-time Fourier transform with
25 ms windows every 10 ms, grouped into 80 frequency bands spaced the way
human hearing is. Every stage of my radio chain is an operation on that
spectrum, which is why a band-pass filter or a 1/f² noise floor changes what
the model "hears".
