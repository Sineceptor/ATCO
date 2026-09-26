# Hand-labelling guide for the word tagger

The tagger has only ever been scored against labels made by my own word-list
rules, so its F1 of 0.8657 measures how well it copies the rules. To score it
against a person, I label the 1,001 words of its 89 test sentences by hand.

These rules are written down before any labelling, so that the labels can't
drift towards what the tagger or the word lists say. The sheet deliberately
shows neither.

```bash
python -m evaluation.hand_labels export   # writes outputs/evaluation/hand_labels.tsv (already done)
# fill in the 'label' column, one label per word
python -m evaluation.evaluate_models distilbert
python -m evaluation.hand_labels score
```

## The five labels

| Label | Use it for | Examples |
| --- | --- | --- |
| CALLSIGN | Every word of the name of the aircraft being called or calling: airline name plus flight number, or a registration spelled in letters, including "heavy" or "super" when said as part of it | QANTAS SEVEN SEVEN ZERO TWO; OSCAR KILO PAPA MIKE BRAVO; EUROTRANS ONE THREE JULIETT |
| COMMAND | Every word of what the aircraft is told or asks to do | CLIMB; DESCEND; CLEARED TO LAND; LINE UP AND WAIT; HOLD SHORT; CONTACT; SQUAWK; TURN LEFT; STARTUP APPROVED |
| VALUE | Every word of a number the instruction carries, with the words that say what kind of number it is | FLIGHT LEVEL SEVEN ZERO; ONE SIX RIGHT (a runway); ONE TWO ONE POINT EIGHT (a frequency); HEADING TWO SEVEN ZERO; FOUR FOUR ZERO TWO (a squawk) |
| WAYPOINT | Every word of a named place the aircraft is sent to or reports: fixes, reporting points, named stands, holding points and taxiways | TUMKA; BERNUM; HOLDING POINT ALPHA; POSITION THREE FOXTROT |
| O | Everything else | GOOD MORNING; ROGER; AND; THE; TO; station names such as SION TOWER; words like RUNWAY or VIA on their own |

## Decisions for the awkward cases

1. **Words inside a command** ("cleared *to* land", "line up *and* wait") are
   COMMAND. The same small words anywhere else are O.
2. **"Flight level", "heading", "runway" before a number** belong to the VALUE.
   "Runway" with no number after it is O.
3. **A letter after a runway or holding point** ("runway one six *right*",
   "holding point *alpha*") belongs to that VALUE or WAYPOINT, not to a callsign.
4. **A callsign said only in part** ("one three juliett" on its own) is still
   CALLSIGN if it names the aircraft.
5. **The station being called** ("Sion tower", "Sydney ground") is O: it is not
   the aircraft and not a place the aircraft is sent to.
6. **Readbacks** are labelled exactly like the original instruction.
7. **A misheard or garbled word** in the transcript is labelled by what it is
   doing in the sentence, if that is clear; otherwise O.
8. **Not sure:** write `?`. Those words are left out of the score, and the
   number left out is reported. A blank label stops the scoring, so nothing
   unfinished is scored by accident.

Label one sentence at a time using the `whole_turn` column, and don't look at the
stored labels, the tagger's output or `training/entity_labels.py` while doing it.
