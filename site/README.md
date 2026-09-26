# Project page

A single page about this project for someone who has a couple of minutes. Plain
HTML, CSS and JavaScript: no build step, no framework, no dependencies.

| File | What it is |
| --- | --- |
| `index.html` | The whole page |
| `style.css` | All the styling |
| `story.js` | In-page navigation, the highlighted menu item, opening every section for printing |
| `radio.js` | The simulated radio demo (Web Audio API) |
| `images/app-output.jpg` | A copy of the app screenshot from `docs/images/` |
| `audio/phrase.m4a` | The spoken call used in the demo ([where it comes from](audio/PROVENANCE.md)) |
| `EVIDENCE.md` | Every claim on the page and the file it rests on |
| `tools/build_charts.py` | Draws the page's charts from `results/retraining.json` |
| `tools/check_evidence.py` | Checks every number on the page against `results/`, and the page's rules |
| `tools/build_single.py` | Packs the page into one HTML file that can be sent without hosting |

## Look at it locally

```bash
python3 -m http.server 8000 --directory site
```

Then open http://127.0.0.1:8000.

## Check it

```bash
python3 site/tools/check_evidence.py
```

This recomputes every number on the page from `results/`, redraws the charts to
make sure they match, and checks the page's rules. After new results, run
`python3 site/tools/build_charts.py` first. Before sending the page to anyone, run it with `--release`,
which also checks that the repository is public and that every GitHub link on
the page resolves.

## Publish it

`.github/workflows/pages.yml` uploads this folder, without the Markdown files
and `tools/`, to GitHub Pages whenever `site/` changes on `main`. It only works
once the repository is public and **Settings → Pages → Source** is set to
**GitHub Actions**. The page then appears at https://sineceptor.github.io/ATCO/.

## Rules the page follows

- **No ATCO2 audio or transcripts.** The corpus is released for research use and
  is not redistributed. The app screenshots are the only corpus-derived content.
- **No live model.** The checkpoints are several gigabytes and run on my own
  computer. The traced radio call is a saved run from the app, labelled as one.
- **Every number comes from the repository.** If it is not in `results/`,
  `docs/` or `experiments/`, it is not on the page. Nothing is rounded for
  display: the page shows the same two decimals as the result files.

## When there are new results

Put the summary in `results/`, run `python -m evaluation.summarise_retraining`
if it is a speech model, then `python3 site/tools/build_charts.py`, update the
text, and run the checker.
