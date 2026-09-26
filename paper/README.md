# Paper

[atco_paper.pdf](atco_paper.pdf) is the full write-up: the 2025 model, the audit
that found its evaluation was flawed, and the 2026 retraining on recordings no
model had heard. [atco_paper.md](atco_paper.md) is the same text in Markdown.

Every table and figure comes from [results/](../results/). To rebuild them:

```bash
python -m pip install matplotlib playwright
python -m playwright install chromium
python paper/build_figures.py    # figures/*.png from results/
python paper/build_pdf.py        # atco_paper.pdf, via pandoc and Chromium
```

`build_pdf.py` needs [pandoc](https://pandoc.org) on the PATH. The numbers in the
paper's tables are checked against `results/` by `site/tools/check_evidence.py`.
