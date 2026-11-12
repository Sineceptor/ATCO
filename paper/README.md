# Paper

[atco_paper.pdf](atco_paper.pdf) is the full write-up: the 2025 model, the audit
that found its evaluation was flawed, and the 2026 retraining on recordings no
model had heard. [atco_paper.md](atco_paper.md) is the same text in Markdown.

Every table and figure comes from [results/](../results/). The full paper uses
a conventional single-column LaTeX layout, with vector figures, complete
tables and numbered references. To rebuild it, install
[pandoc](https://pandoc.org) and [Tectonic](https://tectonic-typesetting.github.io)
on the PATH, then run:

```bash
python -m pip install matplotlib
python paper/build_figures.py    # vector PDFs and Markdown-preview PNGs
python paper/build_pdf.py        # atco_paper.pdf, via pandoc and Tectonic
```

`typesetting.tex` controls the paper's typography and layout. It uses Times New
Roman when installed, with TeX Gyre Termes as the fallback. The build extracts
the abstract, formats the references and uses the existing table contents
directly from the Markdown source. Tectonic may download standard TeX packages
on its first run.

The one-page summary retains its separate HTML/CSS layout in `paper.css`:

```bash
python -m pip install playwright
python -m playwright install chromium
python paper/build_pdf.py atco_summary
```

The numbers in both documents are checked against `results/` by
`site/tools/check_evidence.py`.
