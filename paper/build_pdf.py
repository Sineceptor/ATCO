"""Typeset paper/atco_paper.md as paper/atco_paper.pdf (or another document here).

    python paper/build_figures.py
    python paper/build_pdf.py               # the paper
    python paper/build_pdf.py atco_summary  # the one-page summary

Needs pandoc on the PATH and Playwright's Chromium
(python -m pip install playwright && python -m playwright install chromium).
Pandoc turns the Markdown into one self-contained HTML page with paper.css,
and Chromium prints that page to PDF.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(name="atco_paper"):
    build = HERE / ".build"
    build.mkdir(exist_ok=True)
    page = build / f"{name}.html"
    subprocess.run(
        ["pandoc", f"{name}.md", "--standalone", "--embed-resources", "--css", "paper.css",
         "--metadata", "lang=en-GB", "--output", str(page)],
        cwd=HERE, check=True,
    )
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        tab = browser.new_page()
        tab.goto(page.as_uri(), wait_until="load")
        tab.pdf(path=str(HERE / f"{name}.pdf"), format="A4", print_background=True,
                display_header_footer=True, header_template="<span></span>",
                footer_template='<div style="font-size:8pt;color:#5C6A75;width:100%;text-align:center">'
                                '<span class="pageNumber"></span></div>',
                margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"})
        browser.close()
    print(HERE / f"{name}.pdf")


if __name__ == "__main__":
    main(*sys.argv[1:2])
