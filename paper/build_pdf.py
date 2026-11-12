"""Typeset the paper with LaTeX, or print the standalone one-page summary.

    python paper/build_figures.py
    python paper/build_pdf.py               # the paper
    python paper/build_pdf.py atco_summary  # the one-page summary

The full paper needs pandoc and tectonic on PATH. Its Markdown is the source
for the text, captions and references. Figures use vector PDFs when available.
The summary retains its existing HTML/CSS layout and needs Playwright Chromium.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def strip_caption_number(caption, kind):
    """Let LaTeX number captions without duplicating the Markdown labels."""
    blocks = caption[1]
    if blocks and blocks[0]["t"] in {"Plain", "Para"}:
        inlines = blocks[0]["c"]
        if (len(inlines) >= 4 and inlines[0] == {"t": "Str", "c": kind}
                and inlines[2]["t"] == "Str" and re.fullmatch(r"\d+\.", inlines[2]["c"])):
            blocks[0]["c"] = inlines[4:]


def typeset_table(table, api_version):
    """Keep these short results tables whole, with wrapping aligned columns."""
    _, caption, columns, head, bodies, foot = table["c"]
    strip_caption_number(caption, "Table")
    head_rows = head[1]
    rows = [row for body in bodies for row in body[2] + body[3]] + foot[1]
    entries = [caption[1]] + [cell[4] for row in head_rows + rows for cell in row[1]]
    pieces = []
    for entry in entries:
        pieces.extend(entry)
        pieces.append({"t": "RawBlock", "c": ["latex", "% ATCO-CELL-END"]})
    rendered = subprocess.check_output(
        ["pandoc", "--from", "json", "--to", "latex", "--wrap=none"],
        input=json.dumps({"pandoc-api-version": api_version, "meta": {}, "blocks": pieces}),
        text=True,
    ).split("% ATCO-CELL-END")
    parts = iter(part.strip() for part in rendered[:-1])
    title = next(parts)
    specification = []
    for alignment, width in columns:
        fraction = width.get("c", 1 / len(columns))
        align = "raggedleft" if alignment["t"] == "AlignRight" else "raggedright"
        specification.append(">{\\" + align + r"\arraybackslash}p{\dimexpr "
                             + f"{fraction:.6f}" + r"\linewidth-2\tabcolsep\relax}")
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}",
             r"\caption{" + title + "}", r"\begin{tabular}{" + "".join(specification) + "}", r"\toprule"]
    for i, row in enumerate(head_rows + rows):
        lines.append(" & ".join(next(parts) for _ in row[1]) + r" \\")
        if i + 1 == len(head_rows):
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return {"t": "RawBlock", "c": ["latex", "\n".join(lines)]}


def prepare_paper(document):
    blocks = document["blocks"]
    if blocks[0]["t"] != "Header" or blocks[0]["c"][1][0] != "abstract":
        raise ValueError("Expected the paper to begin with its abstract")
    document["meta"]["abstract"] = {"t": "MetaBlocks", "c": [blocks[1]]}
    blocks = blocks[2:]
    prepared, references = [], False
    for block in blocks:
        if block["t"] == "Header":
            block["c"][0] -= 1
            if block["c"][1][0] == "references":
                references = True
                prepared.append({"t": "RawBlock", "c": ["latex", "\\clearpage\n\\begin{thebibliography}{12}"]})
                continue
        if references:
            if block["t"] != "Para" or not re.fullmatch(r"\[\d+\]", block["c"][0].get("c", "")):
                raise ValueError("References must be numbered paragraphs at the end of the paper")
            number = block["c"][0]["c"][1:-1]
            prepared.append({"t": "RawBlock", "c": ["latex", f"\\bibitem{{ref{number}}}"]})
            block["c"] = block["c"][2:]
        prepared.append(block)
    if not references:
        raise ValueError("No references found")
    prepared.append({"t": "RawBlock", "c": ["latex", "\\end{thebibliography}"]})
    document["blocks"] = prepared

    def walk(node):
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        kind = node.get("t")
        if kind == "Table":
            return typeset_table(node, document["pandoc-api-version"])
        if kind == "Str" and re.fullmatch(r"\[\d+\]", node["c"]):
            return {"t": "RawInline", "c": ["latex", "\\cite{ref" + node["c"][1:-1] + "}"]}
        if kind == "Figure":
            strip_caption_number(node["c"][1], kind)
        if kind == "Image":
            source = HERE / node["c"][2][0]
            vector = source.with_suffix(".pdf")
            node["c"][2][0] = str(vector if vector.exists() else source)
        return {key: walk(value) for key, value in node.items()}

    return walk(document)


def build_paper(build):
    source = subprocess.check_output(["pandoc", "atco_paper.md", "--to", "json"], cwd=HERE, text=True)
    document = prepare_paper(json.loads(source))
    tex = build / "atco_paper.tex"
    subprocess.run(
        ["pandoc", "--from", "json", "--to", "latex", "--standalone", "--pdf-engine=tectonic",
         "--include-in-header", "typesetting.tex", "--metadata", "lang=en-GB",
         "--variable", "fontsize=11pt", "--variable", "geometry=a4paper,margin=22mm,headsep=7mm",
         "--output", str(tex)],
        input=json.dumps(document), text=True, cwd=HERE, check=True,
    )
    subprocess.run(["tectonic", "--keep-logs", "--outdir", str(build), str(tex)], cwd=HERE, check=True)
    if "Missing character:" in (build / "atco_paper.log").read_text():
        raise RuntimeError("The PDF has missing glyphs; check the typesetting log before publishing")
    shutil.copyfile(build / "atco_paper.pdf", HERE / "atco_paper.pdf")


def main(name="atco_paper"):
    build = HERE / ".build"
    build.mkdir(exist_ok=True)
    if name == "atco_paper":
        build_paper(build)
        print(HERE / f"{name}.pdf")
        return
    if name != "atco_summary":
        raise ValueError("Choose atco_paper or atco_summary")
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
