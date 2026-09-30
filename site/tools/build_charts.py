#!/usr/bin/env python3
"""Draw the page's charts from results/retraining.json.

    python3 site/tools/build_charts.py

Each chart is written into index.html between <!-- chart:NAME --> and
<!-- /chart:NAME --> markers, as plain SVG that needs no JavaScript. Numbers
are never typed into the page by hand: rerun this after results/ changes.
"""
import html
import json
import pathlib
import re

SITE = pathlib.Path(__file__).resolve().parent.parent
RESULTS = json.loads((SITE.parent / 'results' / 'retraining.json').read_text(encoding='utf-8'))

LABELS = {
    'zero_shot': 'Whisper-small, not fine-tuned',
    'zero_shot_medium': 'Whisper-medium, not fine-tuned',
    'checkpoint_2025': 'My 2025 model',
    'fixed': 'Retrained with the fixes',
    'synthetic': '+ my synthetic radio clips',
    'augmented': '+ noise and speed copies',
    'continued': '2025 model, trained on with the fix',
    'real_data': '+ 10.5 hours of real ATC speech',
    'soup_fixed_real': 'Soup: fixes-only + real speech',
    'medium_real': 'Whisper-medium + real speech',
    'soup_medium': 'Soup of two Whisper-medium models',
    'soup_medium_long': 'Same soup, Whisper-medium trained 12 epochs',
    'medium_longer': 'Whisper-medium, trained longer',
}


def pct(x):
    return f'{100 * x:.2f}'  # two decimals, as in results/: nothing is rounded for the page


def bars():
    """Horizontal bars with 95% ranges, as HTML so the rows reflow on a phone."""
    rows = [(name, run['headline_test']) for name, run in RESULTS['runs'].items()]
    chosen = RESULTS['chosen_on_validation']
    top = 70  # the axis runs from 0 to 70%
    w = lambda v: f'{100 * min(v, top / 100) / (top / 100):.2f}%'
    summary = '; '.join(f"{LABELS[n]}: {pct(r['wer'])}%, 95% range {pct(r['interval_95'][0])} to "
                        f"{pct(r['interval_95'][1])}" for n, r in rows)
    out = [f'<div class="werbars" role="img" aria-label="Word error rate on the 74 unseen-recording clips. '
           f'{html.escape(summary)}">']
    for name, r in rows:
        kind = 'best' if name == chosen else 'old' if name == 'checkpoint_2025' else 'base' if name == 'zero_shot' else ''
        lo, hi = r['interval_95']
        cls = f'wrow {kind}'.strip()
        out.append(f'<div class="{cls}"><span class="wl">{html.escape(LABELS[name])}</span>'
                   f'<span class="wt"><i style="width:{w(r["wer"])}"></i>'
                   f'<b style="left:{w(lo)};width:calc({w(hi)} - {w(lo)})"></b></span>'
                   f'<span class="wv">{pct(r["wer"])}%</span></div>')
    out.append('<div class="wrow waxis" aria-hidden="true"><span></span><span class="wticks">'
               + ''.join(f'<span style="left:{100 * t / top:.2f}%">{t}%</span>' for t in range(0, top + 1, 10))
               + '</span><span></span></div>')
    out.append('</div>')
    return '\n'.join(out)


def noise():
    sweep = RESULTS['noise_sweep']['results']
    series = [('zero_shot', 'Not fine-tuned', 'line base'), ('checkpoint_2025', 'My 2025 model', 'line old'),
              ('retrained', 'Best 2026 model', 'line best')]
    points = [p['added_noise_snr_db'] for p in sweep['checkpoint_2025']]
    left, top, width, height = 44, 12, 330, 200
    xs = {p: left + width * i / (len(points) - 1) for i, p in enumerate(points)}
    ymax = 1.0
    y = lambda v: top + height * (1 - min(v, ymax) / ymax)
    out = [f'<svg class="chart noise" viewBox="0 0 {left + width + 142} {top + height + 46}" role="img" '
           'aria-labelledby="noise-title noise-desc">',
           '<title id="noise-title">Word error rate as noise is added</title>',
           '<desc id="noise-desc">' + html.escape('; '.join(
               f"{label}: " + ', '.join(
                   f"{'as recorded' if p['added_noise_snr_db'] is None else str(int(p['added_noise_snr_db'])) + ' dB'} {pct(p['wer'])}%"
                   for p in sweep[key]) for key, label, _ in series if key in sweep)) + '</desc>']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{left}" x2="{left + width}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
                   f'<text class="tick" x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{int(tick * 100)}%</text>')
    for p, x in xs.items():
        label = 'as recorded' if p is None else f'{int(p)} dB'
        out.append(f'<text class="tick" x="{x:.1f}" y="{top + height + 20}" text-anchor="middle">{label}</text>')
    out.append(f'<text class="axis" x="{left + width / 2}" y="{top + height + 42}" text-anchor="middle">'
               'signal-to-noise ratio after adding noise (less is noisier)</text>')
    present = [s for s in series if s[0] in sweep]
    # End labels sit beside the last point; push apart any that would overlap.
    ends = sorted(((y(sweep[key][-1]['wer']), key) for key, _, _ in present))
    label_y, gap = {}, 17
    for i, (yy, key) in enumerate(ends):
        label_y[key] = yy if i == 0 else max(yy, label_y[ends[i - 1][1]] + gap)
    for key, label, cls in present:
        pts = ' '.join(f'{xs[p["added_noise_snr_db"]]:.1f},{y(p["wer"]):.1f}' for p in sweep[key])
        out.append(f'<polyline class="{cls}" points="{pts}"/>')
        for p in sweep[key]:
            out.append(f'<circle class="dot {cls.split()[1]}" cx="{xs[p["added_noise_snr_db"]]:.1f}" cy="{y(p["wer"]):.1f}" r="3.5"/>')
        out.append(f'<text class="serieslabel {cls.split()[1]}" x="{left + width + 10}" y="{label_y[key] + 5:.1f}">{label}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def main():
    page = (SITE / 'index.html').read_text(encoding='utf-8')
    charts = {'bars': bars()}
    if 'noise_sweep' in RESULTS:
        charts['noise'] = noise()
    for name, svg in charts.items():
        pattern = re.compile(rf'(<!-- chart:{name} -->).*?(<!-- /chart:{name} -->)', re.S)
        if not pattern.search(page):
            print(f'no markers for chart "{name}" in index.html; skipped')
            continue
        page = pattern.sub(lambda m: f'{m.group(1)}\n{svg}\n{m.group(2)}', page)
        print(f'drew chart "{name}"')
    (SITE / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
