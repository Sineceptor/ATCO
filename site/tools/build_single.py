#!/usr/bin/env python3
"""Build the single-file version of the site.

    python3 site/tools/build_single.py [-o atco-site.html]

Inlines the stylesheet, every local script, any JSON data file referenced with
<script type="application/json" src="...">, and every local image and audio file
as a base64 data URI. The result is one portable .html with no local dependencies.
Only the Google Fonts stylesheet still needs the network.
"""
import argparse, base64, mimetypes, pathlib, re, sys

SITE = pathlib.Path(__file__).resolve().parent.parent
mimetypes.add_type('audio/ogg', '.ogg')
mimetypes.add_type('audio/mp4', '.m4a')


def read(rel):
    f = SITE / rel
    if not f.exists():
        sys.exit(f'build_single: missing {rel}')
    return f


def data_uri(rel):
    f = read(rel)
    mime = mimetypes.guess_type(f.name)[0] or 'application/octet-stream'
    return f'data:{mime};base64,' + base64.b64encode(f.read_bytes()).decode()


def build():
    html = read('index.html').read_text(encoding='utf-8')

    # stylesheet -> <style>
    def css(m):
        return '<style>\n' + read(m.group(1)).read_text(encoding='utf-8') + '\n</style>'
    html = re.sub(r'<link rel="stylesheet" href="(?!https?:)([^"]+)">', css, html)

    # local scripts -> inline, in place, so load order is preserved
    def js(m):
        return '<script>\n' + read(m.group(1)).read_text(encoding='utf-8') + '\n</script>'
    html = re.sub(r'<script src="(?!https?:)([^"]+)"[^>]*></script>', js, html)

    # JSON data -> inline <script type="application/json">
    def json_block(m):
        ident, path = m.group(1), m.group(2)
        return (f'<script type="application/json" id="{ident}">\n'
                + read(path).read_text(encoding='utf-8') + '\n</script>')
    html = re.sub(
        r'<script type="application/json" id="([^"]+)" src="(?!https?:)([^"]+)"></script>',
        json_block, html)

    # local images and audio -> data URIs
    def asset(m):
        attr, path = m.group(1), m.group(2)
        return f'{attr}="{data_uri(path)}"'
    html = re.sub(r'(?<![-\w])(src)="(?!https?:|data:)((?:images|img|audio)/[^"]+)"', asset, html)
    html = re.sub(r'(?<![-\w])(srcset)="(?!https?:|data:)((?:images|img)/[^"]+)"', asset, html)
    # a link to a local image must also survive in the standalone export
    html = re.sub(r'(?<![-\w])(href)="(?!https?:|data:|#)((?:images|img)/[^"]+)"', asset, html)

    left = re.findall(r'(?<![-\w])(?:src|href)="(?!https?:|data:|#)([^"]+)"', html)
    if left:
        print('build_single: note, still referenced locally:', sorted(set(left)))
    return html


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=str(SITE.parent / 'atco-site.html'))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.write_text(build(), encoding='utf-8')
    print(f'{out}  {out.stat().st_size // 1024} KB')
