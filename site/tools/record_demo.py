"""Record site/demo/atco-demo.mp4: the real ATCO app in a browser, with captions.

    python app.py --port 8791            # in another terminal
    python site/tools/record_demo.py     # needs Playwright's Chromium and phrase.wav beside it
    ffmpeg -i raw/*.webm -c:v libx264 -pix_fmt yuv420p -crf 26 -movflags +faststart atco-demo.mp4

phrase.wav is site/audio/phrase.m4a converted to 16 kHz mono WAV. The video is
not edited or cut; marks.json records when each wait started and ended.
"""
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8791/"
CAPTION_JS = """(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;padding:14px 22px;' +
      'background:rgba(14,22,34,0.94);color:#fff;font:600 18px/1.4 -apple-system,Helvetica,Arial,sans-serif;';
    document.body.appendChild(el);
  }
  el.textContent = text;
}"""


def main():
    marks = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 800},
                                      record_video_dir=str(HERE / "raw"),
                                      record_video_size={"width": 1280, "height": 800})
        start = time.time()
        mark = lambda name: marks.__setitem__(name, round(time.time() - start, 2))
        page = context.new_page()
        say = lambda text, hold: (page.evaluate(CAPTION_JS, text), page.wait_for_timeout(int(hold * 1000)))
        page.goto(URL)
        say("ATCO, running on my laptop. Speech model: my best 2026 model (the 12-epoch Whisper-medium "
            "soup, models/whisper-medium-2026). An illustration, not the benchmark.", 5)
        page.set_input_files("#file", str(HERE / "phrase.wav"))
        page.evaluate("() => { const a = document.getElementById('preview'); a.muted = true; a.play(); }")
        say("Input: a sentence from my own generator, read by the project's SpeechT5 voice (not ATCO2 audio): "
            "“Turn right, heading one seven two, Speedbird one eight zero four.”", 6)
        page.click("#transcribe")
        mark("wait_start")
        say("Transcribing, tagging the words, checking the numbers and making a spoken readback. "
            "(The models load for each request, so this takes a few seconds.)", 1)
        page.wait_for_selector("#result:not([hidden])", timeout=300_000)
        page.wait_for_selector("#readback-section:not([hidden])", timeout=300_000)
        mark("wait_end")
        page.locator("#transcript").scroll_into_view_if_needed()
        say("The transcript is word for word right.", 4)
        page.locator("#entities").scroll_into_view_if_needed()
        say("The tagger labels “turn” and “heading” as commands and “one seven two” "
            "as the heading; the range check finds heading 172 within 0 to 360, so no warning.", 6)
        say("A mistake I can explain: it calls “Speedbird” (British Airways’ callsign) a waypoint and "
            "its flight number a value. My word lists don’t know this airline, and the tagger learnt from them.", 8)
        page.locator("#readback-section").scroll_into_view_if_needed()
        say("The readback is spoken from the extracted words, in order, by SpeechT5.", 4)
        page.click("details > summary")
        page.wait_for_timeout(600)
        page.fill("#typed", "jetstar 1 squawk 4582")
        page.locator("#typed").scroll_into_view_if_needed()
        say("Second example, typed text: “jetstar 1 squawk 4582”.", 3)
        page.click("#parse")
        mark("wait2_start")
        say("Extracting from the text.", 1)
        page.wait_for_selector("#warnings:not([hidden])", timeout=300_000)
        mark("wait2_end")
        page.locator("#warnings").scroll_into_view_if_needed()
        say("Range check: transponder codes use only digits 0 to 7, so squawk 4582 is flagged.", 6)
        say("Measured results, on 74 recordings no model trained on, are on the project page: "
            "sineceptor.github.io/ATCO", 5)
        video = page.video.path()
        context.close()
        browser.close()
    marks["video"] = str(video)
    (HERE / "marks.json").write_text(json.dumps(marks, indent=2))
    print(json.dumps(marks, indent=2))


if __name__ == "__main__":
    main()
