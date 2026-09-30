"""HTTP boundary checks; model behavior is covered by the real-model recheck."""

import http.client
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import app


def recording(seconds=0.1, silent=False):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(
            struct.pack("<h", 0 if silent else 1000) * int(16000 * seconds)
        )
    return output.getvalue()


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        settings = SimpleNamespace(
            output_dir=Path(self.temp.name),
            models_dir=Path("models"),
            asr_model=None,
            ner_model=None,
            tts_model=None,
            vocoder_model=None,
            allow_downloads=False,
        )
        self.server = app.AppServer(("127.0.0.1", 0), settings)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, body=None, content_type="audio/wav", **headers):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        connection.request(
            "POST" if body is not None else "GET",
            path,
            body,
            {"Content-Type": content_type, **headers},
        )
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_audio_bytes_reach_connector_and_text_can_be_downloaded(self):
        wav = recording()

        def connected(args):
            self.assertEqual(args.audio.read_bytes(), wav)
            self.assertEqual(args.tts, "none")
            result = app.pipeline.make_result("jetstar one", [], "", [])
            result["tts_status"] = "skipped"
            (args.output_dir / "result.json").write_text(json.dumps(result))
            return result

        with patch("app.pipeline.run", side_effect=connected) as run:
            status, body = self.request("/api/transcribe", wav)
        self.assertEqual(status, 200)
        self.assertEqual(run.call_count, 1)
        result = json.loads(body)
        self.assertEqual(
            self.request(result["transcript_url"]), (200, b"jetstar one\n")
        )

    def test_extensible_wav_is_accepted(self):
        # Many tools (macOS afconvert among them) write the "extensible" WAV variant.
        import io

        import numpy as np
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, 0.1 * np.sin(np.arange(16000) / 5), 16000, format="WAVEX", subtype="PCM_16")

        def connected(args):
            result = app.pipeline.make_result("jetstar one", [], "", [])
            result["tts_status"] = "skipped"
            (args.output_dir / "result.json").write_text(json.dumps(result))
            return result

        with patch("app.pipeline.run", side_effect=connected):
            status, _ = self.request("/api/transcribe", buffer.getvalue())
        self.assertEqual(status, 200)

    def test_invalid_audio_is_rejected_before_model_loading(self):
        with patch("app.pipeline.run") as run:
            for body in [b"not audio", recording(31), recording(silent=True)]:
                with self.subTest(size=len(body)):
                    status, result = self.request("/api/transcribe", body)
                    self.assertEqual(status, 400)
                    self.assertIn("error", json.loads(result))
        run.assert_not_called()

    def test_empty_text_and_external_origin_are_rejected(self):
        with patch("app.pipeline.run") as run:
            self.assertEqual(
                self.request("/api/text", b'{"text":""}', "application/json")[0], 400
            )
            self.assertEqual(
                self.request(
                    "/api/transcribe", recording(), Origin="https://example.com"
                )[0],
                403,
            )
        run.assert_not_called()

    def test_failed_speech_still_returns_successful_text(self):
        def broken_speech(args):
            result = app.pipeline.make_result("jetstar one", [], "", [])
            result.update(tts_status="failed", tts_error="Test synthesis failure")
            (args.output_dir / "result.json").write_text(json.dumps(result))
            raise RuntimeError("Test synthesis failure")

        with patch("app.pipeline.run", side_effect=broken_speech):
            status, body = self.request(
                "/api/text",
                b'{"text":"jetstar one"}',
                "application/json",
                **{"X-ATCO-Speech": "yes"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["transcript"], "jetstar one")
        self.assertEqual(json.loads(body)["tts_status"], "failed")

    def test_concurrent_jobs_and_unlisted_files_are_rejected(self):
        self.server.work_lock.acquire()
        try:
            self.assertEqual(self.request("/api/transcribe", recording())[0], 409)
        finally:
            self.server.work_lock.release()
        for path in [
            "/../README.md",
            "/runs/" + "a" * 32 + "/input.wav",
            "/api/unknown",
        ]:
            self.assertEqual(self.request(path)[0], 404)


if __name__ == "__main__":
    unittest.main()
