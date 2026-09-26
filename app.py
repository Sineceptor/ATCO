"""Local browser interface for ATCO."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit
import uuid

from atco import pipeline

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 16 * 1024 * 1024


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, settings):
        super().__init__(address, Handler)
        self.settings = settings
        self.work_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type="application/json"):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; media-src 'self' blob:; "
            "connect-src 'self'; script-src 'self'; style-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        static = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/recorder.js": ("recorder.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if path in static:
            file, kind = static[path]
            return self.reply(200, (ROOT / "atco/web" / file).read_bytes(), kind)
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "runs":
            run_id, name = parts[1:]
            if len(run_id) == 32 and all(c in "0123456789abcdef" for c in run_id):
                kinds = {
                    "readback.wav": "audio/wav",
                    "transcript.txt": "text/plain; charset=utf-8",
                    "result.json": "application/json",
                }
                file = self.server.settings.output_dir / run_id / name
                if name in kinds and file.is_file():
                    return self.reply(200, file.read_bytes(), kinds[name])
        self.reply(404, {"error": "File not found"})

    def do_POST(self):
        # The interface is local; reject requests initiated by unrelated web pages.
        port = self.server.server_port
        origin = self.headers.get("Origin")
        if origin and origin not in {
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
        }:
            return self.reply(
                403, {"error": "Open the local ATCO page to submit audio"}
            )
        path = urlsplit(self.path).path
        if path not in {"/api/transcribe", "/api/text"}:
            return self.reply(404, {"error": "Unknown request"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.reply(400, {"error": "Invalid upload length"})
        if not 0 < length <= MAX_UPLOAD:
            return self.reply(
                413, {"error": "Use a recording of at most 30 seconds (16 MB maximum)"}
            )
        expected = "audio/wav" if path == "/api/transcribe" else "application/json"
        if self.headers.get_content_type() != expected:
            return self.reply(415, {"error": f"Expected {expected}"})
        if not self.server.work_lock.acquire(blocking=False):
            return self.reply(
                409,
                {
                    "error": "A recording is still being processed. Try again when it finishes."
                },
            )
        try:
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("The upload was incomplete")
            run_id = uuid.uuid4().hex
            output = self.server.settings.output_dir / run_id
            output.mkdir(parents=True)
            audio, text = None, None
            if path == "/api/transcribe":
                import io
                import soundfile as sf
                import numpy as np

                with sf.SoundFile(io.BytesIO(body)) as source:
                    if (
                        source.format != "WAV"
                        or not 0 < source.frames / source.samplerate <= 30
                    ):
                        raise ValueError(
                            "Use a non-empty WAV recording of at most 30 seconds"
                        )
                    samples = source.read(dtype="float32")
                    if not np.isfinite(samples).all():
                        raise ValueError("The recording contains invalid audio samples")
                    if not np.any(samples):
                        raise ValueError(
                            "The recording is silent. Check your microphone and try again."
                        )
                audio = output / "input.wav"
                audio.write_bytes(body)
            else:
                payload = json.loads(body)
                text = payload.get("text") if isinstance(payload, dict) else None
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Enter an instruction first")
                if len(text) > 4000:
                    raise ValueError("Use one short instruction")
            args = SimpleNamespace(**vars(self.server.settings))
            args.audio, args.text, args.output_dir = audio, text, output
            args.tts = (
                "speecht5" if self.headers.get("X-ATCO-Speech") == "yes" else "none"
            )
            try:
                result = pipeline.run(args)
            except Exception:
                # Preserve successful transcription/extraction if only the optional speech stage failed.
                partial = output / "result.json"
                result = json.loads(partial.read_text()) if partial.exists() else None
                if not result or result.get("tts_status") != "failed":
                    raise
            (output / "transcript.txt").write_text(
                result["transcript"] + "\n", encoding="utf-8"
            )
            result["transcript_url"] = f"/runs/{run_id}/transcript.txt"
            result["details_url"] = f"/runs/{run_id}/result.json"
            result["audio_url"] = (
                f"/runs/{run_id}/readback.wav" if result["audio_file"] else None
            )
            self.reply(200, result)
        except ImportError as exc:
            # A missing package is a problem with this computer, not with the upload.
            print(f"Missing dependency: {exc}", flush=True)
            self.reply(
                500,
                {"error": f"A required package is missing ({exc.name}). Run: pip install -r requirements.txt"},
            )
        except (ValueError, OSError, RuntimeError) as exc:
            self.reply(400, {"error": str(exc)})
        except Exception as exc:
            print(f"Request failed: {type(exc).__name__}: {exc}", flush=True)
            self.reply(
                500, {"error": "Processing failed. See the terminal for the error."}
            )
        finally:
            self.server.work_lock.release()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--models-dir", type=Path, default=ROOT / "models")
    cli.add_argument("--asr-model")
    cli.add_argument("--ner-model")
    cli.add_argument("--tts-model")
    cli.add_argument("--vocoder-model")
    cli.add_argument("--port", type=int, default=8765)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "outputs/browser")
    args = cli.parse_args()
    args.allow_downloads = False
    args.output_dir = args.output_dir.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    import torch

    torch.set_num_threads(4)
    server = AppServer(("127.0.0.1", args.port), args)
    print(
        f"Open http://127.0.0.1:{server.server_port} — press Ctrl+C to stop.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
