from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from taterbench.engines.llamacpp import LlamaCppSession
from taterbench.types import ModelCandidate, RunVariant


class _RunningProcess:
    returncode = None

    @staticmethod
    def poll():
        return None


class _StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        return None

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        events = [
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "Tater"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
                "timings": {"prompt_per_second": 100.0, "predicted_per_second": 25.0},
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))


class LlamaCppStreamTests(unittest.TestCase):
    def test_openai_stream_records_ttft_usage_and_speed(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                model_path = Path(temp) / "Model-Q4_K_M.gguf"
                model_path.write_bytes(b"model")
                model = ModelCandidate(
                    id="test",
                    label="test",
                    provider="llama_cpp",
                    model_path=model_path,
                    filename=model_path.name,
                )
                session = LlamaCppSession(model, RunVariant("baseline"))
                session.process = _RunningProcess()  # type: ignore[assignment]
                session.base_url = f"http://127.0.0.1:{server.server_port}"
                result = session.generate(
                    [{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    temperature=0.0,
                    seed=42,
                )
            self.assertEqual(result.text, "Hello Tater")
            self.assertEqual(result.prompt_tokens, 12)
            self.assertEqual(result.completion_tokens, 2)
            self.assertEqual(result.completion_tokens_per_second, 25.0)
            self.assertGreaterEqual(result.ttft_seconds, 0.0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
