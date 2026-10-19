"""Static server for the Moth smoke/demo page (Task 3.5).

Demo pages need ``SharedArrayBuffer`` for WASM threads: every response carries
``Cross-Origin-Opener-Policy: same-origin``, ``Cross-Origin-Embedder-Policy: require-corp`` and
``Cross-Origin-Resource-Policy: same-origin`` so the page is cross-origin isolated. Without them
``onnxruntime-web`` falls back to a single thread (``web/ort.js`` checks ``crossOriginIsolated``).

Usage: ``python3 server.py --root .. --port 8765`` (serve the repository root so that
``/web/…``, ``/node_modules/…`` and ``/release/…`` are all reachable).
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

__all__ = ["Handler", "main"]


class Handler(SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` with the cross-origin-isolation headers."""

    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".json": "application/json",
        ".wav": "audio/wav",
    }

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        pass  # quiet: the page and Playwright report their own errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="directory to serve (default: .)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    handler = partial(Handler, directory=str(root))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"serving {root} on http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
