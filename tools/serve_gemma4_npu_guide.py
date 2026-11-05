#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serve only the generated Gemma4 documentation and its source/print views."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    docs = Path(__file__).resolve().parents[1] / "docs"
    site = docs / "gemma4-moe-npu-guide"
    if not (site / "index.html").is_file():
        parser.error("Generate the site with tools/render_gemma4_npu_guide.py first")
    routes = {"/" + p.relative_to(site).as_posix(): p
              for p in site.rglob("*")
              if p.is_file() and p.suffix in (".html", ".css", ".js", ".png")}
    # Markdown and full-page HTML keep paths relative to docs/.
    for directory, suffixes in (("screenshots", (".png",)),
                                ("benchmarks", (".json", ".csv", ".zip"))):
        for asset in (site / "assets" / directory).glob("*"):
            if asset.is_file() and asset.suffix in suffixes:
                routes["/" + asset.relative_to(site).as_posix()] = asset
                routes["/" + asset.relative_to(docs).as_posix()] = asset
    routes["/"] = site / "index.html"
    routes["/ko/"] = site / "ko/index.html"
    for language in ("", ".ko"):
        for suffix in (".html", ".md"):
            filename = "gemma4-moe-npu-guide" + language + suffix
            routes["/" + filename] = docs / filename

    class Handler(SimpleHTTPRequestHandler):
        def route(self):
            return unquote(urlsplit(self.path).path)

        def translate_path(self, path):
            return str(routes[self.route()])

        def send_head(self):
            if self.route() not in routes:
                self.send_error(404, "Documentation page not found")
                return None
            return super().send_head()

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

    with ThreadingHTTPServer((args.bind, args.port), Handler) as server:
        print(f"Gemma4 documentation: http://localhost:{args.port}/", flush=True)
        print(f"Listening on {args.bind}:{args.port}; Ctrl+C stops the server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
