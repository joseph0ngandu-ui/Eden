from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable


@dataclass
class TradingViewWebhookServer:
    host: str = "127.0.0.1"
    port: int = 9000
    on_signal: Callable[[dict], None] | None = None

    def start(self):
        log = logging.getLogger("eden.execution.tradingview_webhook")
        handler_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length).decode('utf-8')
                try:
                    data = json.loads(body)
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                if handler_ref.on_signal:
                    handler_ref.on_signal(data)
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):  # silence default
                return

        server = HTTPServer((self.host, self.port), Handler)
        log.info("TradingView webhook server listening on %s:%d", self.host, self.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
