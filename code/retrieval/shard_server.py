"""One search shard process.

The process owns a single SQLite file and serves search from the inverted
index loaded out of that file. Bind it to localhost. This is not a secure
remote deployment.
"""

import argparse
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.inverted_index import GlobalBm25Stats
from utils.text_cleaning import overlap_ratio, tokenize
from retrieval.protocol import (
    DEFAULT_BODY_LIMIT,
    commit_request,
    documents_from_batch,
    dumps,
    hit_payload,
    parse_object,
    search_request,
    stats_terms,
    ProtocolError,
)
from retrieval.shard_client import require_loopback
from retrieval.shard_store import ShardStore, StoreError


class ShardState:
    def __init__(self, store: ShardStore, shard_id: str, body_limit: int):
        self.store = store
        self.shard_id = shard_id
        self.body_limit = body_limit
        self.lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        meta = self.store.metadata()
        index = self.store.load_index()
        with self.lock:
            self.meta = meta
            self.index = index

    def health(self) -> dict:
        with self.lock:
            meta = dict(self.meta)
        return {
            "shard_id": self.shard_id,
            "ready": bool(meta["ready"]),
            "generation": meta["generation"],
            "document_count": meta["document_count"],
        }


def make_handler(state: ShardState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            logging.getLogger("shard").info("%s %s", state.shard_id, fmt % args)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
            if self.path == "/health":
                self._send(200, state.health())
                return
            if self.path == "/v1/ids":
                health = state.health()
                if not health["ready"]:
                    self._send(409, {"error": "shard is not ready"})
                    return
                self._send(200, {"ids": state.store.document_ids()})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
            try:
                payload = self._read_body()
                if self.path == "/v1/stats":
                    self._send(200, self._stats(payload))
                elif self.path == "/v1/search":
                    self._send(200, self._search(payload))
                elif self.path == "/v1/index/begin":
                    state.store.clear_staging()
                    self._send(200, {"cleared": True})
                elif self.path == "/v1/index/batch":
                    count = state.store.stage_documents(documents_from_batch(payload))
                    self._send(200, {"staged": count})
                elif self.path == "/v1/index/commit":
                    self._send(200, self._commit(payload))
                else:
                    self._send(404, {"error": "not found"})
            except ProtocolError as exc:
                self._send(exc.status, {"error": exc.message})
            except StoreError as exc:
                self._send(409, {"error": str(exc)})
            except Exception as exc:  # bounded response; the process stays up
                logging.getLogger("shard").exception("shard request failed")
                self._send(500, {"error": str(exc)})

        def _read_body(self) -> dict:
            length_header = self.headers.get("Content-Length", "0")
            try:
                length = int(length_header)
            except ValueError as exc:
                raise ProtocolError(400, "Content-Length must be an integer") from exc
            if length < 0 or length > state.body_limit:
                raise ProtocolError(413, "request body is too large")
            raw = self.rfile.read(length) if length else b""
            return parse_object(raw, state.body_limit)

        def _stats(self, payload: dict) -> dict:
            health = state.health()
            if not health["ready"]:
                raise ProtocolError(409, "shard is not ready")
            with state.lock:
                index = state.index
                generation = state.meta["generation"]
            terms = stats_terms(payload)
            return {
                "shard_id": state.shard_id,
                "generation": generation,
                "n_docs": index.document_count,
                "total_tokens": index.total_tokens,
                "df": {term: int(index.doc_freq.get(term, 0)) for term in terms},
            }

        def _search(self, payload: dict) -> dict:
            request = search_request(payload)
            with state.lock:
                ready = bool(state.meta["ready"])
                generation = state.meta["generation"]
                index = state.index
            if not ready or request["generation"] != generation:
                raise ProtocolError(409, "index generation does not match")
            stats = GlobalBm25Stats(
                n_docs=request["n_docs"],
                total_tokens=request["total_tokens"],
                df=request["df"],
            )
            hits = index.search(
                request["query"],
                company=request["company"],
                top_k=request["top_k"],
                min_overlap=request["min_overlap"],
                stats=stats,
            )
            query_tokens = tokenize(request["query"])
            return {
                "shard_id": state.shard_id,
                "generation": generation,
                "hits": [
                    hit_payload(
                        doc,
                        score,
                        overlap_ratio(query_tokens, doc.term_freq.keys()),
                    )
                    for doc, score in hits
                ],
            }

        def _commit(self, payload: dict) -> dict:
            generation, expected = commit_request(payload)
            state.store.commit_staged(generation, expected)
            state.reload()
            return state.health()

        def _send(self, status: int, payload: dict) -> None:
            body = dumps(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class ShardHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(shard_id: str, host: str, port: int, db_path: Path, body_limit: int = DEFAULT_BODY_LIMIT) -> None:
    host = require_loopback(host)
    store = ShardStore(db_path)
    state = ShardState(store, shard_id, body_limit)
    server = ShardHTTPServer((host, port), make_handler(state))
    logging.getLogger("shard").info("listening shard=%s addr=%s:%s db=%s", shard_id, host, port, db_path)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        store.close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one local search shard")
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--body-limit", type=int, default=DEFAULT_BODY_LIMIT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    serve(args.shard_id, args.host, args.port, Path(args.db), args.body_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
