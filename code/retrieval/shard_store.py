"""SQLite snapshot for one search shard.

Queries are served from an in-memory ``InvertedIndex`` loaded from this file.
The database is the restart copy, not a per-term query engine. Document rows
keep ``term_freq``; postings are rebuilt in memory and are not stored.
"""

import json
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from data_loader.corpus_loader import Document
from retrieval.inverted_index import InvertedIndex
from retrieval.partitioner import SCHEMA_VERSION


class StoreError(RuntimeError):
    """A shard write was rejected and rolled back."""


_DOCUMENT_COLUMNS = """
    doc_id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    product_area TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    tokens_json TEXT NOT NULL,
    term_freq_json TEXT NOT NULL
"""

_POSTING_COLUMNS = """
    term TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    tf INTEGER NOT NULL,
    PRIMARY KEY (term, doc_id)
"""


class ShardStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def metadata(self) -> Dict[str, object]:
        with self._lock:
            return self._metadata_unlocked()

    def upsert_documents(self, docs: Sequence[Document]) -> None:
        """Insert or replace live documents without activating a generation."""

        def work() -> None:
            for doc in docs:
                self._delete_doc("documents", "postings", doc.doc_id)
                self._insert_doc("documents", "postings", doc)
            self._write_counts("documents")
            self._set_meta("ready", "0")

        self._transaction(work)

    def delete_document(self, doc_id: str) -> None:
        def work() -> None:
            self._delete_doc("documents", "postings", doc_id)
            self._write_counts("documents")
            self._set_meta("ready", "0")

        self._transaction(work)

    def replace_partition(self, docs: Sequence[Document], generation: str) -> None:
        """Replace the live partition and activate ``generation`` atomically."""

        def work() -> None:
            self._conn.execute("DELETE FROM postings")
            self._conn.execute("DELETE FROM documents")
            self._conn.execute("DELETE FROM postings_staging")
            self._conn.execute("DELETE FROM documents_staging")
            seen = set()
            for doc in docs:
                if doc.doc_id in seen:
                    raise StoreError(f"duplicate document id {doc.doc_id}")
                seen.add(doc.doc_id)
                self._insert_doc("documents", "postings", doc)
            count = self._count("documents")
            if count != len(docs):
                raise StoreError("partition count does not match the documents written")
            self._activate(generation, "documents")

        self._transaction(work)

    def clear_staging(self) -> None:
        def work() -> None:
            self._conn.execute("DELETE FROM postings_staging")
            self._conn.execute("DELETE FROM documents_staging")

        self._transaction(work)

    def stage_documents(self, docs: Sequence[Document]) -> int:
        def work() -> int:
            for doc in docs:
                self._delete_doc("documents_staging", "postings_staging", doc.doc_id)
                self._insert_doc("documents_staging", "postings_staging", doc)
            return self._count("documents_staging")

        return self._transaction(work)

    def commit_staged(self, generation: str, expected_count: int) -> Dict[str, object]:
        """Publish the staged partition if its count matches ``expected_count``."""

        def work() -> Dict[str, object]:
            count = self._count("documents_staging")
            if count != expected_count:
                raise StoreError(
                    f"staged document count {count} != expected {expected_count}"
                )
            self._conn.execute("DELETE FROM postings")
            self._conn.execute("DELETE FROM documents")
            self._conn.execute(
                """
                INSERT INTO documents (
                    doc_id, company, product_area, text, source,
                    token_count, tokens_json, term_freq_json
                )
                SELECT doc_id, company, product_area, text, source,
                       token_count, tokens_json, term_freq_json
                FROM documents_staging
                """
            )
            self._conn.execute("DELETE FROM postings_staging")
            self._conn.execute("DELETE FROM documents_staging")
            self._activate(generation, "documents")
            return self._metadata_unlocked()

        return self._transaction(work)

    def load_index(self) -> InvertedIndex:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT doc_id, company, product_area, text, source,
                       tokens_json, term_freq_json
                FROM documents
                ORDER BY doc_id
                """
            ).fetchall()
        docs: List[Document] = []
        for row in rows:
            tokens = [str(token) for token in json.loads(row["tokens_json"])]
            frequencies = {
                str(term): int(value)
                for term, value in json.loads(row["term_freq_json"]).items()
            }
            docs.append(
                Document(
                    doc_id=row["doc_id"],
                    company=row["company"],
                    product_area=row["product_area"],
                    text=row["text"],
                    tokens=tokens,
                    term_freq=Counter(frequencies),
                    path=Path(row["source"]),
                )
            )
        return InvertedIndex(docs)

    def document_ids(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id FROM documents ORDER BY doc_id"
            ).fetchall()
        return [row["doc_id"] for row in rows]

    def _migrate(self) -> None:
        self._conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS documents ({_DOCUMENT_COLUMNS});
            CREATE TABLE IF NOT EXISTS documents_staging ({_DOCUMENT_COLUMNS});
            CREATE TABLE IF NOT EXISTS postings ({_POSTING_COLUMNS});
            CREATE TABLE IF NOT EXISTS postings_staging ({_POSTING_COLUMNS});
            CREATE INDEX IF NOT EXISTS postings_term ON postings(term);
            CREATE INDEX IF NOT EXISTS postings_staging_term ON postings_staging(term);
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO index_metadata(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        for key, value in (
            ("generation", ""),
            ("document_count", "0"),
            ("total_tokens", "0"),
            ("ready", "0"),
        ):
            self._conn.execute(
                "INSERT OR IGNORE INTO index_metadata(key, value) VALUES(?, ?)",
                (key, value),
            )
        self._conn.commit()

    def _transaction(self, fn):
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                result = fn()
                self._conn.execute("COMMIT")
                return result
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _metadata_unlocked(self) -> Dict[str, object]:
        rows = {
            row["key"]: row["value"]
            for row in self._conn.execute("SELECT key, value FROM index_metadata")
        }
        return {
            "schema_version": int(rows.get("schema_version", str(SCHEMA_VERSION))),
            "generation": rows.get("generation", ""),
            "document_count": int(rows.get("document_count", "0")),
            "total_tokens": int(rows.get("total_tokens", "0")),
            "ready": rows.get("ready", "0") == "1",
        }

    def _insert_doc(self, documents: str, postings: str, doc: Document) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {documents} (
                doc_id, company, product_area, text, source,
                token_count, tokens_json, term_freq_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.doc_id,
                doc.company,
                doc.product_area,
                doc.text,
                str(doc.path),
                len(doc.tokens),
                json.dumps(list(doc.tokens), ensure_ascii=False),
                json.dumps({term: int(tf) for term, tf in doc.term_freq.items()}, ensure_ascii=False),
            ),
        )

    def _delete_doc(self, documents: str, postings: str, doc_id: str) -> None:
        self._conn.execute(f"DELETE FROM {postings} WHERE doc_id = ?", (doc_id,))
        self._conn.execute(f"DELETE FROM {documents} WHERE doc_id = ?", (doc_id,))

    def _count(self, documents: str) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {documents}").fetchone()
        return int(row["n"])

    def _write_counts(self, documents: str) -> None:
        row = self._conn.execute(
            f"SELECT COUNT(*) AS n, COALESCE(SUM(token_count), 0) AS tokens FROM {documents}"
        ).fetchone()
        self._set_meta("document_count", str(int(row["n"])))
        self._set_meta("total_tokens", str(int(row["tokens"])))

    def _activate(self, generation: str, documents: str) -> None:
        self._write_counts(documents)
        self._set_meta("generation", generation)
        self._set_meta("schema_version", str(SCHEMA_VERSION))
        self._set_meta("ready", "1")

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO index_metadata(key, value) VALUES(?, ?)",
            (key, value),
        )
