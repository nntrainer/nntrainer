# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file code_bank.py
# @brief Code bank: a local SQLite database of reusable nntrainer CausalLM code.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Code bank: a local SQLite database of nntrainer CausalLM code the workbench
can reuse or show as examples.

What it stores (`kind`):
  hook       one overridden method body (createAttention, createMlp, ...)
  layer      a custom nntrainer layer (.h/.cpp)
  converter  a HF -> nntrainer weight converter script
  framework  base classes (Transformer, CausalLM, ...) -- so generation still
             works if Applications/CausalLM/models is not there
  model      a hand-written model's original files, kept as reference

Life cycle (`status`), enforced here:
  generated -> approved (by the user) -> verified (by the machine) ;
  approved/verified -> failed_verification -> approved (after a fix) ;
  verified/approved -> withdrawn ; generated/approved -> rejected.

Reuse rule: only `verified` entries are ever returned as reusable
(`find_reusable`). `approved` entries are only offered as examples
(`search`, default statuses), never reused automatically.

Search: exact match on the feature combination first (`find_matching`),
then keyword search (SQLite FTS5, BM25). If a real embedding backend is
available (sentence-transformers, or a reachable Ollama server), keyword
results are re-ranked by embedding similarity; otherwise ranking is
keyword-only -- there is no random stand-in.

Location: tools/agentic-ai-workbench/code_bank/code_bank.db in the nntrainer
checkout (kept out of git), or $WORKBENCH_CODE_BANK.
Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import urllib.request
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
import argparse

SCHEMA_VERSION = "1"
FTS_VERSION = "2"        # keyword index format; a change rebuilds the index

KINDS = ("hook", "layer", "converter", "framework", "model")
STATUSES = ("generated", "approved", "verified", "failed_verification", "rejected", "withdrawn")
REUSABLE_STATUSES = ("verified",)
EXAMPLE_STATUSES = ("verified", "approved")

_TRANSITIONS = {
    "approve": ({"generated", "failed_verification"}, "approved"),
    "verify": ({"approved"}, "verified"),
    "fail_verification": ({"approved", "verified"}, "failed_verification"),
    "reject": ({"generated", "approved", "failed_verification"}, "rejected"),
    "withdraw": ({"approved", "verified"}, "withdrawn"),
}

_JSON_FIELDS = ("features", "supports", "evidence", "files", "requires", "verification")


class CodeBankError(ValueError):
    """A code bank action is not allowed, or its data is invalid."""


def workbench_dir() -> Optional[Path]:
    """The nntrainer checkout's tools/agentic-ai-workbench folder: found from
    this file's location when the engine runs from the checkout, or from
    $NNTRAINER_ROOT when it runs from an installed copy of the extension."""
    for parent in Path(__file__).resolve().parents:
        if parent.name == "agentic-ai-workbench" and parent.parent.name == "tools":
            return parent
    root = os.environ.get("NNTRAINER_ROOT")
    if root:
        cand = Path(root).expanduser() / "tools" / "agentic-ai-workbench"
        if cand.is_dir():
            return cand
    return None


def default_db_path() -> Path:
    """tools/agentic-ai-workbench/code_bank/code_bank.db (or $WORKBENCH_CODE_BANK)."""
    env = os.environ.get("WORKBENCH_CODE_BANK")
    if env:
        return Path(env).expanduser()
    wb_dir = workbench_dir()
    if wb_dir is None:
        raise CodeBankError(
            "cannot find tools/agentic-ai-workbench: set NNTRAINER_ROOT to the nntrainer "
            "checkout, or WORKBENCH_CODE_BANK to the database file"
        )
    return wb_dir / "code_bank" / "code_bank.db"


def feature_key(features: Iterable[str]) -> str:
    """A stable key for a set of features."""
    return "+".join(sorted(set(features)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def keywords(text: str) -> str:
    """Words for the keyword index (underscores are part of a word in it):
    identifiers are kept whole, split at camelCase / snake_case boundaries,
    and snake_case sub-names are added, so `createAttention` is found by
    "attention", `q_norm` by "q_norm" (not just "norm") and
    `reshaped_rms_norm` by "rms_norm"."""
    out: List[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text or ""):
        word = word.strip("_")
        if not word:
            continue
        out.append(word.lower())
        parts = [p for p in word.split("_") if p]
        lower = [p.lower() for p in parts]
        if len(parts) > 1:
            for i in range(len(lower)):
                for j in range(i + 2, min(len(lower), i + 4) + 1):
                    if j - i < len(lower):
                        out.append("_".join(lower[i:j]))
        pieces = [s.lower() for p in parts for s in _CAMEL.split(p) if s]
        if len(pieces) > 1:
            out.extend(pieces)
    return " ".join(out)


def _content_hash(entry: Dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for part in (entry.get("code", ""), json.dumps(entry.get("files", {}), sort_keys=True),
                 feature_key(entry.get("features", [])), feature_key(entry.get("supports", []))):
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def matches_features(
    required: Sequence[str], supports: Sequence[str], ticked: Iterable[str]
) -> bool:
    """An entry fits a set of ticked features when it implements everything it
    requires and nothing ticked falls outside what it requires or supports."""
    ticked_set, req = set(ticked), set(required)
    return req <= ticked_set <= req | set(supports)


# ----------------------------------------------------------- embeddings
# pylint: disable-next=too-few-public-methods
class Embedder:
    """Interface: `model` names the backend+model (used as a cache key)."""
    model = ""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embedding vectors for the texts."""
        raise NotImplementedError


# pylint: disable-next=too-few-public-methods
class _SentenceTransformersEmbedder(Embedder):
    def __init__(self, name: str):
        # pylint: disable-next=import-outside-toplevel,import-error  # optional package
        from sentence_transformers import SentenceTransformer  # noqa: optional
        self._m = SentenceTransformer(name)
        self.model = f"st:{name}"

    def embed(self, texts):
        return [list(map(float, v)) for v in self._m.encode(list(texts))]


# pylint: disable-next=too-few-public-methods
class _OllamaEmbedder(Embedder):
    def __init__(self, name: str, url: str):
        self.url = url.rstrip("/")
        self.name = name
        self.model = f"ollama:{name}"
        self.embed(["probe"])                       # raises if unusable

    def embed(self, texts):
        out = []
        for text in texts:
            req = urllib.request.Request(
                f"{self.url}/api/embeddings",
                data=json.dumps({"model": self.name, "prompt": text}).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                vec = json.loads(resp.read().decode("utf-8")).get("embedding")
            if not vec:
                raise RuntimeError("Ollama returned no embedding")
            out.append([float(x) for x in vec])
        return out


_EMBEDDER_CACHE: Dict[str, Optional[Embedder]] = {}


def resolve_embedder(spec: Optional[str] = None) -> Optional[Embedder]:
    """`spec`: "off", "auto" (default, or $WORKBENCH_EMBEDDINGS), "st:<model>",
    "ollama:<model>". Returns None when nothing real is available -- callers
    then rank by keywords only."""
    spec = (spec or os.environ.get("WORKBENCH_EMBEDDINGS") or "auto").strip()
    if spec in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[spec]
    emb: Optional[Embedder] = None
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not url.startswith("http"):
        url = "http://" + url
    tries: List[Callable[[], Embedder]] = []
    if spec == "off":
        tries = []
    elif spec.startswith("st:"):
        tries = [lambda: _SentenceTransformersEmbedder(spec[3:])]
    elif spec.startswith("ollama:"):
        tries = [lambda: _OllamaEmbedder(spec[7:], url)]
    else:  # auto
        tries = [lambda: _SentenceTransformersEmbedder("all-MiniLM-L6-v2"),
                 lambda: _OllamaEmbedder("nomic-embed-text", url)]
    for make in tries:
        try:
            emb = make()
            break
        # the embedding model is optional; any failure turns it off
        except Exception:  # pylint: disable=broad-except
            emb = None
    _EMBEDDER_CACHE[spec] = emb
    return emb


def _cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ------------------------------------------------------------- the store
class CodeBank:
    """The SQLite code bank: entries, their review history, metadata and learned features."""
    def __init__(self, path: Optional[os.PathLike] = None):
        self.path = Path(path) if path else default_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.fts = True
        self._init_schema()

    # -- schema
    def _init_schema(self) -> None:
        conn = self.conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                architecture TEXT NOT NULL DEFAULT '',
                hook TEXT NOT NULL DEFAULT '',
                variant TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT '',
                features TEXT NOT NULL DEFAULT '[]',
                supports TEXT NOT NULL DEFAULT '[]',
                feature_key TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '{}',
                files TEXT NOT NULL DEFAULT '{}',
                code TEXT NOT NULL DEFAULT '',
                requires TEXT NOT NULL DEFAULT '[]',
                origin TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                source_ref TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                verification TEXT NOT NULL DEFAULT '{}',
                approved_by TEXT NOT NULL DEFAULT '',
                approved_at TEXT NOT NULL DEFAULT '',
                verified_at TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_match ON entries(kind, hook, status);
            CREATE TABLE IF NOT EXISTS history (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL, at TEXT NOT NULL, action TEXT NOT NULL,
                actor TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS learned_features (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL,
                source_entry TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                entry_id TEXT NOT NULL, model TEXT NOT NULL,
                content_hash TEXT NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY (entry_id, model)
            );
        """)
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,)
        )
        row = conn.execute("SELECT value FROM meta WHERE key = 'fts_version'").fetchone()
        try:
            if not row or row[0] != FTS_VERSION:
                # Index format changed (or first run): rebuild it from the entries.
                conn.execute("DROP TABLE IF EXISTS entries_fts")
            conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                entry_id UNINDEXED, name, architecture, hook, features, evidence, keywords, notes,
                tokenize = "unicode61 tokenchars '_'")""")
            if not row or row[0] != FTS_VERSION:
                for res in conn.execute("SELECT * FROM entries").fetchall():
                    self._index(self._row(res))
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('fts_version', ?)",
                    (FTS_VERSION,),
                )
        except sqlite3.OperationalError:
            self.fts = False                       # no FTS5: Python-side keyword scoring
        conn.commit()

    def close(self) -> None:
        """Close the database."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- rows
    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        for json_field in _JSON_FIELDS:
            data[json_field] = (
                json.loads(data[json_field])
                if data.get(json_field)
                else ([] if json_field in ("features", "supports", "requires") else {})
            )
        data.pop("rank", None)
        return data

    @staticmethod
    def _normalise(entry: Dict[str, Any]) -> Dict[str, Any]:
        elem = dict(entry)
        for k in ("id", "kind", "name", "status"):
            if not elem.get(k):
                raise CodeBankError(f"entry needs '{k}'")
        if elem["kind"] not in KINDS:
            raise CodeBankError(f"unknown kind {elem['kind']!r} (one of {KINDS})")
        if elem["status"] not in STATUSES:
            raise CodeBankError(f"unknown status {elem['status']!r} (one of {STATUSES})")
        elem.setdefault("features", [])
        elem.setdefault("supports", [])
        elem.setdefault("evidence", {})
        elem.setdefault("files", {})
        elem.setdefault("requires", [])
        elem.setdefault("verification", {})
        for k in (
            "architecture",
            "hook",
            "variant",
            "role",
            "code",
            "origin",
            "source",
            "source_ref",
            "approved_by",
            "approved_at",
            "verified_at",
            "notes",
        ):
            elem[k] = elem.get(k) or ""
        elem["features"] = sorted(set(elem["features"]))
        elem["supports"] = sorted(set(elem["supports"]) - set(elem["features"]))
        elem["feature_key"] = feature_key(elem["features"])
        elem["content_hash"] = _content_hash(elem)
        if elem["status"] == "verified":
            verification = elem["verification"]
            if not (isinstance(verification, dict) and verification.get("method")):
                raise CodeBankError(
                    f"{elem['id']}: a verified entry needs verification evidence with a 'method'"
                )
            elem["verified_at"] = elem["verified_at"] or _now()
        return elem

    def _write(self, entry: Dict[str, Any]) -> None:
        cols = (
            "id",
            "kind",
            "name",
            "architecture",
            "hook",
            "variant",
            "role",
            "features",
            "supports",
            "feature_key",
            "evidence",
            "files",
            "code",
            "requires",
            "origin",
            "source",
            "source_ref",
            "status",
            "verification",
            "approved_by",
            "approved_at",
            "verified_at",
            "content_hash",
            "notes",
            "created_at",
            "updated_at",
        )
        vals = [
            json.dumps(entry[c], sort_keys=True) if c in _JSON_FIELDS else entry[c] for c in cols
        ]
        self.conn.execute(
            f"INSERT OR REPLACE INTO entries({','.join(cols)}) VALUES"
            f" ({','.join('?' * len(cols))})",
            vals,
        )
        if self.fts:
            self._index(entry)

    def _index(self, entry: Dict[str, Any]) -> None:
        self.conn.execute("DELETE FROM entries_fts WHERE entry_id = ?", (entry["id"],))
        files_text = " ".join(entry["files"].keys())
        evidence = {k: v for k, v in entry["evidence"].items() if k != "header_text"}
        self.conn.execute(
            "INSERT INTO entries_fts(entry_id, name, architecture, hook, features, evidence,"
            " keywords, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry["id"],
                keywords(entry["name"]),
                keywords(entry["architecture"]),
                keywords(entry["hook"]),
                keywords(" ".join(entry["features"] + entry["supports"])),
                keywords(json.dumps(evidence)),
                keywords(
                    entry["code"] + " " + files_text + " " + " ".join(entry["files"].values())
                ),
                keywords(entry["notes"]),
            ),
        )

    def _log(self, entry_id: str, action: str, actor: str, detail: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO history(entry_id, at, action, actor, detail) VALUES (?, ?, ?, ?, ?)",
            (entry_id, _now(), action, actor or "", json.dumps(detail or {}, sort_keys=True)),
        )

    def put(self, entry: Dict[str, Any], actor: str = "workbench") -> str:
        """Insert or replace an entry as given (used by capture). Returns
        "added", "updated" or "unchanged"."""
        elem = self._normalise(entry)
        old = self.get(elem["id"])
        now = _now()
        elem["created_at"] = old["created_at"] if old else now
        elem["updated_at"] = now
        if (
            old
            and old["content_hash"] == elem["content_hash"]
            and old["status"] == elem["status"]
            and old["verification"] == elem["verification"]
            and old["evidence"] == elem["evidence"]
        ):
            return "unchanged"
        self._write(elem)
        self.conn.execute("DELETE FROM embeddings WHERE entry_id = ?", (elem["id"],))
        self._log(
            elem["id"],
            "update" if old else "add",
            actor,
            {
                "status": elem["status"],
                "content_changed": bool(old and old["content_hash"] != elem["content_hash"]),
            },
        )
        self.conn.commit()
        return "updated" if old else "added"

    def add_generated(self, entry: Dict[str, Any], actor: str = "workbench") -> str:
        """New code from the generator: always starts as `generated`."""
        elem = dict(
            entry,
            status="generated",
            verification={},
            approved_by="",
            approved_at="",
            verified_at="",
        )
        elem.setdefault("source", "generated")
        if self.get(elem["id"]):
            raise CodeBankError(f"{elem['id']} already exists")
        self.put(elem, actor)
        return elem["id"]

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """The entry with this id, or None."""
        row = self.conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        return self._row(row) if row else None

    def list(
        self,
        kind: Optional[str] = None,
        status: Optional[Sequence[str]] = None,
        architecture: Optional[str] = None,
        hook: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Entries, filtered by kind, status, architecture and hook."""
        where, args = [], []
        for col, val in (("kind", kind), ("architecture", architecture), ("hook", hook)):
            if val:
                where.append(f"{col} = ?")
                args.append(val)
        if status:
            status = [status] if isinstance(status, str) else list(status)
            where.append(f"status IN ({','.join('?' * len(status))})")
            args.extend(status)
        sql = (
            "SELECT * FROM entries"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY id"
        )
        return [self._row(r) for r in self.conn.execute(sql, args)]

    def history(self, entry_id: str) -> List[Dict[str, Any]]:
        """An entry's review history, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM history WHERE entry_id = ? ORDER BY seq", (entry_id,)
        )
        return [dict(r, detail=json.loads(r["detail"])) for r in rows]

    def stats(self) -> Dict[str, Any]:
        """Entry counts by kind and by status."""
        out: Dict[str, Any] = {"path": str(self.path), "total": 0, "by_kind": {}, "by_status": {}}
        for row in self.conn.execute(
            "SELECT kind, status, COUNT(*) n FROM entries GROUP BY kind, status"
        ):
            out["total"] += row["n"]
            out["by_kind"].setdefault(row["kind"], {})[row["status"]] = row["n"]
            out["by_status"][row["status"]] = out["by_status"].get(row["status"], 0) + row["n"]
        out["learned_features"] = self.conn.execute(
            "SELECT COUNT(*) FROM learned_features"
        ).fetchone()[0]
        return out

    # -- life cycle
    # pylint: disable-next=too-many-arguments
    def _transition(
        self, entry_id: str, action: str, actor: str, detail: dict, updates: dict
    ) -> Dict[str, Any]:
        entry = self.get(entry_id)
        if not entry:
            raise CodeBankError(f"no entry {entry_id!r}")
        allowed, target = _TRANSITIONS[action]
        previous = entry["status"]
        if previous not in allowed:
            raise CodeBankError(f"{entry_id}: cannot {action} an entry that is {previous!r}")
        entry.update(updates, status=target, updated_at=_now())
        entry = self._normalise(entry)
        self._write(entry)
        self._log(entry_id, action, actor, dict(detail, **{"from": previous, "to": target}))
        self.conn.commit()
        return entry

    def approve(self, entry_id: str, actor: str, notes: str = "") -> Dict[str, Any]:
        """Mark an entry approved by `actor`."""
        if not actor:
            raise CodeBankError("approve needs who approved it")
        entry = self.get(entry_id) or {}
        return self._transition(
            entry_id,
            "approve",
            actor,
            {"notes": notes},
            {
                "approved_by": actor,
                "approved_at": _now(),
                "notes": (
                    entry.get("notes", "") + ("\n" if entry.get("notes") and notes else "") + notes
                ),
            },
        )

    def mark_verified(
        self, entry_id: str, evidence: Dict[str, Any], actor: str = "verifier"
    ) -> Dict[str, Any]:
        """Mark an entry verified, with the evidence of how."""
        if not evidence.get("method"):
            raise CodeBankError("verification evidence needs a 'method'")
        return self._transition(entry_id, "verify", actor, {"evidence": evidence},
                                {"verification": evidence, "verified_at": _now()})

    def mark_failed(
        self, entry_id: str, evidence: Dict[str, Any], actor: str = "verifier"
    ) -> Dict[str, Any]:
        """Mark an entry as failed verification."""
        return self._transition(entry_id, "fail_verification", actor, {"evidence": evidence},
                                {"verification": dict(evidence, passed=False), "verified_at": ""})

    def reject(self, entry_id: str, actor: str, reason: str = "") -> Dict[str, Any]:
        """Reject an entry."""
        return self._transition(entry_id, "reject", actor, {"reason": reason}, {})

    def withdraw(self, entry_id: str, actor: str, reason: str = "") -> Dict[str, Any]:
        """Withdraw an entry from reuse."""
        return self._transition(entry_id, "withdraw", actor, {"reason": reason}, {})

    # -- matching
    def find_matching(self, kind: str, ticked: Iterable[str], hook: Optional[str] = None,
                      statuses: Sequence[str] = EXAMPLE_STATUSES) -> List[Dict[str, Any]]:
        """Entries whose feature set fits `ticked` exactly (see
        `matches_features`), best fit first: an exact feature match, then the
        entry with the fewest extra supported features."""
        ticked = set(ticked)
        found = [e for e in self.list(kind=kind, status=statuses, hook=hook)
                 if e["features"] and matches_features(e["features"], e["supports"], ticked)]
        found.sort(key=lambda e: (set(e["features"]) != ticked, len(e["supports"]),
                                  e["variant"] not in ("", "default"), e["id"]))
        return found

    def find_reusable(
        self, kind: str, ticked: Iterable[str], hook: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """The only way to get code for automatic reuse: verified entries only."""
        return self.find_matching(kind, ticked, hook, statuses=REUSABLE_STATUSES)

    # -- search
    # pylint: disable-next=too-many-locals,too-many-arguments
    def search(self, query: str, kind: Optional[str] = None, hook: Optional[str] = None,
               statuses: Sequence[str] = EXAMPLE_STATUSES, limit: int = 5,
               embeddings: Optional[str] = None) -> List[Dict[str, Any]]:
        """Keyword search (BM25), optionally re-ranked by embeddings.
        `embeddings`: None -> $WORKBENCH_EMBEDDINGS / auto, "off" to disable."""
        words = re.findall(r"[A-Za-z0-9_]+", query or "")
        terms = sorted({w for w in keywords(" ".join(words)).split() if len(w) > 1})
        if not terms:
            return []
        statuses = list(statuses)
        hits: List[Dict[str, Any]] = []
        if self.fts:
            where = ["entries_fts MATCH ?", f"e.status IN ({','.join('?' * len(statuses))})"]
            args: List[Any] = [" OR ".join(f'"{t}"' for t in terms), *statuses]
            if kind:
                where.append("e.kind = ?")
                args.append(kind)
            if hook:
                where.append("e.hook = ?")
                args.append(hook)
            sql = ("SELECT e.*, bm25(entries_fts, 0.0, 3.0, 2.0, 2.0, 3.0, 1.5, 1.0, 1.0) AS rank "
                   "FROM entries_fts JOIN entries e ON e.id = entries_fts.entry_id "
                   f"WHERE {' AND '.join(where)} ORDER BY rank LIMIT ?")
            for row in self.conn.execute(sql, args + [max(limit * 4, 20)]):
                data = self._row(row)
                data["keyword_score"] = -float(row["rank"])
                hits.append(data)
        else:
            for entry in self.list(kind=kind, status=statuses, hook=hook):
                text = keywords(
                    " ".join(
                        [entry["name"], entry["hook"], " ".join(entry["features"]), entry["code"]]
                    )
                ).split()
                score = sum(text.count(t) for t in terms)
                if score:
                    hits.append(dict(entry, keyword_score=float(score)))
            hits.sort(key=lambda d: -d["keyword_score"])
            hits = hits[:max(limit * 4, 20)]
        for i, hit in enumerate(hits):
            hit["match"] = "keyword"
            hit["score"] = 1.0 - i / max(len(hits), 1)
        emb = resolve_embedder(embeddings)
        if emb and hits:
            try:
                query_vec = emb.embed([query])[0]
                for hit in hits:
                    sim = _cosine(query_vec, self._vector(hit, emb))
                    hit["embedding_similarity"] = sim
                    hit["score"] = 0.5 * hit["score"] + 0.5 * sim
                    hit["match"] = "keyword+embedding"
                hits.sort(key=lambda d: -d["score"])
            # the embedding model is optional; any failure turns it off
            except Exception:  # pylint: disable=broad-except
                pass                                # keep keyword ranking
        return hits[:limit]

    def _vector(self, entry: Dict[str, Any], emb: Embedder) -> List[float]:
        row = self.conn.execute(
            "SELECT content_hash, vector FROM embeddings WHERE entry_id = ? AND model = ?",
            (entry["id"], emb.model),
        ).fetchone()
        if row and row["content_hash"] == entry["content_hash"]:
            return array("f", row["vector"]).tolist()
        text = " ".join(
            [entry["name"], entry["hook"], " ".join(entry["features"]), entry["code"][:4000]]
        )
        vec = emb.embed([text])[0]
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings(entry_id, model, content_hash, vector) VALUES"
            " (?,?,?,?)",
            (entry["id"], emb.model, entry["content_hash"], array("f", vec).tobytes()),
        )
        self.conn.commit()
        return vec

    # -- small key/value facts about the bank (e.g. nntrainer's core layer types)
    def get_meta(self, key: str, default: Any = None) -> Any:
        """A stored metadata value, or `default`."""
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except ValueError:
            return row[0]

    def set_meta(self, key: str, value: Any) -> None:
        """Store a metadata value."""
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value))
        )
        self.conn.commit()

    # -- learned checklist boxes
    def put_feature(self, spec: Dict[str, Any], source_entry: str = "") -> None:
        """Store (or replace) a learned checklist feature."""
        if not spec.get("id") or not spec.get("when"):
            raise CodeBankError("a feature needs an 'id' and 'when' rules")
        self.conn.execute(
            "INSERT OR REPLACE INTO learned_features(id, spec, source_entry, created_at) VALUES"
            " (?,?,?,?)",
            (spec["id"], json.dumps(spec, sort_keys=True), source_entry, _now()),
        )
        self.conn.commit()

    def learned_features(self) -> List[Dict[str, Any]]:
        """The learned checklist features, with the entry each came from."""
        out = []
        for row in self.conn.execute("SELECT spec, source_entry FROM learned_features ORDER BY id"):
            spec = json.loads(row["spec"])
            spec.setdefault("source_entry", row["source_entry"])
            spec["source"] = "learned"
            out.append(spec)
        return out


# ------------------------------------------------------------------ CLI
def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m knowledge.code_bank", description="Inspect the local code bank."
    )
    parser.add_argument(
        "--db", help="database path (default: tools/agentic-ai-workbench/code_bank/code_bank.db)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    list_parser = sub.add_parser("list")
    list_parser.add_argument("--kind")
    list_parser.add_argument("--status")
    show_parser = sub.add_parser("show")
    show_parser.add_argument("id")
    search_parser = sub.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--kind")
    search_parser.add_argument("--hook")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument(
        "--embeddings", default=None, help='"off", "auto", "st:<model>", "ollama:<model>"'
    )
    args = parser.parse_args(argv)
    with CodeBank(args.db) as bank:
        if args.cmd == "stats":
            print(json.dumps(bank.stats(), indent=2))
        elif args.cmd == "list":
            for entry in bank.list(kind=args.kind, status=args.status):
                print(f"{entry['status']:<20} {entry['kind']:<10} {entry['id']}")
        elif args.cmd == "show":
            entry = bank.get(args.id)
            if not entry:
                print(f"no entry {args.id}")
                return 1
            print(
                json.dumps({k: v for k, v in entry.items() if k not in ("files", "code")}, indent=2)
            )
            print(
                "\n"
                + (
                    entry["code"]
                    or "\n".join(f"--- {k} ({len(v)} chars)" for k, v in entry["files"].items())
                )
            )
        elif args.cmd == "search":
            for hit in bank.search(
                args.query,
                kind=args.kind,
                hook=args.hook,
                limit=args.limit,
                embeddings=args.embeddings,
            ):
                print(f"{hit['score']:.3f} {hit['match']:<18} {hit['status']:<9} {hit['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
