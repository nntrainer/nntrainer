"""
RAG-style memory store for pipeline run history.

Uses SQLite with FTS5 full-text search for lightweight keyword-based retrieval.
No embedding model required initially — can be added later for semantic search.

Storage location: nntrainer/workbench_output/run_memory.db
"""
from __future__ import annotations

import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default memory store location (inside nntrainer workbench_output)
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent.parent / "workbench_output" / "run_memory.db"

# Import embedding support (lazy, optional)
def _get_embedder():
    """Lazy import of embedding service."""
    try:
        from .embeddings import create_embedder
        return create_embedder(backend="auto")
    except Exception as e:
        logger.debug("Embedding service unavailable: %s", e)
        return None


class MemoryStore:
    """
    FTS5-based memory store for pipeline run episodes.
    
    Captures:
    - Compile errors and fixes
    - Link errors and fixes
    - Runtime errors and fixes
    - Successful runs
    
    Search:
    - Keyword search via FTS5 (BM25 ranking)
    - Filter by model family (Qwen3, Llama2, etc.)
    - Recency boost (recent runs ranked higher)
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize memory store.
        
        Args:
            db_path: Path to SQLite database. Defaults to workbench_output/run_memory.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._init_schema()
        
        logger.info("MemoryStore initialized at %s", self.db_path)
    
    def _init_schema(self):
        """Initialize FTS5 schema for run episodes."""
        
        # FTS5 virtual table for full-text search
        # All searchable text columns go here
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS run_episodes USING fts5(
                model_name,
                model_family,
                event_type,
                error_type,
                error_msg,
                fix_applied,
                file_path,
                content,
                agent_name,
                content_rowid='rowid'
            )
        """)
        
        # Metadata table (joined with FTS5 for structured queries)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episode_meta (
                rowid INTEGER PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_family TEXT NOT NULL,
                event_type TEXT NOT NULL,
                error_type TEXT,
                success BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                duration_sec REAL,
                tokens_used INTEGER,
                pipeline_config TEXT
            )
        """)
        
        # Index on model_family for fast filtering
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_family ON episode_meta(model_family)
        """)
        
        # Index on created_at for recency queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON episode_meta(created_at)
        """)
        
        self.conn.commit()
        logger.debug("MemoryStore schema initialized")
    
    def add_episode(self, event: Dict[str, Any]):
        """
        Add a pipeline event to memory.
        
        Args:
            event: Dictionary with event data. Expected keys:
                - model_name: str (e.g., "Qwen3-1.8B")
                - event_type: str ("compile_error", "link_error", "runtime_error", "success")
                - error_type: str (optional, e.g., "undefined_reference", "missing_include")
                - error_msg: str (optional, the actual error message)
                - fix_applied: str (optional, what fixed it)
                - file_path: str (optional, relevant file)
                - content: str (full event content/details)
                - agent_name: str (which agent generated this)
                - duration_sec: float (optional, pipeline duration)
                - tokens_used: int (optional, LLM tokens consumed)
                - pipeline_config: str (optional, JSON config)
        """
        model_name = event.get("model_name", "")
        model_family = model_name.split("-")[0] if model_name else "unknown"
        
        # Insert into FTS5 table
        self.conn.execute("""
            INSERT INTO run_episodes(
                model_name, model_family, event_type, error_type, error_msg, 
                fix_applied, file_path, content, agent_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            model_family,
            event.get("event_type", "unknown"),
            event.get("error_type", ""),
            event.get("error_msg", ""),
            event.get("fix_applied", ""),
            event.get("file_path", ""),
            event.get("content", ""),
            event.get("agent_name", "unknown"),
        ))
        
        # Get the rowid for metadata insert
        rowid = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        # Insert metadata
        self.conn.execute("""
            INSERT INTO episode_meta(
                rowid, model_name, model_family, event_type, error_type, 
                success, duration_sec, tokens_used, pipeline_config
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rowid,
            model_name,
            model_family,
            event.get("event_type", "unknown"),
            event.get("error_type", ""),
            event.get("success", True),
            event.get("duration_sec"),
            event.get("tokens_used"),
            event.get("pipeline_config"),
        ))
        
        self.conn.commit()
        logger.debug("Added episode for model=%s, event_type=%s", model_name, event.get("event_type"))
    
    def search(
        self, 
        query: str, 
        model_family: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search run episodes using FTS5 BM25 ranking.
        
        Args:
            query: Search query (keywords)
            model_family: Filter by model family (e.g., "Qwen3", "Llama")
            event_type: Filter by event type (e.g., "compile_error", "success")
            limit: Max results to return
        
        Returns:
            List of matching episodes with metadata
        """
        # Build query with filters
        conditions = []
        params = []
        
        # FTS5 query
        fts_query = query if query else "*"
        
        # Additional WHERE conditions
        if model_family:
            conditions.append("m.model_family = ?")
            params.append(model_family)
        
        if event_type:
            conditions.append("e.event_type = ?")
            params.append(event_type)
        
        # Add FTS5 match condition
        conditions.append("run_episodes MATCH ?")
        params.append(fts_query)
        
        where_clause = " AND ".join(conditions)
        
        # Execute query with recency boost (newer results first for same rank)
        sql = f"""
            SELECT 
                e.rowid,
                e.model_name,
                e.model_family,
                e.event_type,
                e.error_type,
                e.error_msg,
                e.fix_applied,
                e.file_path,
                e.content,
                e.agent_name,
                m.success,
                m.created_at,
                m.duration_sec,
                m.tokens_used,
                bm25(run_episodes, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0) as rank
            FROM run_episodes e
            JOIN episode_meta m ON e.rowid = m.rowid
            WHERE {where_clause}
            ORDER BY rank ASC, m.created_at DESC
            LIMIT ?
        """
        params.append(limit * 2)  # Get more for re-ranking
        
        cursor = self.conn.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            results.append({
                "rowid": row["rowid"],
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "event_type": row["event_type"],
                "error_type": row["error_type"],
                "error_msg": row["error_msg"],
                "fix_applied": row["fix_applied"],
                "file_path": row["file_path"],
                "content": row["content"],
                "agent_name": row["agent_name"],
                "success": bool(row["success"]),
                "created_at": row["created_at"],
                "duration_sec": row["duration_sec"],
                "tokens_used": row["tokens_used"],
                "rank": -row["rank"],  # Invert so higher is better
            })
        
        return results[:limit]
    
    def get_similar_runs(self, model_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get prior runs from the same model family.
        
        Args:
            model_name: Model name (e.g., "Qwen3-1.8B")
            limit: Max results
        
        Returns:
            List of prior runs for the same model family
        """
        model_family = model_name.split("-")[0]
        
        cursor = self.conn.execute("""
            SELECT 
                e.rowid,
                e.model_name,
                e.model_family,
                e.event_type,
                e.error_type,
                e.error_msg,
                e.fix_applied,
                e.content,
                m.success,
                m.created_at,
                m.duration_sec
            FROM run_episodes e
            JOIN episode_meta m ON e.rowid = m.rowid
            WHERE m.model_family = ?
            ORDER BY m.created_at DESC
            LIMIT ?
        """, (model_family, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "rowid": row["rowid"],
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "event_type": row["event_type"],
                "error_type": row["error_type"],
                "error_msg": row["error_msg"],
                "fix_applied": row["fix_applied"],
                "content": row["content"],
                "success": bool(row["success"]),
                "created_at": row["created_at"],
                "duration_sec": row["duration_sec"],
            })
        
        return results
    
    def get_error_patterns(
        self, 
        model_family: str, 
        error_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get error patterns for a model family.
        
        Args:
            model_family: Model family (e.g., "Qwen3")
            error_type: Optional filter (e.g., "undefined_reference")
            limit: Max results
        
        Returns:
            List of error patterns with fixes
        """
        conditions = ["m.model_family = ?", "e.error_msg != ''"]
        params = [model_family]
        
        if error_type:
            conditions.append("e.error_type = ?")
            params.append(error_type)
        
        where_clause = " AND ".join(conditions)
        
        cursor = self.conn.execute(f"""
            SELECT 
                e.error_msg,
                e.error_type,
                e.fix_applied,
                e.file_path,
                e.model_name,
                m.created_at,
                COUNT(*) as occurrence_count
            FROM run_episodes e
            JOIN episode_meta m ON e.rowid = m.rowid
            WHERE {where_clause}
            GROUP BY e.error_msg, e.error_type, e.fix_applied
            ORDER BY occurrence_count DESC, m.created_at DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "error_msg": row["error_msg"],
                "error_type": row["error_type"],
                "fix_applied": row["fix_applied"],
                "file_path": row["file_path"],
                "model_name": row["model_name"],
                "created_at": row["created_at"],
                "occurrence_count": row["occurrence_count"],
            })
        
        return results
    
    def get_success_patterns(self, model_family: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get successful run patterns for a model family.
        
        Args:
            model_family: Model family (e.g., "Qwen3")
            limit: Max results
        
        Returns:
            List of successful runs
        """
        cursor = self.conn.execute("""
            SELECT 
                e.model_name,
                e.content,
                m.created_at,
                m.duration_sec,
                m.tokens_used
            FROM run_episodes e
            JOIN episode_meta m ON e.rowid = m.rowid
            WHERE m.model_family = ? AND m.success = 1
            ORDER BY m.created_at DESC
            LIMIT ?
        """, (model_family, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "model_name": row["model_name"],
                "content": row["content"],
                "created_at": row["created_at"],
                "duration_sec": row["duration_sec"],
                "tokens_used": row["tokens_used"],
            })
        
        return results
    
    def semantic_search(
        self,
        query: str,
        model_family: Optional[str] = None,
        limit: int = 5,
        use_semantic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search with semantic similarity (hybrid: FTS5 + embeddings).
        
        Args:
            query: Search query
            model_family: Optional filter
            limit: Max results
            use_semantic: Enable semantic re-ranking (requires embedding backend)
        
        Returns:
            List of matching episodes, re-ranked by semantic similarity
        """
        # First get FTS5 results
        fts_results = self.search(query, model_family, limit=None, event_type=None)
        
        if not fts_results or not use_semantic:
            return fts_results[:limit]
        
        # Try semantic re-ranking
        embedder = _get_embedder()
        if embedder is None:
            logger.debug("No embedding service available, using FTS5 results")
            return fts_results[:limit]
        
        try:
            # Generate query embedding
            query_embedding = embedder.embed(query)
            
            # Generate document embeddings (cache would be better, but this works for demo)
            doc_embeddings = []
            valid_results = []
            for r in fts_results:
                content = (r.get("error_msg", "") + " " + 
                          r.get("fix_applied", "") + " " + 
                          r.get("content", ""))[:500]  # Truncate for speed
                if content.strip():
                    doc_embeddings.append(embedder.embed(content))
                    valid_results.append(r)
            
            if not doc_embeddings:
                return fts_results[:limit]
            
            # Compute similarities
            import numpy as np
            doc_matrix = np.array(doc_embeddings)
            
            # Normalize
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            doc_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8)
            
            # Cosine similarity
            similarities = np.dot(doc_norms, query_norm)
            
            # Re-rank by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            reranked = [valid_results[i] for i in sorted_indices[:limit]]
            
            # Add similarity scores
            for i, idx in enumerate(sorted_indices[:limit]):
                reranked[i]["semantic_similarity"] = float(similarities[idx])
            
            logger.debug("Semantic re-ranking complete: %d results", len(reranked))
            return reranked
            
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)
            return fts_results[:limit]
    
    def get_cross_model_patterns(
        self,
        source_model_family: str,
        target_model_family: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find transferable patterns between model families.
        
        This enables cross-model learning: errors/fixes from Qwen2 that
        also apply to Qwen3, or Llama2 patterns that help with Llama3.
        
        Args:
            source_model_family: Source model family (e.g., "Qwen2")
            target_model_family: Target model family (e.g., "Qwen3")
            limit: Max results
        
        Returns:
            List of patterns from source that may apply to target
        """
        # Get error patterns from source model
        source_patterns = self.get_error_patterns(source_model_family, limit=limit * 2)
        
        # Find similar errors in target model
        target_errors = set()
        cursor = self.conn.execute("""
            SELECT DISTINCT e.error_msg, e.error_type
            FROM run_episodes e
            JOIN episode_meta m ON e.rowid = m.rowid
            WHERE m.model_family = ? AND e.error_msg != ''
        """, (target_model_family,))
        for row in cursor.fetchall():
            target_errors.add((row["error_msg"], row["error_type"]))
        
        # Find transferable patterns (same error type, different model)
        transferable = []
        for pattern in source_patterns:
            error_type = pattern.get("error_type", "")
            error_msg = pattern.get("error_msg", "")
            fix = pattern.get("fix_applied", "")
            
            # Check if similar error exists in target
            similar_in_target = any(
                t_error_type == error_type or 
                (error_msg and t_msg and error_msg[:50] == t_msg[:50])
                for t_msg, t_error_type in target_errors
            )
            
            if similar_in_target and fix:
                transferable.append({
                    **pattern,
                    "source_model_family": source_model_family,
                    "target_model_family": target_model_family,
                    "transfer_confidence": "high" if error_type else "medium",
                    "reason": f"Same error type '{error_type}' seen in both {source_model_family} and {target_model_family}"
                })
        
        return transferable[:limit]
    
    def get_model_family_insights(self, model_family: str) -> Dict[str, Any]:
        """
        Get comprehensive insights for a model family.
        
        Combines stats, common errors, success patterns, and cross-model learnings.
        
        Args:
            model_family: Model family (e.g., "Qwen3")
        
        Returns:
            Dictionary with comprehensive insights
        """
        stats = self.get_stats(model_family)
        error_patterns = self.get_error_patterns(model_family, limit=5)
        success_patterns = self.get_success_patterns(model_family, limit=3)
        
        # Find related model families for cross-model learning
        all_families = self.conn.execute(
            "SELECT DISTINCT model_family FROM episode_meta"
        ).fetchall()
        related = []
        for (fam,) in all_families:
            if fam != model_family:
                # Check if same architecture family
                if fam.split("-")[0] == model_family.split("-")[0]:
                    related.append(fam)
        
        cross_model_patterns = []
        for rel_fam in related[:2]:  # Check up to 2 related families
            patterns = self.get_cross_model_patterns(rel_fam, model_family, limit=2)
            cross_model_patterns.extend(patterns)
        
        return {
            "model_family": model_family,
            "stats": stats,
            "common_errors": error_patterns,
            "success_patterns": success_patterns,
            "cross_model_learnings": cross_model_patterns,
            "related_families": related,
        }
    
    def get_stats(self, model_family: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory store statistics.
        
        Args:
            model_family: Optional filter
        
        Returns:
            Dictionary with stats
        """
        if model_family:
            cursor = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_runs,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_runs,
                    AVG(duration_sec) as avg_duration,
                    MIN(created_at) as first_run,
                    MAX(created_at) as last_run
                FROM episode_meta
                WHERE model_family = ?
            """, (model_family,))
        else:
            cursor = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_episodes,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_runs,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_runs,
                    AVG(duration_sec) as avg_duration,
                    MIN(created_at) as first_run,
                    MAX(created_at) as last_run
                FROM episode_meta
            """)
        
        row = cursor.fetchone()
        return {
            "total_episodes": row["total_episodes"] or 0,
            "successful_runs": row["successful_runs"] or 0,
            "failed_runs": row["failed_runs"] or 0,
            "avg_duration": row["avg_duration"],
            "first_run": row["first_run"],
            "last_run": row["last_run"],
        }
    
    def clear(self, model_family: Optional[str] = None):
        """
        Clear memory (optionally for a specific model family).
        
        Args:
            model_family: Optional filter
        """
        if model_family:
            # Get rowids to delete
            cursor = self.conn.execute(
                "SELECT rowid FROM episode_meta WHERE model_family = ?"
            )
            rowids = [r[0] for r in cursor.fetchall()]
            
            # Delete from FTS5
            self.conn.execute("DELETE FROM run_episodes WHERE rowid IN (" + ",".join("?" * len(rowids)) + ")", rowids)
            
            # Delete from metadata
            self.conn.execute("DELETE FROM episode_meta WHERE model_family = ?", (model_family,))
        else:
            self.conn.execute("DELETE FROM run_episodes")
            self.conn.execute("DELETE FROM episode_meta")
        
        self.conn.commit()
        logger.info("Cleared memory%s", f" for {model_family}" if model_family else "")
    
    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.debug("MemoryStore connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Singleton for easy import
# =============================================================================

_memory_store: Optional[MemoryStore] = None


def get_memory_store(db_path: Optional[Path] = None) -> MemoryStore:
    """
    Get or create the singleton MemoryStore instance.
    
    Args:
        db_path: Optional custom database path
    
    Returns:
        MemoryStore instance
    """
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore(db_path)
    return _memory_store


def record_event(event: Dict[str, Any]):
    """
    Convenience function to record an event to memory.
    
    Args:
        event: Event dictionary
    """
    store = get_memory_store()
    store.add_episode(event)
