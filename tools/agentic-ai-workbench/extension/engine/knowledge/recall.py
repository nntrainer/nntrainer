"""
Recall API for agents to query the RAG memory store.

This module provides a simple interface for agents to search
prior pipeline runs for similar errors, fixes, and patterns.

Usage:
    from knowledge.recall import recall, recall_errors, recall_success_patterns
    
    # Search for similar errors
    prior_errors = recall("compile error undefined reference", model="Qwen3")
    
    # Get error patterns for a model family
    patterns = recall_errors("Qwen3", error_type="missing_include")
    
    # Get success patterns
    successes = recall_success_patterns("Llama3")
"""
from typing import Any, Dict, List, Optional
from .memory_store import get_memory_store


def recall(
    query: str,
    model: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search memory for similar prior runs.
    
    Args:
        query: Search query (keywords)
        model: Model name (e.g., "Qwen3-1.8B") or family (e.g., "Qwen3")
        event_type: Filter by event type ("compile_error", "build_error", etc.)
        limit: Max results to return
    
    Returns:
        List of matching episodes with metadata
    
    Example:
        >>> recall("compile error", model="Qwen3", limit=3)
        [
            {
                "model_name": "Qwen3-1.8B",
                "error_msg": "undefined reference to nntrainer::Layer",
                "fix_applied": "Added LINK_LIBRARIES(nntrainer)",
                "success": True,
                ...
            },
            ...
        ]
    """
    memory = get_memory_store()
    
    # Extract model family if full model name provided
    model_family = None
    if model:
        model_family = model.split("-")[0]
    
    return memory.search(
        query=query,
        model_family=model_family,
        event_type=event_type,
        limit=limit
    )


def recall_errors(
    model: str,
    error_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get error patterns for a model family.
    
    Args:
        model: Model name or family (e.g., "Qwen3" or "Qwen3-1.8B")
        error_type: Optional filter (e.g., "undefined_reference", "missing_include")
        limit: Max results
    
    Returns:
        List of error patterns with occurrence counts and fixes
    
    Example:
        >>> recall_errors("Qwen3")
        [
            {
                "error_msg": "undefined reference to nntrainer::Layer",
                "error_type": "undefined_reference",
                "fix_applied": "Added LINK_LIBRARIES(nntrainer)",
                "occurrence_count": 5,
                ...
            },
            ...
        ]
    """
    memory = get_memory_store()
    model_family = model.split("-")[0]
    
    return memory.get_error_patterns(
        model_family=model_family,
        error_type=error_type,
        limit=limit
    )


def recall_success_patterns(
    model: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get successful run patterns for a model family.
    
    Args:
        model: Model name or family
        limit: Max results
    
    Returns:
        List of successful runs
    
    Example:
        >>> recall_success_patterns("Llama3")
        [
            {
                "model_name": "Llama3-8B",
                "content": "Pipeline completed successfully...",
                "duration_sec": 120.5,
                "tokens_used": 5000,
                ...
            },
            ...
        ]
    """
    memory = get_memory_store()
    model_family = model.split("-")[0]
    
    return memory.get_success_patterns(
        model_family=model_family,
        limit=limit
    )


def get_similar_runs(
    model: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get prior runs from the same model family.
    
    Args:
        model: Model name (e.g., "Qwen3-1.8B")
        limit: Max results
    
    Returns:
        List of prior runs
    
    Example:
        >>> get_similar_runs("Qwen3-1.8B")
        [
            {
                "model_name": "Qwen3-4B",
                "event_type": "pipeline_complete",
                "success": True,
                "duration_sec": 180.2,
                ...
            },
            ...
        ]
    """
    memory = get_memory_store()
    return memory.get_similar_runs(model=model, limit=limit)


def get_memory_stats(model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get memory store statistics.
    
    Args:
        model: Optional model family filter
    
    Returns:
        Dictionary with stats
    
    Example:
        >>> get_memory_stats("Qwen3")
        {
            "total_episodes": 25,
            "successful_runs": 20,
            "failed_runs": 5,
            "avg_duration": 150.5,
            "first_run": "2024-01-01T00:00:00",
            "last_run": "2024-01-15T00:00:00"
        }
    """
    memory = get_memory_store()
    model_family = model.split("-")[0] if model else None
    return memory.get_stats(model_family=model_family)


def format_recall_context(
    query: str,
    model: Optional[str] = None,
    limit: int = 3
) -> str:
    """
    Format recall results as context for LLM prompts.
    
    Args:
        query: Search query
        model: Model name/family
        limit: Max results
    
    Returns:
        Formatted string for injection into LLM prompt
    
    Example:
        >>> print(format_recall_context("compile error", "Qwen3"))
        ## Prior Run Insights for Qwen3
        
        ### Run 1: Qwen3-1.8B (Success)
        Error: undefined reference to nntrainer::Layer
        Fix: Added LINK_LIBRARIES(nntrainer)
        
        ### Run 2: Qwen3-4B (Failed)
        Error: missing include: rms_norm.h
        Fix: (none)
        ...
    """
    results = recall(query, model=model, limit=limit)
    
    if not results:
        return f"No prior runs found for '{query}'" + (f" (model: {model})" if model else "")
    
    lines = [f"## Prior Run Insights" + (f" for {model}" if model else "")]
    
    for i, r in enumerate(results, 1):
        status = "✅ Success" if r.get("success") else "❌ Failed"
        lines.append(f"\n### Run {i}: {r.get('model_name', 'unknown')} ({status})")
        
        if r.get("error_msg"):
            lines.append(f"Error: {r['error_msg'][:200]}")
        
        if r.get("fix_applied"):
            lines.append(f"Fix: {r['fix_applied']}")
        
        if r.get("event_type"):
            lines.append(f"Event: {r['event_type']}")
    
    return "\n".join(lines)


# =============================================================================
# Convenience functions for specific use cases
# =============================================================================

def recall_compile_errors(model: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get prior compile errors for a model family."""
    return recall("compile error", model=model, event_type="compile_error", limit=limit)


def recall_build_errors(model: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get prior build errors for a model family."""
    return recall("build error", model=model, event_type="build_error", limit=limit)


def recall_fixes(model: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get prior fixes applied for a model family."""
    results = recall("", model=model, limit=limit * 3)
    return [r for r in results if r.get("fix_applied")][:limit]


def recall_by_agent(agent_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get events from a specific agent."""
    memory = get_memory_store()
    # Search all events from this agent
    return memory.search(query="*", limit=limit)  # Would need agent_name filter in search()


# =============================================================================
# Semantic Search and Cross-Model Learning
# =============================================================================

def semantic_recall(
    query: str,
    model: Optional[str] = None,
    limit: int = 5,
    use_semantic: bool = True
) -> List[Dict[str, Any]]:
    """
    Search memory with semantic similarity (hybrid: FTS5 + embeddings).
    
    Args:
        query: Search query
        model: Model name/family
        limit: Max results
        use_semantic: Enable semantic re-ranking
    
    Returns:
        List of matching episodes, re-ranked by semantic similarity
    """
    from .memory_store import _get_embedder
    
    memory = get_memory_store()
    model_family = model.split("-")[0] if model else None
    
    return memory.semantic_search(
        query=query,
        model_family=model_family,
        limit=limit,
        use_semantic=use_semantic
    )


def get_cross_model_learnings(
    source_model: str,
    target_model: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find transferable patterns between model families.
    
    Args:
        source_model: Source model name/family (e.g., "Qwen2")
        target_model: Target model name/family (e.g., "Qwen3")
        limit: Max results
    
    Returns:
        List of patterns from source that may apply to target
    """
    memory = get_memory_store()
    source_family = source_model.split("-")[0]
    target_family = target_model.split("-")[0]
    
    return memory.get_cross_model_patterns(
        source_model_family=source_family,
        target_model_family=target_family,
        limit=limit
    )


def get_model_insights(model: str) -> Dict[str, Any]:
    """
    Get comprehensive insights for a model family.
    
    Args:
        model: Model name/family
    
    Returns:
        Dictionary with stats, errors, success patterns, and cross-model learnings
    """
    memory = get_memory_store()
    model_family = model.split("-")[0]
    
    return memory.get_model_family_insights(model_family)
