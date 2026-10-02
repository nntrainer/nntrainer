#!/usr/bin/env python3
"""
CLI script for querying the RAG memory store.
Used by extension.js to provide search functionality to the webview UI.

Usage:
    python memory_search_cli.py --query "compile error" --model-family Qwen3 --limit 10
"""
import argparse
import json
import sys
import os

# Add parent directory to path so we can import knowledge module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_store import get_memory_store


def main():
    parser = argparse.ArgumentParser(description="Search RAG memory store")
    parser.add_argument("--query", "-q", type=str, default="",
                        help="Search query (keywords)")
    parser.add_argument("--model-family", "-m", type=str, default=None,
                        help="Filter by model family (e.g., Qwen3, Llama)")
    parser.add_argument("--event-type", "-e", type=str, default=None,
                        help="Filter by event type (e.g., compile_error, build_error)")
    parser.add_argument("--limit", "-l", type=int, default=10,
                        help="Max results to return")
    parser.add_argument("--stats", "-s", action="store_true",
                        help="Show statistics instead of search results")
    
    args = parser.parse_args()
    
    try:
        memory = get_memory_store()
        
        if args.stats:
            # Show statistics
            stats = memory.get_stats(args.model_family)
            output = {
                "type": "stats",
                "model_family": args.model_family or "all",
                "data": stats
            }
            print(json.dumps(output))
            return
        
        # Perform search
        results = memory.search(
            query=args.query,
            model_family=args.model_family,
            event_type=args.event_type,
            limit=args.limit
        )
        
        # Format results for JSON output
        output = {
            "type": "results",
            "query": args.query,
            "model_family": args.model_family,
            "event_type": args.event_type,
            "count": len(results),
            "results": results
        }
        
        print(json.dumps(output))
        
    except Exception as e:
        # Output error as JSON
        error_output = {
            "type": "error",
            "message": str(e)
        }
        print(json.dumps(error_output))
        sys.exit(1)


if __name__ == "__main__":
    main()
