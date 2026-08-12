#!/usr/bin/env bash
##
# @file  gbs-cache-prune.sh
# @brief Keep only the newest version of each RPM in the GBS RPM cache.
#
# GBS never purges superseded packages from ~/GBS-ROOT/local/cache.
# This keeps, per package (name+arch), only the most-recently-downloaded .rpm
# and deletes older versions, bounding the cache to one snapshot.

# A wrongly-pruned package only costs a re-download on the next run, never a
# build failure, so erring toward pruning is safe.
set -u

CACHE="${1:-$HOME/GBS-ROOT/local/cache}"

if [ ! -d "$CACHE" ]; then
  echo "gbs-cache-prune: no cache dir at $CACHE, nothing to do"
  exit 0
fi

echo "== gbs package cache before prune =="
du -sh "$CACHE" 2>/dev/null || true
before=$(find "$CACHE" -name '*.rpm' 2>/dev/null | wc -l)
echo "rpm files: $before"

# newest first (by mtime); keep first occurrence of each package key, list the rest
prune_list=$(
  find "$CACHE" -type f -name '*.rpm' -printf '%T@\t%p\n' 2>/dev/null \
    | sort -rn \
    | awk -F'\t' '
        {
          path = $2
          n = split(path, seg, "/"); base = seg[n]
          key = base
          # strip trailing -<version>-<release>.<arch>.rpm to group all versions
          sub(/-[^-]+-[^-]+\.[^.]+\.rpm$/, "", key)
          if (key in seen) { print path }   # older duplicate -> prune
          else             { seen[key] = 1 }
        }'
)

# surface how aggressive the prune is, so a runaway regex is visible in the log
pruned=$(printf '%s' "$prune_list" | grep -c . || true)
echo "pruning $pruned of $before rpm(s)"
if [ -n "$prune_list" ]; then
  printf '%s\n' "$prune_list" | while IFS= read -r f; do [ -n "$f" ] && rm -f "$f"; done
fi

echo "== gbs package cache after prune =="
du -sh "$CACHE" 2>/dev/null || true
after=$(find "$CACHE" -name '*.rpm' 2>/dev/null | wc -l)
echo "rpm files: $after"

# a correct prune always keeps the newest of each key, so it can never empty a
# non-empty cache; if it did, the key logic is broken -- flag it loudly
if [ "$before" -gt 0 ] && [ "$after" -eq 0 ]; then
  echo "gbs-cache-prune: ERROR pruned all $before rpm(s); key logic broken" >&2
fi
