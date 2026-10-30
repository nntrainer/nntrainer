#!/usr/bin/env bash
##
# @file  gbs-cache-evict.sh
# @brief Delete superseded GBS RPM cache snapshots, keeping only the given key.
#
# The daily build saves a new date-keyed snapshot (gbs-cache-<arch>-<date>)
# each run, and PRs only restore the newest one. Old snapshots linger until
# GitHub's 7-day idle expiry, so ~7 copies per arch pile up and push the repo
# toward the 10GB Actions cache limit.

# Usage: gbs-cache-evict.sh <key prefix> <key to keep>
# Needs GH_TOKEN with actions:write and GITHUB_REF/GITHUB_REPOSITORY (set by Actions).
set -u

PREFIX="$1"
KEEP="$2"
REF="${GITHUB_REF:?GITHUB_REF must be set}"

case "$KEEP" in
  "$PREFIX"*) ;;
  *) echo "gbs-cache-evict: ERROR $KEEP does not start with $PREFIX" >&2; exit 1 ;;
esac

if ! keys=$(gh cache list -R "$GITHUB_REPOSITORY" --key "$PREFIX" --ref "$REF" \
    --limit 100 --json key --jq '.[].key'); then
  echo "::warning::gbs-cache-evict: failed to list $PREFIX caches, skipping eviction"
  exit 0
fi

# never evict when the fresh snapshot is missing (e.g. save upload failed),
# otherwise PRs would be left with no cache at all
if ! printf '%s\n' "$keys" | grep -qxF "$KEEP"; then
  echo "gbs-cache-evict: $KEEP not found on $REF, skipping eviction"
  exit 0
fi

printf '%s\n' "$keys" | while IFS= read -r k; do
  if [ -z "$k" ] || [ "$k" = "$KEEP" ]; then continue; fi
  echo "deleting $k ($REF)"
  gh cache delete -R "$GITHUB_REPOSITORY" "$k" --ref "$REF" \
    || echo "::warning::gbs-cache-evict: failed to delete $k"
done
