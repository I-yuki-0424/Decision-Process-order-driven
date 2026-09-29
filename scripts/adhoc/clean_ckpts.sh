#!/bin/bash
# delete checkpoint dirs of COMPLETED runs (summary.json/result.json present next to them) under output/arc_proposals
cd "$(dirname "$0")/../.."
while true; do
  for d in $(find output/arc_proposals -type d -name checkpoints 2>/dev/null); do
    p=$(dirname "$d")
    if [ -f "$p/summary.json" ] || ls "$p"/*.result.json >/dev/null 2>&1; then rm -rf "$d"; fi
  done
  sleep 600
done
