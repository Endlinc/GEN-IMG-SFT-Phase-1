#!/usr/bin/env bash
# Run every offline test. No API key, no GPU, no network, no real images needed.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0

for t in test_caption_pipeline.py; do
  echo "############ $t ############"
  python3 "$HERE/$t" || rc=1
  echo
done

if [ $rc -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $rc
