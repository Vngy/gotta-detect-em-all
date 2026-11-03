#!/bin/sh
# Runs once `detect --all` finishes: retrain probe on all boxes, re-evaluate, then discover unlisted cameos.
set -e
cd "$(dirname "$0")/.."
F='Redirect|Loading weights|Warning'
while pgrep -f "gde.detect --all" >/dev/null; do sleep 60; done
uv run python -m gde.train_probe 2>&1 | grep -vE "$F" > data/logs/probe-v2.log
for d in en-cameo ja-151; do
  uv run python -m gde.evaluate $d --n 150 --probe 2>&1 | grep -vE "$F" > data/logs/eval-$d-probe-v2.log
done
uv run python -m gde.discover 2>&1 | grep -E --line-buffered '^\[|cards to|Error' > data/logs/discover.log
