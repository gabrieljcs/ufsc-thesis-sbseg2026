#!/usr/bin/env bash
set -euo pipefail

uv run thesis-eval model-status --model sagui_7b
uv run thesis-eval generate-targets \
  --translations outputs/pilot/translations_por.blaser.jsonl \
  --model sagui_7b \
  --profile macbook \
  --backend transformers \
  --model-path assets/models/sagui-7b-instruct-v0.1 \
  --max-tokens 64 \
  --stream \
  --output outputs/pilot/target_generations_por.macbook_smoke.jsonl
uv run thesis-eval score-strongreject \
  --generations outputs/pilot/target_generations_por.macbook_smoke.jsonl \
  --evaluator mock \
  --output outputs/pilot/strongreject_scores_por.macbook_smoke.jsonl
