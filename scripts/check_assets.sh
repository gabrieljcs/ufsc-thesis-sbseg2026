#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- strongreject nllb_200 blaser_2_0_qe multijail belebele sagui_7b llamantino_2_ultrachat_7b llamantino_anita_8b gpt_sw3 ai_sweden_llama3_8b bggpt_7b bggpt_gemma_9b
fi

for asset in "$@"; do
  echo "== ${asset} =="
  uv run thesis-eval asset-status --name "${asset}"
  uv run thesis-eval verify-assets --name "${asset}"
done
