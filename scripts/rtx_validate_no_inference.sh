#!/usr/bin/env bash
set -euo pipefail

docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
docker run --rm --gpus all \
  -v "$PWD":/work/eval \
  -w /work/eval \
  thesis-eval:gpu thesis-eval validate-config
docker run --rm --gpus all thesis-eval:gpu nvidia-smi

echo "RTX Docker validation completed without running model inference."
