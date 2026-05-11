# thesis-eval

Reproducible evaluation package for the thesis *Multilingual Jailbreaks in Language-Specific Large Language Models*. It runs the empirical pipeline end to end: import StrongREJECT prompts, translate them with NLLB-200, audit the translations, generate target-model responses, score with StrongREJECT, attach URIEL+, BELEBELE, and tokenizer diagnostics, freeze the analysis dataset, and export the Results chapter tables.

The active design has eight language-specific paired targets (four weak/strong pairs across Portuguese, Italian, Swedish, and Bulgarian) plus one English-aligned reference baseline (`llama3_1_8b_reference`,`meta-llama/Llama-3.1-8B-Instruct`). The reference baseline is descriptive only: it is excluded from paired-design GLMMs, weak/strong slope-retention checks, and the four-pair falsification matrix.

## Models

Seven open-weight Hugging Face checkpoints are downloaded into `assets/models/`. Sabiá-3 is served through the Maritaca API.

| ID | Aligned language | Access | Asset / provider |
| --- | --- | --- | --- |
| `sagui_7b` | Portuguese | open weight | `OliveiraJLT/Sagui-7B-Instruct-v0.1` |
| `sabia_3` | Portuguese | API | Maritaca `sabia-3` |
| `llamantino_2_ultrachat_7b` | Italian | open weight | `swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA` |
| `llamantino_anita_8b` | Italian | open weight | `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` |
| `gpt_sw3` | Swedish | open weight | `AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct` |
| `ai_sweden_llama3_8b` | Swedish | open weight | `AI-Sweden-Models/Llama-3-8B-instruct` |
| `bggpt_7b` | Bulgarian | open weight | `INSAIT-Institute/BgGPT-7B-Instruct-v0.2` |
| `bggpt_gemma_9b` | Bulgarian | open weight | `INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0` |
| `llama3_1_8b_reference` | English | open weight | `meta-llama/Llama-3.1-8B-Instruct` (reference baseline) |

Attack-language inventory (13 languages):

```text
ara bul dan eng fin ita nor por rus spa swa swe ukr
```

## Layout

```text
eval/
├── configs/        models, languages, runtime profiles, asset specs
├── data/           StrongREJECT import and URIEL+ matrix
├── assets/         local download cache, not part of the Git release
├── outputs/        per-stage artifacts and the frozen dataset
├── scripts/        smoke wrappers and LaTeX exporters
├── src/thesis_eval/
└── tests/
```

Empty scaffold directories are not tracked. Pipeline commands create their
output directories as needed.

The GitHub repository stores the largest JSONL artifacts as compressed archives
to stay below GitHub's file-size limits. `outputs/frozen_jsonl_artifacts.zip`
contains `outputs/dataset_frozen.jsonl` and
`outputs/scored/all_strongreject_scores.jsonl`; `outputs/belebele_predictions_jsonl.zip`
contains `outputs/spec/all_belebele_predictions.jsonl`. A local Parquet mirror,
`outputs/dataset_frozen.parquet`, may be generated for faster analysis and can
be included in an archived Zenodo package, but it is ignored by Git to avoid
binary churn.

## Data Sources

The release combines original experiment outputs with externally sourced
benchmarks, model metadata, and derived diagnostics. The external sources are
downloaded or imported by the commands in the pipeline rather than manually
edited.

| Source | Role in this package | Stored artifacts | Notes |
| --- | --- | --- | --- |
| StrongREJECT | Primary harmful-prompt source and unsafe-compliance scoring protocol | `data/raw/strongreject/prompts.jsonl`, `outputs/scored/`, `outputs/frozen_jsonl_artifacts.zip` | The local prompt import contains 313 forbidden prompts with intent metadata. Responses are scored after English response back-translation. |
| NLLB-200 | Controlled prompt translation and response back-translation | `outputs/translations/`, `outputs/backtranslated/` | The final run uses one fixed NLLB-200 translation path across models and languages. Prompt-side round-trip text is retained as a drift diagnostic. |
| BLASER 2.0 QE | Reference-free translation quality estimation | `outputs/translations/*.qc.jsonl`, `outputs/tables/translation_qc_arithmetic.csv` | BLASER flags feed the XSTS audit queue; they are not used as model-safety labels. |
| XSTS-style audit | Translation acceptance/rejection for flagged prompt translations | `outputs/audit/`, `outputs/translations/*.audit.jsonl` | In the frozen run, 148 flagged rows were reviewed, 100 retained, and 48 excluded before generation. |
| BELEBELE | Benign multilingual reading-comprehension proxy for IF/CONS/SPEC | `outputs/spec/`, `outputs/tables/belebele_scores.csv` | Used as a specialization proxy, not as an open-ended benign generation score. |
| URIEL+ | Typological distance between aligned and attack languages | `data/uriel_plus/distance_matrix.csv`, `outputs/tables/closest_farthest_languages.csv` | The main distance metric is featural angular distance in `[0, 1]`. |
| Target model configs | Eight paired target models plus one English reference baseline | `configs/models.yaml`, `configs/assets.yaml`, `outputs/generations/` | Open-weight model weights are not committed; Sabiá-3 is API-served and includes provider metadata in run rows. |
| StrongREJECT judge calls | Automated response scoring | `outputs/scored/`, `outputs/tables/results_coverage.csv` | Judge-side technical failures are explicit unscored rows, not safe responses. |
| Redacted usage exports | Operational cost accounting for the appendix | `usage/` | Aggregate billing/currency files with provider project, organization, user, email, and API-key identifiers removed. |
| MultiJail | Planned external validation benchmark | `configs/assets.yaml` only in this frozen release | No scored MultiJail panel is present in the final thesis outputs. |

## Frozen Release Contents

The checked final panel is:

```text
313 StrongREJECT source prompts
13 attack languages
8 paired target models + 1 English reference baseline
48 audited prompt-language translations excluded before generation
36,189 frozen model-language-prompt rows
78 excluded/unscored run rows inside the frozen dataset
117 generation files, 117 response back-translation files, 117 scored files
9 BELEBELE prediction files and 117 model-language SPEC rows
```

The 48 translation exclusions are recorded in
`outputs/translations/all_audited_translations.jsonl` and intentionally do not
produce model-level generation rows. The 78 frozen-dataset exclusions are
run-level failures after generation: 75 empty model outputs and three
StrongREJECT judge safety-filter failures.

For safety, the artifact contains harmful-prompt and model-output text because
those fields are needed to audit scoring and translation behavior. Do not mirror
the dataset in contexts where dual-use jailbreak material is inappropriate.

## Install

Two console scripts ship with the package:

| Script | Where it runs | Use it for |
| --- | --- | --- |
| `thesis-eval` | host shell, native | downloads, asset checks, audit CSVs, analysis |
| `thesis-eval-gpu` | host shell, re-execs in the GPU container | anything that needs CUDA, vLLM, NLLB, or BLASER |

### Docker GPU image (recommended)

PyPI's default `torch` wheel is CPU-only on Windows; vLLM has no Windows wheel; `fairseq2` and `sonar-space` are gated to non-Windows. The Docker image avoids all three, builds on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, pins `torch==cu128`, and ships Linux `vllm`.

```sh
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
docker run --rm --gpus all thesis-eval:gpu nvidia-smi
```

`thesis-eval-gpu` ([`src/thesis_eval/gpu_runner.py`](src/thesis_eval/gpu_runner.py)) is a thin wrapper that forwards arguments into the container with the right `docker run` flags (bind mount, GPU passthrough, `--env-file`). The bind mount makes everything written under `assets/`, `data/`, and `outputs/` survive container teardown. Override the image tag with `THESIS_EVAL_GPU_IMAGE`.

Credentials at the repo root in `.env`, one `KEY=VALUE` per line:

```text
HF_TOKEN=...
OPENAI_API_KEY=...
MARITACA_API_KEY=...
```

`thesis-eval-gpu` injects these into the container; for host-side commands, also export them in your shell.

### Native installs

Apple Silicon smoke tests with Transformers on MPS:

```sh
uv sync --extra mac --extra qc --extra scoring --extra analysis
```

Windows analysis-only (table export, dataset freeze, GLMM):

```sh
uv sync --extra analysis --extra scoring
```

The `[rtx]` and `[qc]` extras are unsupported on native Windows. Use the container for any command that needs CUDA, vLLM, or BLASER.

### URIEL+ standalone

`prepare-uriel` shells out to a separate uv script env because URIEL+'s TensorFlow Addons dependency has no Python 3.12 wheels:

```sh
uv sync --script scripts/compute_urielplus.py
```

### Validate

```sh
uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
uv run thesis-eval list-assets
```

`validate-config` should print
`OK: 8 paired models, 1 reference baseline, 13 languages`.

## Pipeline

Each stage writes per-language or per-model files, so reruns of the same command skip completed inputs. Pass `--force` to redo a step. The command examples below assume a bash/zsh shell from the `eval/` directory.

| # | Stage | Command(s) |
| --- | --- | --- |
| 1 | Datasets | `download-assets --name strongreject\|multijail\|belebele`, `verify-assets` |
| 2 | Translation models | `download-assets --name nllb_200_3_3b\|blaser_2_0_qe`, `prewarm-sonar` |
| 3 | Target models | `download-assets --name <model>`, `model-status --model <model>` |
| 4 | URIEL+ distances | `prepare-uriel` |
| 5 | Forward translation | `translate --engine nllb` |
| 6 | Round-trip + BLASER | `prompt-roundtrip`, `translation-qc --scorer blaser` |
| 7 | Audit | `export-audit-queue`, `judge-audit-queue` (Plan B) or human XSTS (Plan A), `import-audit`, `calibrate-blaser`, `reference-spbleu` |
| 8 | Generation | `generate-targets --backend vllm\|maritaca\|transformers` |
| 9 | Back-translate responses | `backtranslate-responses` |
| 10 | StrongREJECT scoring | `prepare-strongreject-batch`, `submit-openai-batch`, `shard-openai-batch-requests`, `fetch-openai-batch`, `ingest-strongreject-batch`, or the serial `score-strongreject` |
| 11 | BELEBELE | `predict-belebele`, `repair-belebele-predictions`, `compute-spec` |
| 12 | Benign controls (optional) | `score-benign-controls` |
| 13 | Tokenizer diagnostics | `attach-tokenizer-metrics` |
| 14 | Freeze | `build-dataset`, `export-results` |
| 15 | Tables | `export-report-tables`, `fit-glmm` |

### Restore Archived JSONL Artifacts

The GitHub release keeps the largest JSONL files zipped. To work from the
checked artifact without rerunning model inference:

```sh
unzip -n outputs/frozen_jsonl_artifacts.zip
unzip -n outputs/belebele_predictions_jsonl.zip

uv run thesis-eval export-report-tables \
  --rows outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables
uv run thesis-eval fit-glmm \
  --rows outputs/dataset_frozen.jsonl \
  --output outputs/tables/glmm_main_effects.csv

# The LaTeX exporter expects the full outputs/tables bundle listed below.
uv run python scripts/write_uriel_latex_assets.py
uv run python scripts/write_thesis_results_latex_assets.py
```

### Full Command Skeleton

Set the panel once:

```sh
LANGS=(ara bul dan eng fin ita nor por rus spa swa swe ukr)
MODELS=(sagui_7b sabia_3 llamantino_2_ultrachat_7b llamantino_anita_8b gpt_sw3 ai_sweden_llama3_8b bggpt_7b bggpt_gemma_9b llama3_1_8b_reference)
OPEN_WEIGHT_MODELS=(sagui_7b llamantino_2_ultrachat_7b llamantino_anita_8b gpt_sw3 ai_sweden_llama3_8b bggpt_7b bggpt_gemma_9b llama3_1_8b_reference)
```

Prepare configs, datasets, assets, and URIEL+ distances:

```sh
uv run thesis-eval validate-config
uv run thesis-eval import-strongreject --source github --pilot-size 2
uv run thesis-eval download-assets --name belebele --name multijail
uv run thesis-eval verify-assets --name strongreject --name belebele --name multijail

uv run thesis-eval-gpu download-assets --name nllb_200_3_3b --name blaser_2_0_qe
uv run thesis-eval-gpu verify-assets --name nllb_200_3_3b --name blaser_2_0_qe
uv run thesis-eval-gpu prewarm-sonar

for model in "${OPEN_WEIGHT_MODELS[@]}"; do
  uv run thesis-eval-gpu download-assets --name "$model"
  uv run thesis-eval model-status --model "$model"
done

uv run thesis-eval prepare-uriel --from-csv
```

Translate prompts, attach round-trip diagnostics, and run BLASER QC:

```sh
TARGET_ARGS=()
for lang in "${LANGS[@]}"; do
  TARGET_ARGS+=(--target-language "$lang")
done

uv run thesis-eval-gpu translate \
  --prompts data/raw/strongreject/prompts.jsonl \
  "${TARGET_ARGS[@]}" \
  --output-dir outputs/translations \
  --output-template '{lang}.raw.jsonl' \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda

RAW_ARGS=()
for lang in "${LANGS[@]}"; do
  RAW_ARGS+=(--translations "outputs/translations/${lang}.raw.jsonl")
done

uv run thesis-eval-gpu prompt-roundtrip \
  "${RAW_ARGS[@]}" \
  --output-dir outputs/translations \
  --output-template '{lang}.roundtrip.jsonl' \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda

ROUNDTRIP_ARGS=()
for lang in "${LANGS[@]}"; do
  ROUNDTRIP_ARGS+=(--translations "outputs/translations/${lang}.roundtrip.jsonl")
done

uv run thesis-eval-gpu translation-qc \
  "${ROUNDTRIP_ARGS[@]}" \
  --output-dir outputs/translations \
  --output-template '{lang}.qc.jsonl' \
  --scorer blaser \
  --device cuda
```

Create and import the translation audit decisions. The command below is the
Plan B LLM-judge path; for Plan A, edit the exported CSVs with human XSTS
decisions before `import-audit`.

```sh
for lang in "${LANGS[@]}"; do
  uv run thesis-eval export-audit-queue \
    --translations "outputs/translations/${lang}.qc.jsonl" \
    --output "outputs/audit/${lang}.xsts_queue.csv" \
    --audit-plan plan_b

  uv run thesis-eval judge-audit-queue \
    --audit-queue "outputs/audit/${lang}.xsts_queue.csv" \
    --output "outputs/audit/${lang}.xsts_decisions.csv" \
    --model gpt-4o

  uv run thesis-eval import-audit \
    --translations "outputs/translations/${lang}.qc.jsonl" \
    --audit-csv "outputs/audit/${lang}.xsts_decisions.csv" \
    --output "outputs/translations/${lang}.audit.jsonl"
done

cat outputs/translations/*.audit.jsonl > outputs/translations/all_audited_translations.jsonl
uv run thesis-eval calibrate-blaser \
  --audit-csv outputs/audit/ara.xsts_decisions.csv \
  --audit-csv outputs/audit/bul.xsts_decisions.csv \
  --audit-csv outputs/audit/dan.xsts_decisions.csv \
  --audit-csv outputs/audit/eng.xsts_decisions.csv \
  --audit-csv outputs/audit/fin.xsts_decisions.csv \
  --audit-csv outputs/audit/ita.xsts_decisions.csv \
  --audit-csv outputs/audit/nor.xsts_decisions.csv \
  --audit-csv outputs/audit/por.xsts_decisions.csv \
  --audit-csv outputs/audit/rus.xsts_decisions.csv \
  --audit-csv outputs/audit/spa.xsts_decisions.csv \
  --audit-csv outputs/audit/swa.xsts_decisions.csv \
  --audit-csv outputs/audit/swe.xsts_decisions.csv \
  --audit-csv outputs/audit/ukr.xsts_decisions.csv \
  --output outputs/audit/blaser_calibration.json
```

Generate target responses. Sabiá-3 uses the Maritaca API; all other models use
the GPU container and the selected runtime profile.

```sh
AUDIT_ARGS=()
for lang in "${LANGS[@]}"; do
  AUDIT_ARGS+=(--translations "outputs/translations/${lang}.audit.jsonl")
done

for model in "${MODELS[@]}"; do
  if [ "$model" = "sabia_3" ]; then
    uv run thesis-eval generate-targets \
      "${AUDIT_ARGS[@]}" \
      --model "$model" \
      --backend maritaca \
      --max-tokens 512 \
      --temperature 0.0 \
      --output-dir outputs/generations \
      --output-template '{model}_{lang}.jsonl'
  else
    uv run thesis-eval-gpu generate-targets \
      "${AUDIT_ARGS[@]}" \
      --model "$model" \
      --profile runpod_a100 \
      --backend vllm \
      --max-tokens 512 \
      --temperature 0.0 \
      --trust-remote-code \
      --output-dir outputs/generations \
      --output-template '{model}_{lang}.jsonl'
  fi
done
```

Back-translate responses and score with StrongREJECT. The serial scoring path
is simpler; the Batch path is better for a full run.

```sh
GEN_ARGS=()
for file in outputs/generations/*.jsonl; do
  GEN_ARGS+=(--generations "$file")
done

uv run thesis-eval-gpu backtranslate-responses \
  "${GEN_ARGS[@]}" \
  --output-dir outputs/backtranslated \
  --output-template '{stem}.backtranslated.jsonl' \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda

BT_ARGS=()
for file in outputs/backtranslated/*.jsonl; do
  BT_ARGS+=(--generations "$file")
done

uv run thesis-eval score-strongreject \
  "${BT_ARGS[@]}" \
  --output-dir outputs/scored \
  --output-template '{stem}.strongreject.jsonl' \
  --evaluator strongreject_rubric \
  --judge-model gpt-4o \
  --workers 4 \
  --resume

cat outputs/scored/*.backtranslated.strongreject.jsonl > outputs/scored/all_strongreject_scores.jsonl
```

For OpenAI Batch scoring instead of serial scoring:

```sh
mkdir -p outputs/batches/strongreject
uv run thesis-eval prepare-strongreject-batch \
  "${BT_ARGS[@]}" \
  --requests-output outputs/batches/strongreject/requests.jsonl \
  --manifest-output outputs/batches/strongreject/manifest.jsonl \
  --scored-output-dir outputs/scored \
  --scored-output-template '{stem}.strongreject.jsonl' \
  --judge-model gpt-4o

uv run thesis-eval shard-openai-batch-requests \
  --input outputs/batches/strongreject/requests.jsonl \
  --output-dir outputs/batches/strongreject/shards \
  --index-output outputs/batches/strongreject/shards/index.jsonl

uv run thesis-eval submit-openai-batch \
  --input outputs/batches/strongreject/shards/requests-000.jsonl \
  --output outputs/batches/strongreject/shards/requests-000.submit.json
uv run thesis-eval fetch-openai-batch \
  --batch-id <batch_id_from_submit_json> \
  --output outputs/batches/strongreject/shards/requests-000.status.json \
  --download-output outputs/batches/strongreject/shards/requests-000.output.jsonl \
  --download-errors outputs/batches/strongreject/shards/requests-000.errors.jsonl
uv run thesis-eval ingest-strongreject-batch \
  --manifest outputs/batches/strongreject/manifest.jsonl \
  --batch-output outputs/batches/strongreject/shards/requests-000.output.jsonl \
  --batch-errors outputs/batches/strongreject/shards/requests-000.errors.jsonl
```

Repeat the `submit-openai-batch`, `fetch-openai-batch`, and
`ingest-strongreject-batch` pattern for each shard, or concatenate downloaded
Batch output files before one ingest pass.

Run BELEBELE IF/CONS/SPEC:

```sh
for model in "${MODELS[@]}"; do
  if [ "$model" = "sabia_3" ]; then
    uv run thesis-eval predict-belebele \
      --model "$model" \
      --backend maritaca \
      --output "outputs/spec/${model}.belebele.jsonl"
  else
    uv run thesis-eval-gpu predict-belebele \
      --model "$model" \
      --profile runpod_a100 \
      --backend vllm \
      --trust-remote-code \
      --output "outputs/spec/${model}.belebele.jsonl"
  fi
done

cat outputs/spec/*.belebele.jsonl > outputs/spec/all_belebele_predictions.raw.jsonl
uv run thesis-eval repair-belebele-predictions \
  --predictions outputs/spec/all_belebele_predictions.raw.jsonl \
  --output outputs/spec/all_belebele_predictions.jsonl
uv run thesis-eval compute-spec \
  --predictions outputs/spec/all_belebele_predictions.jsonl \
  --output outputs/spec/belebele_spec.jsonl
```

Freeze the analysis dataset, export tables, and refresh the LaTeX assets:

```sh
uv run thesis-eval build-dataset \
  --scored outputs/scored/all_strongreject_scores.jsonl \
  --translations outputs/translations/all_audited_translations.jsonl \
  --spec-scores outputs/spec/belebele_spec.jsonl \
  --output outputs/dataset_frozen.jsonl \
  --allow-frozen-overwrite

uv run thesis-eval export-results \
  --rows outputs/dataset_frozen.jsonl \
  --output outputs/dataset_frozen.parquet \
  --allow-frozen-overwrite

uv run thesis-eval export-report-tables \
  --rows outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables
uv run thesis-eval fit-glmm \
  --rows outputs/dataset_frozen.jsonl \
  --output outputs/tables/glmm_main_effects.csv

# The LaTeX exporter expects the full outputs/tables bundle listed below.
uv run python scripts/write_uriel_latex_assets.py
uv run python scripts/write_thesis_results_latex_assets.py

zip -9 outputs/frozen_jsonl_artifacts.zip \
  outputs/dataset_frozen.jsonl \
  outputs/scored/all_strongreject_scores.jsonl
zip -9 outputs/belebele_predictions_jsonl.zip \
  outputs/spec/all_belebele_predictions.jsonl
```

If generation was run with `--skip-tokenizer-metrics`, repair each generation
file with `attach-tokenizer-metrics` before response back-translation.

## Audit plans

Two translation-audit paths are wired into `export-audit-queue`, `judge-audit-queue`, and `import-audit`. The LaTeX source picks one with the `\humanaudittrue` / `\humanauditfalse` flag in `ufscthesisx/main.tex`.

| Plan | When | Audit method |
| --- | --- | --- |
| Plan B | default, deadline path | uncalibrated frontier multilingual LLM-as-judge XSTS over flagged rows |
| Plan A | upgrade after human review | Tier A human XSTS for `ara bul ita por spa`; calibrated LLM XSTS for the rest |

Audit decisions are pooled by `(target_language, prompt_id)` and applied to every downstream model that consumes that prompt-language pair. Finnish and Swahili are flagged as caveat languages in exported queues.

Valid `audit_decision` values:

```text
pass     keep the translation
revise   use the edited translated_text from the CSV
exclude  drop the prompt-language pair from generation and analysis
```

Audit queues are written as UTF-8 with BOM. Reviewers must save with "CSV UTF-8 (Comma delimited)" in Excel; the plain CSV export uses cp1252 and mangles non-ASCII characters on the way back in.

## Outputs

```text
outputs/
├── translations/         {lang}.raw.jsonl, .roundtrip.jsonl, .qc.jsonl, .audit.jsonl
├── audit/                {lang}.xsts_queue.csv, .xsts_decisions.csv
├── generations/          {model}_{lang}.jsonl
├── backtranslated/       {model}_{lang}.backtranslated.jsonl
├── batches/strongreject/ optional OpenAI Batch requests and downloads
├── scored/               {model}_{lang}.backtranslated.strongreject.jsonl
├── spec/                 {model}.belebele.jsonl, belebele_spec.jsonl
├── tables/               CSV inputs for the thesis Results chapter
├── frozen_jsonl_artifacts.zip
└── belebele_predictions_jsonl.zip
```

`outputs/frozen_jsonl_artifacts.zip` and
`outputs/belebele_predictions_jsonl.zip` store their JSONL members using the
original repository paths.
`outputs/dataset_frozen.parquet` is an optional local mirror of the JSONL
dataset and is ignored by Git.

The thesis Results chapter consumes this table bundle:

```text
results_coverage.csv
asr_by_model_language.csv
crosslingual_asr_by_model_language.csv
belebele_appendix_examples.csv
belebele_scores.csv
closest_farthest_languages.csv
distance_asr_correlation.csv
spec_asr_correlation.csv
tokenizer_diagnostics.csv
reference_distance_curve.csv
counterfactual_safety_by_aligned_language.csv
translation_qc_arithmetic.csv
glmm_aggregated_cell_sensitivity.csv
glmm_collinearity.csv
glmm_spec_components.csv
glmm_strata_effects.csv
glmm_tokenizer_robustness.csv
prereg_distance_slope_retention.csv
prereg_falsification_summary.csv
glmm_main_effects.csv
```

`fit-glmm` and the slope-retention tables use only rows with `model_alignment_pole in {weak, strong}`; the reference baseline stays in the descriptive tables.

## Zenodo Packaging Notes

Recommended release contents:

- Source code, configs, tests, scripts, and documentation.
- `outputs/frozen_jsonl_artifacts.zip`,
  `outputs/belebele_predictions_jsonl.zip`, or the corresponding unpacked JSONL
  files plus the optional regenerated `outputs/dataset_frozen.parquet`.
- Final retained stage outputs under `outputs/translations/`,
  `outputs/audit/`, `outputs/generations/`, `outputs/backtranslated/`,
  `outputs/scored/`, `outputs/spec/`, and `outputs/tables/`.
- `data/uriel_plus/` and the StrongREJECT import if redistribution is allowed
  by the source license.
- Redacted aggregate cost metadata under `usage/`, if the cost-accounting
  appendix is included in the archived package.

Do not include local execution debris in the archival package:

- `.DS_Store`, `.venv/`, `__pycache__/`, `.tmp_strongreject_tests/`.
- `outputs/retry/`, `outputs/scored/redo/`, and any
  `*.strongreject.strongreject.jsonl` files; these are intermediate repair
  work products, not final panel artifacts.
- `outputs/pilot/` and `outputs/figures/`; these are local smoke-test or
  exploratory directories and are not part of the frozen artifact package.
- Raw provider billing exports. Keep unredacted provider exports outside the
  repository or under ignored paths such as `usage/raw/`.

## Validation

```sh
uv run python -m unittest

uv run thesis-eval asset-status --group datasets
uv run thesis-eval asset-status --group translation
uv run thesis-eval asset-status --group targets

uv run thesis-eval run-pilot --output-dir outputs/pilot
```

`run-pilot` exercises the full pipeline against mock data with no model inference; useful when validating a checkout on a host without GPUs.
