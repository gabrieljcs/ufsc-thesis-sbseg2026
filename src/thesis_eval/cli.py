from __future__ import annotations

import argparse
import gc
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from time import perf_counter
from pathlib import Path
from typing import Any

from thesis_eval.assets import asset_status, download_asset, load_assets, select_assets, verify_assets
from thesis_eval.analysis.dataset import build_frozen_rows
from thesis_eval.analysis.reporting import (
    asr_by_model_language,
    belebele_scores,
    closest_farthest_by_model,
    collinearity_diagnostics,
    counterfactual_safety_by_aligned_language,
    coverage_by_model,
    prereg_distance_slope_retention,
    prereg_falsification_summary,
    reference_distance_curve,
    spearman_tables,
    tokenizer_diagnostics,
)
from thesis_eval.analysis.glmm import fit_main_glmm
from thesis_eval.benchmarks.belebele import (
    attach_belebele_predictions,
    build_belebele_prediction_rows,
    build_belebele_prompt,
    compute_if_cons,
    load_belebele_rows,
    repair_belebele_predictions,
)
from thesis_eval.benchmarks.strongreject import import_strongreject, write_prompt_records
from thesis_eval.benchmarks.uriel import (
    compare_matrices,
    compute_urielplus_matrix,
    compute_urielplus_matrix_external,
    compute_urielplus_matrix_uv,
    load_distance_matrix,
    subset_matrix,
    write_distance_matrix,
    write_latex_table,
    write_run_config,
)
from thesis_eval.config import load_config
from thesis_eval.evaluation.benign import score_benign_rows
from thesis_eval.evaluation.strongreject import repair_strongreject_parse_failures, score_generation_rows
from thesis_eval.evaluation.strongreject_batch import (
    ingest_strongreject_batch_results,
    prepare_strongreject_batch_entries,
)
from thesis_eval.io import append_jsonl, read_csv_dicts, read_jsonl, write_csv_dicts, write_jsonl
from thesis_eval.metrics.tokenizer import attach_tokenizer_metrics, load_tokenizer
from thesis_eval.models.generation import (
    attach_outputs,
    build_generation_runner,
    build_generation_rows,
    describe_prompt_format,
    estimate_model_asset_state,
    generate_outputs,
    iter_generate_with_maritaca,
)
from thesis_eval.openai_batch import (
    create_batch,
    download_file_content,
    retrieve_batch,
    shard_batch_requests,
    upload_batch_input_file,
)
from thesis_eval.paths import DATA_DIR, OUTPUT_DIR, THESIS_ROOT, ensure_dir
from thesis_eval.pipeline import run_mock_pilot
from thesis_eval.progress import info, step
from thesis_eval.runtime import load_runtime_profiles, resolve_runtime_profile
from thesis_eval.translation.sonar import prewarm_sonar_text_encoder
from thesis_eval.translation.backtranslate import attach_prompt_roundtrip, attach_response_backtranslations
from thesis_eval.schema import validate_analysis_rows
from thesis_eval.translation.audit import AUDIT_COLUMNS, apply_audit_decisions, build_audit_queue
from thesis_eval.translation.calibration import calibrate_blaser_thresholds
from thesis_eval.translation.log import load_translation_logs, save_translation_logs
from thesis_eval.translation.llm_audit import judge_audit_rows
from thesis_eval.translation.pipeline import NllbTranslator, PromptRecord, apply_translation_qc, run_translation
from thesis_eval.translation.reference_spbleu import attach_reference_scores

_DEFAULT_LOCAL_GENERATION_CHECKPOINT_BATCH_SIZE = 32
_DEFAULT_VLLM_LOCAL_GENERATION_CHECKPOINT_BATCH_SIZE = 8


def cmd_validate_config(_: argparse.Namespace) -> None:
    cfg = load_config()
    paired = sum(1 for model in cfg.models.values() if model.get("alignment_pole") in {"weak", "strong"})
    reference = sum(1 for model in cfg.models.values() if model.get("analysis_role") == "reference_baseline")
    reference_label = "reference baseline" if reference == 1 else "reference baselines"
    print(f"OK: {paired} paired models, {reference} {reference_label}, {len(cfg.attack_languages)} languages")


def cmd_runtime_profiles(_: argparse.Namespace) -> None:
    profiles = load_runtime_profiles()
    for profile in profiles.values():
        print(f"{profile.name}: backend={profile.target_backend} device={profile.device} dtype={profile.dtype} - {profile.notes}")


def cmd_list_assets(_: argparse.Namespace) -> None:
    for asset in load_assets().values():
        repo = f" ({asset.repo_id})" if asset.repo_id else ""
        print(f"{asset.name}: kind={asset.kind} group={asset.group}{repo} - {asset.description}")


def cmd_asset_status(args: argparse.Namespace) -> None:
    selected = select_assets(names=args.name, groups=args.group)
    print(json.dumps([asset_status(asset) for asset in selected], indent=2))


def cmd_download_assets(args: argparse.Namespace) -> None:
    selected = select_assets(names=args.name, groups=args.group)
    info(f"Selected {len(selected)} asset(s): {', '.join(asset.name for asset in selected)}")
    results = [download_asset(asset, dry_run=args.dry_run, force=args.force) for asset in selected]
    print(json.dumps(results, indent=2))


def cmd_verify_assets(args: argparse.Namespace) -> None:
    selected = select_assets(names=args.name, groups=args.group)
    info(f"Verifying {len(selected)} asset(s): {', '.join(asset.name for asset in selected)}")
    print(json.dumps(verify_assets(selected, deep=args.deep), indent=2))


def cmd_prepare_uriel(args: argparse.Namespace) -> None:
    cfg = load_config()
    source = _resolve_uriel_source(Path(args.source_csv), required=args.from_csv)
    if args.from_csv:
        if source is None:
            raise FileNotFoundError("A source CSV is required when --from-csv is set")
        matrix = subset_matrix(load_distance_matrix(source), cfg.attack_languages)
        metadata: dict[str, Any] = {"source": "csv", "source_csv": str(source)}
        mode = "csv"
    else:
        if args.uriel_python:
            uriel_python = Path(args.uriel_python).expanduser()
            if not uriel_python.exists():
                raise FileNotFoundError(f"--uriel-python does not exist: {uriel_python}")
            matrix, metadata = compute_urielplus_matrix_external(
                uriel_python,
                cfg.attack_languages,
                distance_type=args.distance_type,
            )
        elif args.uriel_backend == "in-process":
            matrix, metadata = compute_urielplus_matrix(cfg.attack_languages, distance_type=args.distance_type)
        else:
            matrix, metadata = compute_urielplus_matrix_uv(
                cfg.attack_languages,
                distance_type=args.distance_type,
            )
        mode = "urielplus"
    output_csv = Path(args.output_csv)
    output_tex = Path(args.output_tex)
    write_distance_matrix(output_csv, matrix, cfg.attack_languages)
    write_latex_table(output_tex, matrix, cfg.attack_languages)
    if args.run_config_output:
        write_run_config(Path(args.run_config_output), metadata)
    comparison: dict[str, Any] = {}
    if source is not None:
        comparison = {
            "comparison_source_csv": str(source),
            **compare_matrices(matrix, load_distance_matrix(source)),
        }
    print(
        json.dumps(
            {
                "mode": mode,
                "distance_type": args.distance_type,
                "csv": str(output_csv),
                "latex": str(output_tex),
                "run_config": args.run_config_output,
                **comparison,
            },
            indent=2,
        )
    )


def _resolve_uriel_source(source: Path, required: bool = False) -> Path | None:
    if source.exists():
        return source
    candidates = [
        THESIS_ROOT / "pictures" / source.name,
        DATA_DIR / "uriel_plus" / source.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not required:
        return None
    searched = [str(source)] + [str(candidate) for candidate in candidates]
    raise FileNotFoundError(
        "URIEL+ CSV preparation expects an existing static distance-matrix CSV. "
        f"Searched: {', '.join(searched)}"
    )


def cmd_import_strongreject(args: argparse.Namespace) -> None:
    records = import_strongreject(
        source=args.source,
        input_path=Path(args.input) if args.input else None,
        limit=args.limit,
        small=args.small,
    )
    output = Path(args.output)
    write_prompt_records(output, records)
    if args.pilot_output:
        pilot_output = Path(args.pilot_output)
        write_prompt_records(pilot_output, records[: args.pilot_size])
    else:
        pilot_output = None
    print(
        json.dumps(
            {
                "records": len(records),
                "output": str(output),
                "pilot_records": min(args.pilot_size, len(records)) if pilot_output else 0,
                "pilot_output": str(pilot_output) if pilot_output else None,
                "source": args.source,
                "small": args.small,
            },
            indent=2,
        )
    )


def _load_prompt_records(path: Path) -> list[PromptRecord]:
    rows = read_jsonl(path)
    prompts: list[PromptRecord] = []
    for row in rows:
        prompts.append(
            PromptRecord(
                prompt_id=str(row["prompt_id"]),
                text=str(row["text"]),
                harmful_goal=str(row.get("harmful_goal", "pending_manual_intent_spec")),
                expected_output_form=str(row.get("expected_output_form", "other")),
                fixed_constraints=str(row.get("fixed_constraints", "pending_manual_intent_spec")),
                allowed_adaptation=str(row.get("allowed_adaptation", "fluency_only_no_semantic_change")),
            )
        )
    return prompts


def cmd_translate(args: argparse.Namespace) -> None:
    prompts = _load_prompt_records(Path(args.prompts))
    target_languages = list(args.target_language)
    if len(target_languages) > 1 and args.output:
        raise ValueError("Use --output-dir for multi-language translation; --output is only for a single target language.")
    if not args.output and not args.output_dir:
        raise ValueError("translate requires --output for one language or --output-dir for one/more languages.")

    pending: list[tuple[str, Path]] = []
    for target_language in target_languages:
        output_path = _translation_output_path(args, target_language)
        if _skip_existing(output_path, args.force, f"translate -> {target_language}"):
            continue
        pending.append((target_language, output_path))
    if not pending:
        info("translate: all target outputs already exist; nothing to do.")
        return

    translator = None
    if args.engine == "nllb" and any(language != "eng" for language, _ in pending):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    written: dict[str, str] = {}
    for target_language, output_path in pending:
        with step(f"translate file {args.prompts} -> {target_language}"):
            logs = run_translation(
                prompts,
                target_language,
                engine=args.engine,
                checkpoint=args.checkpoint,
                device=args.device,
                dtype=args.dtype,
                stream=args.stream,
                translator=translator,
            )
        save_translation_logs(output_path, logs)
        written[target_language] = str(output_path)
        print(f"Wrote {len(logs)} translation logs to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def _translation_output_path(args: argparse.Namespace, target_language: str) -> Path:
    if args.output:
        return Path(args.output)
    output_dir = ensure_dir(Path(args.output_dir))
    filename = str(args.output_template).format(lang=target_language, target_language=target_language)
    return output_dir / filename


def _skip_existing(output_path: Path, force: bool, what: str) -> bool:
    # Returns True when a non-empty output already exists and --force was not set.
    if force:
        return False
    if output_path.exists() and output_path.stat().st_size > 0:
        info(f"skip {what}: {output_path} already exists (use --force to overwrite)")
        return True
    return False


def _parse_metadata_pairs(items: list[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--metadata expects key=value entries, got {item!r}")
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata


def _batch_request_file_endpoint(path: Path) -> str:
    endpoint: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            current = str(payload.get("url") or "")
            if not current:
                raise ValueError(f"{path} line {line_number} is missing a Batch request url.")
            if endpoint is None:
                endpoint = current
            elif endpoint != current:
                raise ValueError(
                    f"{path} mixes Batch endpoints ({endpoint!r} and {current!r}); "
                    "OpenAI Batch requires one endpoint per input file."
                )
    if endpoint is None:
        raise ValueError(f"{path} does not contain any Batch requests.")
    return endpoint


def cmd_translation_qc(args: argparse.Namespace) -> None:
    translation_paths = [Path(path) for path in args.translations]
    if not args.output and not args.output_dir:
        raise ValueError("translation-qc requires --output or --output-dir.")
    if len(translation_paths) > 1 and args.output:
        raise ValueError("Use --output-dir for multi-file QC; --output is only for a single input file.")

    pass_threshold = args.pass_threshold
    fail_threshold = args.fail_threshold
    if pass_threshold is None:
        pass_threshold = 4.0 if args.scorer == "blaser" else 0.72
    if fail_threshold is None:
        fail_threshold = 3.0 if args.scorer == "blaser" else 0.55
    info(f"QC scorer={args.scorer} pass_threshold={pass_threshold} fail_threshold={fail_threshold}")

    pending: list[tuple[Path, Path, list[dict[str, object]]]] = []
    for translation_path in translation_paths:
        group_logs = load_translation_logs(translation_path)
        if args.output_dir:
            output_path = _language_output_path(
                args.output_dir,
                args.output_template,
                _single_target_language(group_logs),
                translation_path,
                model=None,
            )
        else:
            output_path = Path(args.output)
        if _skip_existing(output_path, args.force, f"QC {translation_path.name}"):
            continue
        pending.append((translation_path, output_path, group_logs))
    if not pending:
        info("translation-qc: all outputs already exist; nothing to do.")
        return

    blaser_scorer = None
    if args.scorer == "blaser":
        from thesis_eval.translation.blaser import BlaserScorer

        blaser_scorer = BlaserScorer(device=args.device)

    written: dict[str, str] = {}
    for translation_path, output_path, group_logs in pending:
        with step(f"translation QC {translation_path.name} ({len(group_logs)} records)"):
            qc_logs = apply_translation_qc(
                group_logs,
                pass_threshold=pass_threshold,
                fail_threshold=fail_threshold,
                scorer=args.scorer,
                device=args.device,
                blaser_scorer=blaser_scorer,
            )
        write_jsonl(output_path, qc_logs)
        written[str(output_path)] = str(len(qc_logs))
        print(f"Wrote {len(qc_logs)} QC records to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_prompt_roundtrip(args: argparse.Namespace) -> None:
    translation_paths = [Path(path) for path in args.translations]
    if not args.output and not args.output_dir:
        raise ValueError("prompt-roundtrip requires --output or --output-dir.")
    if len(translation_paths) > 1 and args.output:
        raise ValueError("Use --output-dir for multi-file round-trip; --output is only for a single input file.")

    pending: list[tuple[Path, Path, list[dict[str, object]]]] = []
    for translation_path in translation_paths:
        group_logs = load_translation_logs(translation_path)
        if args.output_dir:
            output_path = _language_output_path(
                args.output_dir,
                args.output_template,
                _single_target_language(group_logs),
                translation_path,
                model=None,
            )
        else:
            output_path = Path(args.output)
        if _skip_existing(output_path, args.force, f"round-trip {translation_path.name}"):
            continue
        pending.append((translation_path, output_path, group_logs))
    if not pending:
        info("prompt-roundtrip: all outputs already exist; nothing to do.")
        return

    translator = None
    if args.engine == "nllb" and any(
        str(log["target_language"]) != "eng" for _, _, group_logs in pending for log in group_logs
    ):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    written: dict[str, str] = {}
    for input_path, output_path, group_logs in pending:
        with step(f"prompt round-trip {input_path}"):
            updated = attach_prompt_roundtrip(
                group_logs,
                engine=args.engine,
                checkpoint=args.checkpoint,
                device=args.device,
                dtype=args.dtype,
                pass_threshold=args.pass_threshold,
                fail_threshold=args.fail_threshold,
                translator=translator,
            )
        write_jsonl(output_path, updated)
        written[str(output_path)] = str(len(updated))
        print(f"Wrote {len(updated)} round-trip records to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_prewarm_sonar(args: argparse.Namespace) -> None:
    with step("prewarm SONAR text encoder"):
        result = prewarm_sonar_text_encoder(Path(args.cache_dir) if args.cache_dir else None)
    print(json.dumps(result, indent=2))


def cmd_generate_targets(args: argparse.Namespace) -> None:
    if not args.output and not args.output_dir:
        raise ValueError("generate-targets requires --output or --output-dir.")
    cfg = load_config()
    profile = resolve_runtime_profile(args.profile) if args.profile else None
    model_cfg = cfg.models[args.model]
    default_backend = "maritaca" if model_cfg.get("access_mode") == "api" else "auto"
    backend = args.backend or (profile.target_backend if profile else default_backend)
    device = args.device or (profile.device if profile else "auto")
    dtype = args.dtype or (profile.dtype if profile else "auto")
    model_ref = _resolve_model_ref(args.model, args.model_path)
    local_batch_size = _resolve_local_generation_batch_size(args.local_batch_size, backend)
    gpu_batch_size = _resolve_gpu_batch_size(
        args.gpu_batch_size, backend, default=_DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_GENERATE
    )
    prompt_metadata = describe_prompt_format(model_ref, backend=backend)
    translation_paths = [Path(path) for path in args.translations]
    groups: list[tuple[Path, Path, list[dict[str, Any]]]] = []
    rows: list[dict[str, Any]] = []
    for translation_path in translation_paths:
        translations = load_translation_logs(translation_path)
        base_rows = build_generation_rows(
            translations,
            model=args.model,
            aligned_language=cfg.aligned_language[args.model],
            benchmark=args.benchmark,
        )
        if args.max_records is not None:
            base_rows = base_rows[: args.max_records]
        group_rows = _attach_generation_request_metadata(
            base_rows,
            backend=backend,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            model_ref=model_ref,
            prompt_metadata=prompt_metadata,
        )
        if args.output_dir:
            output_path = _language_output_path(
                args.output_dir,
                args.output_template,
                _single_attack_language(group_rows),
                translation_path,
                model=args.model,
            )
        else:
            output_path = Path(args.output)
        if _skip_existing(output_path, args.force, f"generate {args.model} <- {translation_path.name}"):
            continue
        groups.append((translation_path, output_path, group_rows))
        rows.extend(group_rows)
    if not groups:
        info("generate-targets: all outputs already exist; nothing to do.")
        return
    info(f"Generating {len(rows)} target responses with backend={backend} model={args.model} model_ref={model_ref}")
    info(
        "Prompt format "
        f"strategy={prompt_metadata['strategy']} "
        f"layout={prompt_metadata['message_layout']} "
        f"chat_template={prompt_metadata['uses_chat_template']} "
        f"system_prompt={prompt_metadata['system_prompt_id'] or 'none'}"
    )
    tokenizer_metric_map = _prepare_tokenizer_metric_map(
        rows,
        backend=backend,
        skip_tokenizer_metrics=args.skip_tokenizer_metrics,
        model=args.model,
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    distance_matrix = load_distance_matrix(Path(args.distance_csv))
    if args.output_dir:
        shared_generate_batch: Callable[[list[str]], list[Any]] | None = None
        if not args.mock_output and backend in {"auto", "transformers", "vllm"}:
            with step("initialize target model backend"):
                shared_generate_batch = build_generation_runner(
                    model_ref,
                    backend=backend,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=device,
                    dtype=dtype,
                    trust_remote_code=args.trust_remote_code,
                    stream=args.stream,
                    prompt_texts=[str(row["prompt_text"]) for row in rows],
                    gpu_batch_size=gpu_batch_size,
                    max_input_length=args.max_input_length,
                )
            info("Reusing initialized local backend across language files; the model will not reload between batches.")
            info(
                f"Local generation batch size={local_batch_size} (checkpoint), "
                f"gpu batch size={gpu_batch_size} for backend={backend}"
            )
        written: dict[str, str] = {}
        total_groups = len(groups)
        for group_index, (input_path, output_path, group_rows) in enumerate(groups, start=1):
            info(
                f"Language batch {group_index}/{total_groups}: "
                f"{output_path.name} prompts={len(group_rows)}"
            )
            partial_path = _partial_output_path(output_path)
            if backend == "maritaca" and not args.mock_output:
                count = _generate_maritaca_group_checkpointed(
                    group_rows,
                    output_path=output_path,
                    partial_path=partial_path,
                    model_ref=model_ref,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    distance_matrix=distance_matrix,
                    tokenizer_metric_map=tokenizer_metric_map,
                )
            else:
                count = _generate_group_checkpointed(
                    group_rows,
                    generate_batch=shared_generate_batch,
                    output_path=output_path,
                    partial_path=partial_path,
                    backend=backend,
                    model_ref=model_ref,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=device,
                    dtype=dtype,
                    trust_remote_code=args.trust_remote_code,
                    stream=args.stream,
                    mock_output=args.mock_output,
                    distance_matrix=distance_matrix,
                    tokenizer_metric_map=tokenizer_metric_map,
                    force=args.force,
                    local_batch_size=local_batch_size,
                    gpu_batch_size=gpu_batch_size,
                    max_input_length=args.max_input_length,
                )
            written[str(output_path)] = str(count)
            print(f"Wrote {count} target generations to {output_path}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            if group_index < total_groups:
                _release_inter_language_memory()
                info(f"Ready for next language batch {group_index + 1}/{total_groups}")
        if len(written) > 1:
            print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        shared_generate_batch = None
        if not args.mock_output and backend in {"auto", "transformers", "vllm"}:
            with step("initialize target model backend"):
                shared_generate_batch = build_generation_runner(
                    model_ref,
                    backend=backend,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=device,
                    dtype=dtype,
                    trust_remote_code=args.trust_remote_code,
                    stream=args.stream,
                    prompt_texts=[str(row["prompt_text"]) for row in rows],
                    gpu_batch_size=gpu_batch_size,
                    max_input_length=args.max_input_length,
                )
            info(
                f"Local generation batch size={local_batch_size} (checkpoint), "
                f"gpu batch size={gpu_batch_size} for backend={backend}"
            )
        output_path = Path(args.output)
        partial_path = _partial_output_path(output_path)
        count = _generate_group_checkpointed(
            rows,
            generate_batch=shared_generate_batch,
            output_path=output_path,
            partial_path=partial_path,
            backend=backend,
            model_ref=model_ref,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            stream=args.stream,
            mock_output=args.mock_output,
            distance_matrix=distance_matrix,
            tokenizer_metric_map=tokenizer_metric_map,
            force=args.force,
            local_batch_size=local_batch_size,
            gpu_batch_size=gpu_batch_size,
            max_input_length=args.max_input_length,
        )
        print(f"Wrote {count} target generations to {args.output}")


def cmd_predict_belebele(args: argparse.Namespace) -> None:
    if not args.output and not args.output_dir:
        raise ValueError("predict-belebele requires --output or --output-dir.")
    cfg = load_config()
    profile = resolve_runtime_profile(args.profile) if args.profile else None
    model_cfg = cfg.models[args.model]
    default_backend = "maritaca" if model_cfg.get("access_mode") == "api" else "auto"
    backend = args.backend or (profile.target_backend if profile else default_backend)
    device = args.device or (profile.device if profile else "auto")
    dtype = args.dtype or (profile.dtype if profile else "auto")
    model_ref = _resolve_model_ref(args.model, args.model_path)
    local_batch_size = _resolve_local_generation_batch_size(args.local_batch_size, backend)
    gpu_batch_size = _resolve_gpu_batch_size(
        args.gpu_batch_size, backend, default=_DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_BELEBELE
    )
    dataset_dir = _resolve_belebele_dataset_dir(args.dataset_dir)
    prompt_metadata = describe_prompt_format(model_ref, backend=backend)
    languages = list(args.language) if args.language else list(cfg.attack_languages)
    all_prompts: list[str] = []
    groups: list[tuple[str, Path, list[dict[str, Any]], list[str]]] = []
    rows: list[dict[str, Any]] = []
    for language in languages:
        dataset_rows = load_belebele_rows(dataset_dir, language, limit=args.limit)
        prediction_rows = build_belebele_prediction_rows(dataset_rows, model=args.model, language=language)
        prompts = [build_belebele_prompt(row) for row in dataset_rows]
        if args.output_dir:
            output_path = _language_output_path(
                args.output_dir,
                args.output_template,
                language,
                Path(f"{language}.jsonl"),
                model=args.model,
            )
            if _skip_existing(output_path, args.force, f"predict-belebele {args.model} <- {language}"):
                continue
            groups.append((language, output_path, prediction_rows, prompts))
        else:
            rows.extend(prediction_rows)
        all_prompts.extend(prompts)
    if not args.output_dir and not rows:
        info("predict-belebele: no rows selected; nothing to do.")
        return
    if args.output_dir and not groups:
        info("predict-belebele: all outputs already exist; nothing to do.")
        return
    info(
        f"Predicting BELEBELE with backend={backend} model={args.model} model_ref={model_ref} "
        f"languages={len(groups) if args.output_dir else len(languages)}"
    )
    info(
        "Prompt format "
        f"strategy={prompt_metadata['strategy']} "
        f"layout={prompt_metadata['message_layout']} "
        f"chat_template={prompt_metadata['uses_chat_template']} "
        f"system_prompt={prompt_metadata['system_prompt_id'] or 'none'}"
    )
    shared_generate_batch: Callable[[list[str]], list[Any]] | None = None
    if backend != "mock" and not args.mock_output and backend in {"auto", "transformers", "vllm"}:
        with step("initialize BELEBELE model backend"):
            shared_generate_batch = build_generation_runner(
                model_ref,
                backend=backend,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
                dtype=dtype,
                trust_remote_code=args.trust_remote_code,
                stream=False,
                prompt_texts=all_prompts,
                gpu_batch_size=gpu_batch_size,
                max_input_length=args.max_input_length,
            )
        info(
            f"BELEBELE local batch size={local_batch_size} (checkpoint), "
            f"gpu batch size={gpu_batch_size} for backend={backend}"
        )
    if args.output_dir:
        total_groups = len(groups)
        written: dict[str, str] = {}
        for group_index, (language, output_path, prediction_rows, prompts) in enumerate(groups, start=1):
            info(f"BELEBELE batch {group_index}/{total_groups}: {output_path.name} items={len(prediction_rows)}")
            if backend == "maritaca" and not args.mock_output:
                count = _predict_belebele_maritaca_group_checkpointed(
                    prediction_rows,
                    prompts=prompts,
                    output_path=output_path,
                    partial_path=_partial_output_path(output_path),
                    model_ref=model_ref,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    force=args.force,
                )
            else:
                count = _predict_belebele_group_checkpointed(
                    prediction_rows,
                    prompts=prompts,
                    generate_batch=shared_generate_batch,
                    output_path=output_path,
                    partial_path=_partial_output_path(output_path),
                    backend=backend,
                    model_ref=model_ref,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=device,
                    dtype=dtype,
                    trust_remote_code=args.trust_remote_code,
                    mock_output=args.mock_output,
                    force=args.force,
                    local_batch_size=local_batch_size,
                    gpu_batch_size=gpu_batch_size,
                    max_input_length=args.max_input_length,
                )
            written[str(output_path)] = str(count)
            print(f"Wrote {count} BELEBELE predictions to {output_path}")
        if len(written) > 1:
            print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))
        return

    prompts = all_prompts
    if backend == "maritaca" and not args.mock_output:
        count = _predict_belebele_maritaca_group_checkpointed(
            rows,
            prompts=prompts,
            output_path=Path(args.output),
            partial_path=_partial_output_path(Path(args.output)),
            model_ref=model_ref,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force=args.force,
        )
    else:
        count = _predict_belebele_group_checkpointed(
            rows,
            prompts=prompts,
            generate_batch=shared_generate_batch,
            output_path=Path(args.output),
            partial_path=_partial_output_path(Path(args.output)),
            backend=backend,
            model_ref=model_ref,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            mock_output=args.mock_output,
            force=args.force,
            local_batch_size=local_batch_size,
            gpu_batch_size=gpu_batch_size,
            max_input_length=args.max_input_length,
        )
    print(f"Wrote {count} BELEBELE predictions to {args.output}")


def cmd_repair_belebele_predictions(args: argparse.Namespace) -> None:
    dataset_dir = _resolve_belebele_dataset_dir(args.dataset_dir)
    rows = read_jsonl(Path(args.predictions))
    repaired = repair_belebele_predictions(rows, dataset_dir=dataset_dir)
    write_jsonl(Path(args.output), repaired)
    print(f"Wrote {len(repaired)} repaired BELEBELE predictions to {args.output}")


def cmd_attach_tokenizer_metrics(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.generations))
    if not rows:
        write_jsonl(Path(args.output), [])
        print(f"Wrote 0 generation rows with tokenizer metrics to {args.output}")
        return
    row_models = {str(row["model"]) for row in rows}
    if args.model:
        unexpected = row_models.difference({args.model})
        if unexpected:
            raise ValueError(f"--model {args.model!r} does not match generation row models: {sorted(row_models)}")
        models = [args.model]
    else:
        models = sorted(row_models)
    if args.model_path and len(models) != 1:
        raise ValueError("--model-path can only be used when the generation file contains one model or --model is set.")

    output_rows = [dict(row) for row in rows]
    for model in models:
        tokenizer_ref = _resolve_tokenizer_ref(model, args.model_path if len(models) == 1 else None)
        model_indexes = [index for index, row in enumerate(output_rows) if str(row["model"]) == model]
        model_rows = [output_rows[index] for index in model_indexes]
        if tokenizer_ref is None:
            info(
                f"Skipping tokenizer metrics for model={model}; "
                "no local or Hugging Face tokenizer reference is configured."
            )
            continue
        info(f"Computing tokenizer metrics for {len(model_rows)} rows model={model} model_ref={tokenizer_ref}")
        with step(f"tokenizer metrics for {model}"):
            metric_rows = attach_tokenizer_metrics(
                model_rows,
                load_tokenizer(tokenizer_ref, trust_remote_code=args.trust_remote_code),
            )
        for index, metric_row in zip(model_indexes, metric_rows, strict=True):
            output_rows[index] = metric_row

    write_jsonl(Path(args.output), output_rows)
    print(f"Wrote {len(output_rows)} generation rows with tokenizer metrics to {args.output}")


def cmd_backtranslate_responses(args: argparse.Namespace) -> None:
    generation_paths = [Path(path) for path in args.generations]
    if not args.output and not args.output_dir:
        raise ValueError("backtranslate-responses requires --output or --output-dir.")
    if len(generation_paths) > 1 and args.output:
        raise ValueError(
            "Use --output-dir for multi-file back-translation; --output is only for a single input file."
        )

    pending: list[tuple[Path, Path, list[dict[str, Any]]]] = []
    for generation_path in generation_paths:
        rows = read_jsonl(generation_path)
        if args.output_dir:
            output_dir = ensure_dir(Path(args.output_dir))
            filename = str(args.output_template).format(stem=generation_path.stem)
            output_path = output_dir / filename
        else:
            output_path = Path(args.output)
        if _skip_existing(output_path, args.force, f"backtranslate {generation_path.name}"):
            continue
        pending.append((generation_path, output_path, rows))
    if not pending:
        info("backtranslate-responses: all outputs already exist; nothing to do.")
        return

    translator = None
    if args.engine == "nllb" and any(
        str(row.get("attack_language")) != "eng" for _, _, group_rows in pending for row in group_rows
    ):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    written: dict[str, str] = {}
    for generation_path, output_path, rows in pending:
        with step(f"backtranslate {generation_path.name} ({len(rows)} rows)"):
            output = attach_response_backtranslations(
                rows,
                engine=args.engine,
                checkpoint=args.checkpoint,
                device=args.device,
                dtype=args.dtype,
                translator=translator,
            )
        write_jsonl(output_path, output)
        written[str(output_path)] = str(len(output))
        print(f"Wrote {len(output)} generation rows with response back-translations to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_prepare_strongreject_batch(args: argparse.Namespace) -> None:
    generation_paths = [Path(path) for path in args.generations]
    requests_output = Path(args.requests_output)
    manifest_output = Path(args.manifest_output)
    scored_output_dir = ensure_dir(Path(args.scored_output_dir))

    for artifact_path in (requests_output, manifest_output):
        if artifact_path.exists():
            if not args.force:
                raise RuntimeError(
                    f"{artifact_path} already exists. Use --force to overwrite or pick a new path."
                )
            artifact_path.unlink()

    pending: list[tuple[Path, Path, list[dict[str, Any]]]] = []
    for generation_path in generation_paths:
        group_rows = read_jsonl(generation_path)
        output_path = scored_output_dir / str(args.scored_output_template).format(stem=generation_path.stem)
        if _skip_existing(output_path, args.force, f"prepare batch {generation_path.name}"):
            continue
        pending.append((generation_path, output_path, group_rows))
    if not pending:
        info("prepare-strongreject-batch: all scored outputs already exist; nothing to do.")
        return

    translator = None
    if args.backtranslate_engine == "nllb" and any(
        str(row.get("attack_language")) != "eng" and not str(row.get("model_output_backtranslated") or "").strip()
        for _, _, group_rows in pending
        for row in group_rows
    ):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    next_request_index = 0
    request_count = 0
    preset_count = 0
    manifest_count = 0
    for generation_path, output_path, group_rows in pending:
        with step(f"prepare batch {generation_path.name} ({len(group_rows)} rows)"):
            request_rows, manifest_rows, next_request_index = prepare_strongreject_batch_entries(
                group_rows,
                input_path=generation_path,
                output_path=output_path,
                judge_model=args.judge_model,
                custom_id_prefix=args.custom_id_prefix,
                next_request_index=next_request_index,
                backtranslate_engine=args.backtranslate_engine,
                translator=translator,
            )
        if request_rows:
            append_jsonl(requests_output, request_rows)
        append_jsonl(manifest_output, manifest_rows)
        request_count += len(request_rows)
        manifest_count += len(manifest_rows)
        preset_count += sum(1 for row in manifest_rows if row.get("custom_id") is None)

    summary = {
        "generation_files": len(pending),
        "manifest_rows": manifest_count,
        "prepared_requests": request_count,
        "preset_rows": preset_count,
        "requests_output": str(requests_output),
        "manifest_output": str(manifest_output),
    }
    if requests_output.exists():
        size_bytes = requests_output.stat().st_size
        summary["requests_bytes"] = size_bytes
        if size_bytes > 200 * 1024 * 1024:
            raise RuntimeError(
                "Prepared Batch input exceeds OpenAI's 200 MB Batch file limit. "
                "Split the generation files into multiple batch prepares."
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_submit_openai_batch(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    endpoint = _batch_request_file_endpoint(input_path)
    upload = upload_batch_input_file(input_path, api_base=args.api_base)
    metadata = _parse_metadata_pairs(args.metadata)
    batch = create_batch(
        input_file_id=str(upload["id"]),
        endpoint=endpoint,
        completion_window=args.completion_window,
        metadata=metadata or None,
        api_base=args.api_base,
    )
    payload = {
        "input_path": str(input_path),
        "input_file": upload,
        "batch": batch,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_shard_openai_batch_requests(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    requests = read_jsonl(input_path)
    shards = shard_batch_requests(
        requests,
        max_estimated_input_tokens=args.max_estimated_input_tokens,
        max_requests=args.max_requests,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or input_path.stem
    shard_summaries: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        shard_path = output_dir / f"{prefix}-{index:03d}.jsonl"
        if shard_path.exists() and not args.force:
            raise RuntimeError(f"{shard_path} already exists. Use --force to overwrite shard files.")
        write_jsonl(shard_path, shard["rows"])
        shard_summaries.append(
            {
                "path": str(shard_path),
                "request_count": shard["request_count"],
                "estimated_input_tokens": shard["estimated_input_tokens"],
            }
        )

    summary = {
        "input": str(input_path),
        "total_requests": len(requests),
        "shard_count": len(shards),
        "max_estimated_input_tokens": args.max_estimated_input_tokens,
        "max_requests": args.max_requests,
        "shards": shard_summaries,
    }
    index_path = Path(args.index_output) if args.index_output else output_dir / f"{prefix}-shards.json"
    if index_path.exists() and not args.force:
        raise RuntimeError(f"{index_path} already exists. Use --force to overwrite the shard index.")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_fetch_openai_batch(args: argparse.Namespace) -> None:
    batch = retrieve_batch(args.batch_id, api_base=args.api_base)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    downloads: dict[str, str] = {}
    if args.download_output and batch.get("output_file_id"):
        download_path = download_file_content(
            str(batch["output_file_id"]),
            Path(args.download_output),
            api_base=args.api_base,
        )
        downloads["batch_output"] = str(download_path)
    if args.download_errors and batch.get("error_file_id"):
        download_path = download_file_content(
            str(batch["error_file_id"]),
            Path(args.download_errors),
            api_base=args.api_base,
        )
        downloads["batch_errors"] = str(download_path)

    payload: dict[str, Any] = {"batch": batch, "downloads": downloads}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_ingest_strongreject_batch(args: argparse.Namespace) -> None:
    manifest_rows = read_jsonl(Path(args.manifest))
    batch_output_rows = read_jsonl(Path(args.batch_output))
    batch_error_rows = read_jsonl(Path(args.batch_errors)) if args.batch_errors else None
    grouped = ingest_strongreject_batch_results(
        manifest_rows,
        batch_output_rows,
        batch_error_rows=batch_error_rows,
    )
    written: dict[str, str] = {}
    for output_path_str, rows in grouped.items():
        output_path = Path(output_path_str)
        if output_path.exists() and not args.force:
            raise RuntimeError(
                f"{output_path} already exists. Use --force to overwrite ingested StrongREJECT outputs."
            )
        write_jsonl(output_path, rows)
        written[str(output_path)] = str(len(rows))
        print(f"Wrote {len(rows)} StrongREJECT-scored rows to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def _resolve_model_ref(model: str, model_path: str | None = None) -> str:
    cfg = load_config()
    assets = load_assets()
    model_asset = assets.get(model)
    asset_path = model_asset.local_dir if model_asset and model_asset.local_dir and model_asset.local_dir.exists() else None
    return model_path or (str(asset_path) if asset_path else str(cfg.models[model].get("hf_id") or cfg.models[model]["provider_model_id"]))


def _resolve_tokenizer_ref(model: str, model_path: str | None = None) -> str | None:
    cfg = load_config()
    assets = load_assets()
    model_asset = assets.get(model)
    asset_path = model_asset.local_dir if model_asset and model_asset.local_dir and model_asset.local_dir.exists() else None
    if model_path:
        return model_path
    if asset_path:
        return str(asset_path)
    hf_id = cfg.models[model].get("hf_id")
    return str(hf_id) if hf_id else None


def _partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")


def _release_inter_language_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_output_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("text", "") or "")
    return ""


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN guard
        return "0s"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _attach_generation_request_metadata(
    rows: list[dict[str, Any]],
    *,
    backend: str,
    max_tokens: int,
    temperature: float,
    model_ref: str,
    prompt_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["generation_backend"] = backend
        updated["generation_max_tokens"] = max_tokens
        updated["generation_temperature"] = temperature
        updated["generation_model_ref"] = model_ref
        updated["generation_prompt_strategy"] = prompt_metadata["strategy"]
        updated["generation_prompt_message_layout"] = prompt_metadata["message_layout"]
        updated["generation_prompt_uses_chat_template"] = prompt_metadata["uses_chat_template"]
        updated["generation_system_prompt_id"] = prompt_metadata["system_prompt_id"]
        output.append(updated)
    return output


def _resolve_belebele_dataset_dir(dataset_dir: str | None) -> Path:
    if dataset_dir:
        return Path(dataset_dir)
    asset = load_assets()["belebele"]
    if asset.local_dir is None:
        raise ValueError("The BELEBELE asset does not define a local_dir")
    return asset.local_dir


def _prepare_tokenizer_metric_map(
    rows: list[dict[str, Any]],
    *,
    backend: str,
    skip_tokenizer_metrics: bool,
    model: str,
    model_path: str | None,
    trust_remote_code: bool,
) -> dict[str, dict[str, Any]]:
    if skip_tokenizer_metrics:
        info("Skipping tokenizer metrics by request")
        return {}
    if backend == "mock":
        info("Skipping tokenizer metrics for mock generation")
        return {}
    tokenizer_ref = _resolve_tokenizer_ref(model, model_path)
    if tokenizer_ref is None:
        info(
            f"Skipping tokenizer metrics for model={model}; "
            "no local or Hugging Face tokenizer reference is configured."
        )
        return {}
    with step("tokenizer metrics precompute"):
        metric_rows = attach_tokenizer_metrics(
            rows,
            load_tokenizer(tokenizer_ref, trust_remote_code=trust_remote_code),
        )
    return {
        str(row["run_id"]): {
            "input_tokens": row.get("input_tokens"),
            "tokens_per_char": row.get("tokens_per_char"),
            "token_inflation": row.get("token_inflation"),
        }
        for row in metric_rows
    }


def _enrich_generation_rows_with_metrics(
    rows: list[dict[str, Any]],
    *,
    distance_matrix: dict[str, dict[str, float]],
    tokenizer_metric_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        aligned = str(row["aligned_language"])
        attack = str(row["attack_language"])
        prompt_text = str(row["prompt_text"])
        enriched.update(
            {
                "distance": distance_matrix[aligned][attack],
                "if_score": None,
                "cons_score": None,
                "spec_score": None,
                "input_tokens": None,
                "tokens_per_char": None,
                "token_inflation": None,
                "excluded": False,
                "exclusion_reason": None,
            }
        )
        if prompt_text:
            enriched["tokens_per_char"] = None
        metrics = tokenizer_metric_map.get(str(row["run_id"]))
        if metrics:
            enriched.update(metrics)
        output.append(enriched)
    return output


def _generate_group_checkpointed(
    group_rows: list[dict[str, Any]],
    *,
    generate_batch: Callable[[list[str]], list[Any]] | None,
    output_path: Path,
    partial_path: Path,
    backend: str,
    model_ref: str,
    max_tokens: int,
    temperature: float,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    stream: bool,
    mock_output: str | None,
    distance_matrix: dict[str, dict[str, float]],
    tokenizer_metric_map: dict[str, dict[str, Any]],
    force: bool,
    local_batch_size: int,
    gpu_batch_size: int = 1,
    max_input_length: int | None = None,
) -> int:
    if force and partial_path.exists():
        partial_path.unlink()
    completed_rows = read_jsonl(partial_path) if partial_path.exists() else []
    if len(completed_rows) > len(group_rows):
        raise ValueError(f"Checkpoint for {output_path.name} has more rows than expected")
    if len(completed_rows) == len(group_rows):
        info(f"Promoting complete checkpoint for {output_path.name} without regeneration")
        partial_path.replace(output_path)
        return len(completed_rows)
    if completed_rows:
        info(f"Resuming {output_path.name} from checkpoint rows={len(completed_rows)}/{len(group_rows)}")
    remaining_rows = group_rows[len(completed_rows) :]
    if not remaining_rows:
        raise RuntimeError(f"Checkpoint accounting for {output_path.name} is inconsistent")
    group_started = perf_counter()
    group_prompts_done = 0
    group_output_chars = 0
    for chunk_start in range(0, len(remaining_rows), local_batch_size):
        chunk_rows = remaining_rows[chunk_start : chunk_start + local_batch_size]
        chunk_offset = len(completed_rows) + chunk_start
        chunk_end = chunk_offset + len(chunk_rows)
        if mock_output:
            outputs = [mock_output for _ in chunk_rows]
            chunk_elapsed = 0.0
        else:
            chunk_started = perf_counter()
            with step(
                f"target model generation [{output_path.name}] "
                f"rows {chunk_offset + 1}-{chunk_end}/{len(group_rows)}"
            ):
                prompts = [str(row["prompt_text"]) for row in chunk_rows]
                if generate_batch is not None:
                    outputs = generate_batch(prompts)
                else:
                    outputs = generate_outputs(
                        prompts,
                        model_ref,
                        backend=backend,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        device=device,
                        dtype=dtype,
                        trust_remote_code=trust_remote_code,
                        stream=stream,
                        gpu_batch_size=gpu_batch_size,
                        max_input_length=max_input_length,
                    )
            chunk_elapsed = perf_counter() - chunk_started
            chunk_output_chars = sum(len(_extract_output_text(item)) for item in outputs)
            group_prompts_done += len(chunk_rows)
            group_output_chars += chunk_output_chars
            info(
                f"Pass speed [{output_path.name}] rows {chunk_offset + 1}-{chunk_end}: "
                f"{len(chunk_rows)} prompts in {chunk_elapsed:.1f}s "
                f"({len(chunk_rows) / chunk_elapsed:.2f} prompts/s, "
                f"~{chunk_output_chars / max(chunk_elapsed, 1e-9) / 4:.1f} tok/s, "
                f"~{chunk_output_chars / max(len(chunk_rows), 1):.0f} chars/prompt)"
            )
            group_elapsed = perf_counter() - group_started
            remaining = len(remaining_rows) - group_prompts_done
            avg_prompts_per_sec = group_prompts_done / max(group_elapsed, 1e-9)
            eta_seconds = remaining / avg_prompts_per_sec if avg_prompts_per_sec > 0 else 0.0
            info(
                f"Group running average [{output_path.name}]: "
                f"{group_prompts_done}/{len(remaining_rows)} prompts, "
                f"{avg_prompts_per_sec:.2f} prompts/s, "
                f"~{group_output_chars / max(group_elapsed, 1e-9) / 4:.1f} tok/s, "
                f"ETA {_format_eta(eta_seconds)}"
            )
        enriched_rows = _enrich_generation_rows_with_metrics(
            attach_outputs(chunk_rows, outputs),
            distance_matrix=distance_matrix,
            tokenizer_metric_map=tokenizer_metric_map,
        )
        append_jsonl(partial_path, enriched_rows)
    final_count = len(completed_rows) + len(remaining_rows)
    if final_count != len(group_rows):
        raise RuntimeError(f"Checkpoint for {output_path.name} is incomplete after local generation")
    partial_path.replace(output_path)
    return final_count


def _predict_belebele_group_checkpointed(
    prediction_rows: list[dict[str, Any]],
    *,
    prompts: list[str],
    generate_batch: Callable[[list[str]], list[Any]] | None,
    output_path: Path,
    partial_path: Path,
    backend: str,
    model_ref: str,
    max_tokens: int,
    temperature: float,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    mock_output: str | None,
    force: bool,
    local_batch_size: int,
    gpu_batch_size: int = 1,
    max_input_length: int | None = None,
) -> int:
    if len(prediction_rows) != len(prompts):
        raise ValueError("BELEBELE prompt count and prediction row count differ")
    if force and partial_path.exists():
        partial_path.unlink()
    completed_rows = read_jsonl(partial_path) if partial_path.exists() else []
    if len(completed_rows) > len(prediction_rows):
        raise ValueError(f"Checkpoint for {output_path.name} has more rows than expected")
    if len(completed_rows) == len(prediction_rows):
        info(f"Promoting complete checkpoint for {output_path.name} without regeneration")
        partial_path.replace(output_path)
        return len(completed_rows)
    if completed_rows:
        info(f"Resuming {output_path.name} from checkpoint rows={len(completed_rows)}/{len(prediction_rows)}")
    remaining_rows = prediction_rows[len(completed_rows) :]
    remaining_prompts = prompts[len(completed_rows) :]
    for chunk_start in range(0, len(remaining_rows), local_batch_size):
        chunk_rows = remaining_rows[chunk_start : chunk_start + local_batch_size]
        chunk_prompts = remaining_prompts[chunk_start : chunk_start + local_batch_size]
        chunk_offset = len(completed_rows) + chunk_start
        chunk_end = chunk_offset + len(chunk_rows)
        if backend == "mock":
            outputs: list[Any] = [mock_output or "A" for _ in chunk_rows]
        elif mock_output is not None:
            outputs = [mock_output for _ in chunk_rows]
        else:
            with step(
                f"belebele generation [{output_path.name}] "
                f"rows {chunk_offset + 1}-{chunk_end}/{len(prediction_rows)}"
            ):
                if generate_batch is not None:
                    outputs = generate_batch(chunk_prompts)
                else:
                    outputs = generate_outputs(
                        chunk_prompts,
                        model_ref,
                        backend=backend,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        device=device,
                        dtype=dtype,
                        trust_remote_code=trust_remote_code,
                        stream=False,
                        gpu_batch_size=gpu_batch_size,
                        max_input_length=max_input_length,
                    )
        append_jsonl(partial_path, attach_belebele_predictions(chunk_rows, outputs))
    final_count = len(completed_rows) + len(remaining_rows)
    if final_count != len(prediction_rows):
        raise RuntimeError(f"Checkpoint for {output_path.name} is incomplete after BELEBELE generation")
    partial_path.replace(output_path)
    return final_count


def _predict_belebele_maritaca_group_checkpointed(
    prediction_rows: list[dict[str, Any]],
    *,
    prompts: list[str],
    output_path: Path,
    partial_path: Path,
    model_ref: str,
    max_tokens: int,
    temperature: float,
    force: bool,
) -> int:
    if len(prediction_rows) != len(prompts):
        raise ValueError("BELEBELE prompt count and prediction row count differ")
    if force and partial_path.exists():
        partial_path.unlink()
    completed_rows = read_jsonl(partial_path) if partial_path.exists() else []
    if len(completed_rows) > len(prediction_rows):
        raise ValueError(f"Checkpoint for {output_path.name} has more rows than expected")
    if len(completed_rows) == len(prediction_rows):
        info(f"Promoting complete checkpoint for {output_path.name} without regeneration")
        partial_path.replace(output_path)
        return len(completed_rows)
    if completed_rows:
        info(f"Resuming {output_path.name} from checkpoint rows={len(completed_rows)}/{len(prediction_rows)}")
    remaining_rows = prediction_rows[len(completed_rows) :]
    remaining_prompts = prompts[len(completed_rows) :]
    for row, output in zip(
        remaining_rows,
        iter_generate_with_maritaca(
            remaining_prompts,
            model_ref=model_ref,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        strict=True,
    ):
        append_jsonl(partial_path, attach_belebele_predictions([row], [output]))
    final_count = len(read_jsonl(partial_path))
    if final_count != len(prediction_rows):
        raise RuntimeError(f"Checkpoint for {output_path.name} is incomplete after Maritaca BELEBELE generation")
    partial_path.replace(output_path)
    return final_count


def _resolve_local_generation_batch_size(requested: int | None, backend: str) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--local-batch-size must be at least 1")
        return requested
    if backend == "vllm":
        return _DEFAULT_VLLM_LOCAL_GENERATION_CHECKPOINT_BATCH_SIZE
    return _DEFAULT_LOCAL_GENERATION_CHECKPOINT_BATCH_SIZE


_DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_GENERATE = 8
_DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_BELEBELE = 32


def _resolve_gpu_batch_size(requested: int | None, backend: str, *, default: int) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--gpu-batch-size must be at least 1")
        return requested
    # vLLM has its own continuous-batching engine; the on-device batch size is
    # not user-controlled. Mock/maritaca don't use a GPU batch. Only the
    # transformers backend honors --gpu-batch-size.
    if backend == "transformers" or backend == "auto":
        return default
    return 1


def _generate_maritaca_group_checkpointed(
    group_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    partial_path: Path,
    model_ref: str,
    max_tokens: int,
    temperature: float,
    distance_matrix: dict[str, dict[str, float]],
    tokenizer_metric_map: dict[str, dict[str, Any]],
) -> int:
    completed_rows = read_jsonl(partial_path) if partial_path.exists() else []
    if len(completed_rows) > len(group_rows):
        raise ValueError(f"Checkpoint for {output_path.name} has more rows than expected")
    if completed_rows:
        info(f"Resuming {output_path.name} from checkpoint rows={len(completed_rows)}/{len(group_rows)}")
    remaining_rows = group_rows[len(completed_rows) :]
    for row, output in zip(
        remaining_rows,
        iter_generate_with_maritaca(
            [str(group_row["prompt_text"]) for group_row in remaining_rows],
            model_ref=model_ref,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        strict=True,
    ):
        enriched_row = _enrich_generation_rows_with_metrics(
            attach_outputs([row], [output]),
            distance_matrix=distance_matrix,
            tokenizer_metric_map=tokenizer_metric_map,
        )[0]
        append_jsonl(partial_path, [enriched_row])
    if len(read_jsonl(partial_path)) != len(group_rows):
        raise RuntimeError(f"Checkpoint for {output_path.name} is incomplete after Maritaca generation")
    partial_path.replace(output_path)
    return len(group_rows)


def _single_target_language(rows: list[dict[str, object]]) -> str:
    languages = {str(row["target_language"]) for row in rows}
    if len(languages) != 1:
        raise ValueError(f"Expected one target language in output group, found {sorted(languages)}")
    return next(iter(languages))


def _single_attack_language(rows: list[dict[str, Any]]) -> str:
    languages = {str(row["attack_language"]) for row in rows}
    if len(languages) != 1:
        raise ValueError(f"Expected one attack language in output group, found {sorted(languages)}")
    return next(iter(languages))


def _language_output_path(output_dir: str, output_template: str, language: str, input_path: Path, model: str | None) -> Path:
    output_root = ensure_dir(Path(output_dir))
    filename = output_template.format(
        lang=language,
        target_language=language,
        attack_language=language,
        stem=input_path.stem,
        model=model or "",
    )
    return output_root / filename


def cmd_model_status(args: argparse.Namespace) -> None:
    cfg = load_config()
    assets = load_assets()
    asset = assets.get(args.model)
    local_dir = str(asset.local_dir) if asset and asset.local_dir else None
    model_cfg = cfg.models[args.model]
    if model_cfg.get("access_mode") == "api":
        print(
            json.dumps(
                {
                    "model": args.model,
                    "access_mode": "api",
                    "provider": model_cfg.get("provider"),
                    "provider_model_id": model_cfg.get("provider_model_id"),
                    "download_required": False,
                },
                indent=2,
            )
        )
        return
    print(json.dumps(estimate_model_asset_state(str(model_cfg["hf_id"]), local_dir=local_dir), indent=2))


def cmd_score_strongreject(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1; got {args.workers}")
    generation_paths = [Path(path) for path in args.generations]
    if not args.output and not args.output_dir:
        raise ValueError("score-strongreject requires --output or --output-dir.")
    if len(generation_paths) > 1 and args.output:
        raise ValueError("Use --output-dir for multi-file scoring; --output is only for a single input file.")

    pending: list[tuple[Path, Path, list[dict[str, Any]], int]] = []
    for generation_path in generation_paths:
        group_rows = read_jsonl(generation_path)
        if args.output_dir:
            output_dir = ensure_dir(Path(args.output_dir))
            filename = str(args.output_template).format(stem=generation_path.stem)
            output_path = output_dir / filename
        else:
            output_path = Path(args.output)
        completed_count = 0
        if args.resume and output_path.exists() and not args.force:
            completed_count = len(read_jsonl(output_path))
            if completed_count > len(group_rows):
                raise RuntimeError(
                    f"{output_path} has {completed_count} rows but {generation_path} has only {len(group_rows)}."
                )
            group_rows = group_rows[completed_count:]
            if not group_rows:
                info(f"score {generation_path.name}: already complete via {output_path}")
                continue
        else:
            if _skip_existing(output_path, args.force, f"score {generation_path.name}"):
                continue
        pending.append((generation_path, output_path, group_rows, completed_count))
    if not pending:
        info("score-strongreject: all outputs already exist; nothing to do.")
        return

    translator = None
    if args.backtranslate_engine == "nllb" and any(
        str(row.get("attack_language")) != "eng" for _, _, group_rows, _ in pending for row in group_rows
    ):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    written: dict[str, str] = {}
    for generation_path, output_path, group_rows, completed_count in pending:
        with step(f"score {generation_path.name} ({len(group_rows)} rows)"):
            if args.resume:
                if args.force and output_path.exists():
                    output_path.unlink()
                    completed_count = 0
                scored_count = completed_count
                total_count = completed_count + len(group_rows)
                if args.workers == 1:
                    for row in group_rows:
                        scored = score_generation_rows(
                            [row],
                            evaluator=args.evaluator,
                            backtranslate_engine=args.backtranslate_engine,
                            checkpoint=args.checkpoint,
                            device=args.device,
                            dtype=args.dtype,
                            translator=translator,
                            judge_model=args.judge_model,
                            api_base=args.api_base,
                            max_retries=args.max_retries,
                            retry_initial_delay=args.retry_initial_delay,
                        )
                        append_jsonl(output_path, scored)
                        scored_count += len(scored)
                        if scored_count % 25 == 0:
                            info(f"score {generation_path.name}: checkpointed {scored_count}/{total_count} rows")
                else:
                    if args.backtranslate_engine == "nllb":
                        raise RuntimeError(
                            "Parallel score-strongreject is only supported when response backtranslations already "
                            "exist or --backtranslate-engine is placeholder. Run backtranslate-responses first."
                        )
                    scored_count = _score_rows_parallel_resume(
                        group_rows,
                        output_path=output_path,
                        completed_count=completed_count,
                        total_count=total_count,
                        generation_name=generation_path.name,
                        workers=args.workers,
                        evaluator=args.evaluator,
                        checkpoint=args.checkpoint,
                        device=args.device,
                        dtype=args.dtype,
                        judge_model=args.judge_model,
                        api_base=args.api_base,
                        max_retries=args.max_retries,
                        retry_initial_delay=args.retry_initial_delay,
                    )
                written[str(output_path)] = str(scored_count)
                print(f"Wrote/resumed {scored_count} StrongREJECT-scored rows to {output_path}")
            else:
                if args.workers > 1:
                    if args.backtranslate_engine == "nllb":
                        raise RuntimeError(
                            "Parallel score-strongreject is only supported when response backtranslations already "
                            "exist or --backtranslate-engine is placeholder. Run backtranslate-responses first."
                        )
                    scored = _score_rows_parallel_collect(
                        group_rows,
                        workers=args.workers,
                        evaluator=args.evaluator,
                        checkpoint=args.checkpoint,
                        device=args.device,
                        dtype=args.dtype,
                        judge_model=args.judge_model,
                        api_base=args.api_base,
                        max_retries=args.max_retries,
                        retry_initial_delay=args.retry_initial_delay,
                    )
                else:
                    scored = score_generation_rows(
                        group_rows,
                        evaluator=args.evaluator,
                        backtranslate_engine=args.backtranslate_engine,
                        checkpoint=args.checkpoint,
                        device=args.device,
                        dtype=args.dtype,
                        translator=translator,
                        judge_model=args.judge_model,
                        api_base=args.api_base,
                        max_retries=args.max_retries,
                        retry_initial_delay=args.retry_initial_delay,
                    )
                write_jsonl(output_path, scored)
                written[str(output_path)] = str(len(scored))
                print(f"Wrote {len(scored)} StrongREJECT-scored rows to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_repair_strongreject_scores(args: argparse.Namespace) -> None:
    scored_paths = [Path(path) for path in args.scored]
    if not args.output and not args.output_dir and not args.in_place:
        raise ValueError("repair-strongreject-scores requires --output, --output-dir, or --in-place.")
    if len(scored_paths) > 1 and args.output:
        raise ValueError("Use --output-dir or --in-place for multiple scored files.")
    if args.in_place and (args.output or args.output_dir):
        raise ValueError("--in-place cannot be combined with --output or --output-dir.")

    summary: dict[str, dict[str, int | str]] = {}
    for scored_path in scored_paths:
        rows = read_jsonl(scored_path)
        repaired_rows, repaired_count = repair_strongreject_parse_failures(rows)
        if args.in_place:
            output_path = scored_path
        elif args.output_dir:
            output_dir = ensure_dir(Path(args.output_dir))
            filename = str(args.output_template).format(stem=scored_path.stem)
            output_path = output_dir / filename
        else:
            output_path = Path(args.output)
        if output_path.exists() and not args.force and output_path != scored_path:
            raise RuntimeError(f"{output_path} already exists. Use --force to overwrite repaired scores.")
        if output_path == scored_path and not args.force:
            raise RuntimeError(f"{output_path} would be overwritten in place. Use --force to confirm.")
        write_jsonl(output_path, repaired_rows)
        summary[str(scored_path)] = {
            "rows": len(rows),
            "repaired": repaired_count,
            "output": str(output_path),
        }
        print(f"Repaired {repaired_count}/{len(rows)} StrongREJECT rows into {output_path}")
    if len(summary) > 1:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _score_rows_parallel_resume(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    completed_count: int,
    total_count: int,
    generation_name: str,
    workers: int,
    evaluator: str,
    checkpoint: str,
    device: str,
    dtype: str,
    judge_model: str | None,
    api_base: str,
    max_retries: int,
    retry_initial_delay: float,
) -> int:
    next_to_write = 0
    buffered: dict[int, dict[str, Any]] = {}
    scored_count = completed_count

    def score_one(index_and_row: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, row = index_and_row
        scored = score_generation_rows(
            [row],
            evaluator=evaluator,
            backtranslate_engine="placeholder",
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
            translator=None,
            judge_model=judge_model,
            api_base=api_base,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
        )
        return index, scored[0]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_one, item) for item in enumerate(rows)]
        for future in as_completed(futures):
            index, scored_row = future.result()
            buffered[index] = scored_row
            ready_rows: list[dict[str, Any]] = []
            while next_to_write in buffered:
                ready_rows.append(buffered.pop(next_to_write))
                next_to_write += 1
            if ready_rows:
                append_jsonl(output_path, ready_rows)
                scored_count += len(ready_rows)
                if scored_count % 25 == 0 or scored_count == total_count:
                    info(f"score {generation_name}: checkpointed {scored_count}/{total_count} rows")
    return scored_count


def _score_rows_parallel_collect(
    rows: list[dict[str, Any]],
    *,
    workers: int,
    evaluator: str,
    checkpoint: str,
    device: str,
    dtype: str,
    judge_model: str | None,
    api_base: str,
    max_retries: int,
    retry_initial_delay: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any] | None] = [None] * len(rows)

    def score_one(index_and_row: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, row = index_and_row
        scored = score_generation_rows(
            [row],
            evaluator=evaluator,
            backtranslate_engine="placeholder",
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
            translator=None,
            judge_model=judge_model,
            api_base=api_base,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
        )
        return index, scored[0]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_one, item) for item in enumerate(rows)]
        for future in as_completed(futures):
            index, scored_row = future.result()
            output[index] = scored_row
    return [row for row in output if row is not None]


def cmd_compute_spec(args: argparse.Namespace) -> None:
    cfg = load_config()
    predictions = read_jsonl(Path(args.predictions))
    rows = compute_if_cons(predictions, cfg.aligned_language)
    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} BELEBELE IF/CONS/SPEC rows to {args.output}")


def cmd_score_benign(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.generations))
    scored = score_benign_rows(rows)
    write_jsonl(Path(args.output), scored)
    print(f"Wrote {len(scored)} benign-control scored rows to {args.output}")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    cfg = load_config()
    scored_rows = read_jsonl(Path(args.scored))
    translations = load_translation_logs(Path(args.translations)) if args.translations else None
    spec_rows = read_jsonl(Path(args.spec_scores)) if args.spec_scores else None
    rows = build_frozen_rows(scored_rows, cfg, translations=translations, spec_rows=spec_rows)
    output = Path(args.output)
    if output.name == "dataset_frozen.parquet" and output.exists() and not args.allow_frozen_overwrite:
        raise RuntimeError("Refusing to overwrite dataset_frozen.parquet without --allow-frozen-overwrite")
    if output.suffix == ".parquet":
        import pandas as pd

        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output, index=False)
    else:
        write_jsonl(output, rows)
    print(f"Wrote {len(rows)} frozen-schema rows to {output}")


def cmd_export_report_tables(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.rows))
    output_dir = ensure_dir(Path(args.output_dir))
    write_csv_dicts(output_dir / "results_coverage.csv", coverage_by_model(rows), ["model", "total_attempts", "api_failure_rate", "provider_block_rate", "scoring_failure_rate", "refusal_rate"])
    write_csv_dicts(output_dir / "asr_by_model_language.csv", asr_by_model_language(rows), ["model", "attack_language", "n", "unsafe", "asr", "ci_low", "ci_high", "mean_strongreject_score", "refusal_rate"])
    write_csv_dicts(output_dir / "crosslingual_asr_by_model_language.csv", asr_by_model_language(rows, exclude_aligned=True), ["model", "attack_language", "n", "unsafe", "asr", "ci_low", "ci_high", "mean_strongreject_score", "refusal_rate"])
    write_csv_dicts(output_dir / "closest_farthest_languages.csv", closest_farthest_by_model(rows), ["model", "closest_language", "closest_distance", "closest_asr", "farthest_language", "farthest_distance", "farthest_asr", "gap"])
    distance_corr, spec_corr = spearman_tables(rows)
    write_csv_dicts(output_dir / "distance_asr_correlation.csv", distance_corr, ["model", "predictor", "n", "spearman_rho", "p_value", "interpretation"])
    write_csv_dicts(output_dir / "spec_asr_correlation.csv", spec_corr, ["model", "predictor", "n", "spearman_rho", "p_value", "interpretation"])
    write_csv_dicts(output_dir / "tokenizer_diagnostics.csv", tokenizer_diagnostics(rows), ["model", "attack_language", "n", "mean_token_inflation", "mean_tokens_per_char", "truncation_risk_rate"])
    write_csv_dicts(output_dir / "belebele_scores.csv", belebele_scores(rows), ["model", "attack_language", "if_score", "cons_score", "spec_score"])
    write_csv_dicts(output_dir / "reference_distance_curve.csv", reference_distance_curve(rows), ["reference_model", "attack_language", "distance_from_english", "n", "unsafe", "asr", "mean_strongreject_score", "refusal_rate"])
    write_csv_dicts(output_dir / "counterfactual_safety_by_aligned_language.csv", counterfactual_safety_by_aligned_language(rows), ["aligned_language", "model", "model_alignment_pole", "model_asr", "reference_model", "reference_asr_same_language", "asr_gap_model_minus_reference", "model_n", "reference_n"])
    write_csv_dicts(output_dir / "glmm_collinearity.csv", collinearity_diagnostics(rows), ["diagnostic", "value"])
    write_csv_dicts(output_dir / "prereg_distance_slope_retention.csv", prereg_distance_slope_retention(rows), ["model_pair_language", "weak_model", "weak_distance_slope", "weak_slope_sign", "weak_n", "strong_model", "strong_distance_slope", "strong_slope_sign", "strong_n", "slope_sign_retained"])
    write_csv_dicts(output_dir / "prereg_falsification_summary.csv", prereg_falsification_summary(rows), ["pairs_evaluated", "pairs_with_slope_sign_retained", "panel_complete", "interpretation"])
    print(f"Wrote report table CSVs to {output_dir}")


def cmd_fit_glmm(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.rows))
    effects = fit_main_glmm(rows)
    write_csv_dicts(Path(args.output), effects, ["predictor", "estimate", "std_error", "odds_ratio", "p_value", "fit_method"])
    print(f"Wrote GLMM main effects to {args.output}")


def cmd_export_audit_queue(args: argparse.Namespace) -> None:
    logs = load_translation_logs(Path(args.translations))
    queue = build_audit_queue(logs, include_passed=args.include_passed, audit_plan=args.audit_plan)
    write_csv_dicts(Path(args.output), queue, AUDIT_COLUMNS)
    print(f"Wrote {len(queue)} audit rows to {args.output}")


def cmd_import_audit(args: argparse.Namespace) -> None:
    logs = load_translation_logs(Path(args.translations))
    decisions = read_csv_dicts(Path(args.audit_csv))
    updated = apply_audit_decisions(logs, decisions)
    write_jsonl(Path(args.output), updated)
    print(f"Wrote {len(updated)} audit-updated translation logs to {args.output}")


def cmd_judge_audit_queue(args: argparse.Namespace) -> None:
    rows = read_csv_dicts(Path(args.audit_queue))
    judged = judge_audit_rows(
        rows,
        provider=args.provider,
        model=args.model,
        sleep_seconds=args.sleep_seconds,
        limit=args.limit,
        force=args.force,
    )
    write_csv_dicts(Path(args.output), judged, AUDIT_COLUMNS)
    completed = sum(1 for row in judged if str(row.get("xsts_score", "")).strip() and str(row.get("audit_decision", "")).strip())
    print(f"Wrote {completed}/{len(judged)} judged audit rows to {args.output}")


def cmd_calibrate_blaser(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for path_str in args.audit_csv:
        path = Path(path_str)
        chunk = read_csv_dicts(path) if path.suffix == ".csv" else read_jsonl(path)
        rows.extend(chunk)
    result = calibrate_blaser_thresholds(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))


def cmd_reference_spbleu(args: argparse.Namespace) -> None:
    logs = load_translation_logs(Path(args.translations))
    references = read_jsonl(Path(args.references))
    updated = attach_reference_scores(logs, references)
    write_jsonl(Path(args.output), updated)
    print(f"Wrote {len(updated)} translation logs with reference scores to {args.output}")


def cmd_run_pilot(args: argparse.Namespace) -> None:
    prompts_path = Path(args.prompts) if args.prompts else None
    if prompts_path is not None and not prompts_path.exists():
        prompts_path = None
    outputs = run_mock_pilot(
        Path(args.output_dir),
        prompts_path=prompts_path,
        prompt_limit=args.prompt_limit,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    print("Pilot produced mock StrongREJECT-scored rows; run real generation/scoring before research use.")


def cmd_export_results(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.rows))
    validate_analysis_rows(rows)
    output = Path(args.output)
    if output.name == "dataset_frozen.parquet" and not args.allow_frozen_overwrite and output.exists():
        raise RuntimeError("Refusing to overwrite dataset_frozen.parquet without --allow-frozen-overwrite")
    if output.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Parquet export requires pandas and a parquet engine.") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output, index=False)
    else:
        write_jsonl(output, rows)
    print(f"Wrote {len(rows)} rows to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thesis-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    runtime_profile_choices = sorted(load_runtime_profiles())

    validate = sub.add_parser("validate-config")
    validate.set_defaults(func=cmd_validate_config)

    runtime = sub.add_parser("runtime-profiles")
    runtime.set_defaults(func=cmd_runtime_profiles)

    list_assets = sub.add_parser("list-assets")
    list_assets.set_defaults(func=cmd_list_assets)

    asset_status_parser = sub.add_parser("asset-status")
    asset_status_parser.add_argument("--name", action="append")
    asset_status_parser.add_argument("--group", action="append")
    asset_status_parser.set_defaults(func=cmd_asset_status)

    download_assets = sub.add_parser("download-assets")
    download_assets.add_argument("--name", action="append")
    download_assets.add_argument("--group", action="append")
    download_assets.add_argument("--dry-run", action="store_true")
    download_assets.add_argument("--force", action="store_true")
    download_assets.set_defaults(func=cmd_download_assets)

    verify_assets_parser = sub.add_parser("verify-assets")
    verify_assets_parser.add_argument("--name", action="append")
    verify_assets_parser.add_argument("--group", action="append")
    verify_assets_parser.add_argument("--deep", action="store_true")
    verify_assets_parser.set_defaults(func=cmd_verify_assets)

    prepare = sub.add_parser("prepare-uriel")
    prepare.add_argument("--source-csv", default=str(THESIS_ROOT / "pictures" / "urielplus_distance_featural.csv"))
    prepare.add_argument("--from-csv", action="store_true", help="Prepare from a static matrix CSV instead of recomputing with URIEL+.")
    prepare.add_argument("--distance-type", default="featural")
    prepare.add_argument("--uriel-backend", choices=["uv", "in-process"], default="uv")
    prepare.add_argument("--uriel-python", help="Explicit Python executable from an environment with urielplus installed.")
    prepare.add_argument("--output-csv", default=str(DATA_DIR / "uriel_plus" / "distance_matrix.csv"))
    prepare.add_argument("--output-tex", default=str(THESIS_ROOT / "aftertext" / "urielplus_distance_table_generated.tex"))
    prepare.add_argument("--run-config-output", default=str(DATA_DIR / "uriel_plus" / "urielplus_run_config.json"))
    prepare.set_defaults(func=cmd_prepare_uriel)

    strongreject = sub.add_parser("import-strongreject")
    strongreject.add_argument("--source", choices=["github", "hf", "local"], default="github")
    strongreject.add_argument("--input")
    strongreject.add_argument("--output", default=str(DATA_DIR / "raw" / "strongreject" / "prompts.jsonl"))
    strongreject.add_argument("--pilot-output", default=str(DATA_DIR / "raw" / "strongreject" / "pilot_prompts.jsonl"))
    strongreject.add_argument("--pilot-size", type=int, default=2)
    strongreject.add_argument("--limit", type=int)
    strongreject.add_argument("--small", action="store_true")
    strongreject.set_defaults(func=cmd_import_strongreject)

    translate = sub.add_parser("translate")
    translate.add_argument("--prompts", required=True)
    translate.add_argument("--target-language", action="append", required=True)
    translate.add_argument("--output")
    translate.add_argument("--output-dir")
    translate.add_argument("--output-template", default="translations_{lang}.jsonl")
    translate.add_argument("--engine", choices=["placeholder", "nllb"], default="placeholder")
    translate.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    translate.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    translate.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    translate.add_argument("--stream", action="store_true", help="Print each prompt and its translation as it is produced.")
    translate.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    translate.set_defaults(func=cmd_translate)

    qc = sub.add_parser("translation-qc")
    qc.add_argument("--translations", action="append", required=True)
    qc.add_argument("--output")
    qc.add_argument("--output-dir")
    qc.add_argument("--output-template", default="{stem}.qc.jsonl")
    qc.add_argument("--scorer", choices=["heuristic", "blaser"], default="heuristic")
    qc.add_argument("--pass-threshold", type=float)
    qc.add_argument("--fail-threshold", type=float)
    qc.add_argument("--device")
    qc.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    qc.set_defaults(func=cmd_translation_qc)

    roundtrip = sub.add_parser("prompt-roundtrip")
    roundtrip.add_argument("--translations", action="append", required=True)
    roundtrip.add_argument("--output")
    roundtrip.add_argument("--output-dir")
    roundtrip.add_argument("--output-template", default="{stem}.roundtrip.jsonl")
    roundtrip.add_argument("--engine", choices=["placeholder", "nllb"], default="placeholder")
    roundtrip.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    roundtrip.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    roundtrip.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    roundtrip.add_argument("--pass-threshold", type=float, default=0.60)
    roundtrip.add_argument("--fail-threshold", type=float, default=0.35)
    roundtrip.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    roundtrip.set_defaults(func=cmd_prompt_roundtrip)

    sonar = sub.add_parser("prewarm-sonar")
    sonar.add_argument("--cache-dir")
    sonar.set_defaults(func=cmd_prewarm_sonar)

    gen = sub.add_parser("generate-targets")
    gen.add_argument("--translations", action="append", required=True)
    gen.add_argument("--model", default="sagui_7b")
    gen.add_argument("--benchmark", default="strongreject")
    gen.add_argument("--output")
    gen.add_argument("--output-dir")
    gen.add_argument("--output-template", default="target_generations_{model}_{lang}.jsonl")
    gen.add_argument("--profile", choices=runtime_profile_choices)
    gen.add_argument("--backend", choices=["auto", "mock", "vllm", "transformers", "maritaca"])
    gen.add_argument("--device", choices=["auto", "cpu", "mps", "cuda", "xla"])
    gen.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
    gen.add_argument("--mock-output")
    gen.add_argument("--max-tokens", type=int, default=512)
    gen.add_argument("--max-records", type=int, help="Only generate the first N retained prompts from each translation file.")
    gen.add_argument("--local-batch-size", type=int, help="Local open-weight prompts per checkpointed generation chunk.")
    gen.add_argument(
        "--gpu-batch-size",
        type=int,
        help=(
            "Prompts per on-device forward pass for the transformers backend. "
            "Defaults to 8 for generate-targets. Ignored for vLLM/Maritaca/mock."
        ),
    )
    gen.add_argument(
        "--max-input-length",
        type=int,
        help=(
            "Fixed input padding length for the batched transformers runner. "
            "Required on XLA/TPU for shape-stable HLOs; defaults to "
            "min(model.max_position_embeddings - max_tokens, 4096) on XLA, dynamic elsewhere."
        ),
    )
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--trust-remote-code", action="store_true")
    gen.add_argument("--model-path")
    gen.add_argument("--stream", action="store_true")
    gen.add_argument("--skip-tokenizer-metrics", action="store_true")
    gen.add_argument("--distance-csv", default=str(DATA_DIR / "uriel_plus" / "distance_matrix.csv"))
    gen.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    gen.set_defaults(func=cmd_generate_targets)

    belebele = sub.add_parser("predict-belebele")
    belebele.add_argument("--model", default="sagui_7b")
    belebele.add_argument("--language", action="append", help="One attack language code (e.g. por, ara). Defaults to all thesis languages.")
    belebele.add_argument("--output")
    belebele.add_argument("--output-dir")
    belebele.add_argument("--output-template", default="{model}_{lang}.belebele.jsonl")
    belebele.add_argument("--dataset-dir", default=str(load_assets()["belebele"].local_dir))
    belebele.add_argument("--profile", choices=runtime_profile_choices)
    belebele.add_argument("--backend", choices=["auto", "mock", "vllm", "transformers", "maritaca"])
    belebele.add_argument("--device", choices=["auto", "cpu", "mps", "cuda", "xla"])
    belebele.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
    belebele.add_argument("--mock-output")
    belebele.add_argument("--max-tokens", type=int, default=8)
    belebele.add_argument("--limit", type=int, help="Only score the first N BELEBELE items from each language.")
    belebele.add_argument("--local-batch-size", type=int, help="Local prompts per checkpointed BELEBELE chunk.")
    belebele.add_argument(
        "--gpu-batch-size",
        type=int,
        help=(
            "Prompts per on-device forward pass for the transformers backend. "
            "Defaults to 32 for predict-belebele. Ignored for vLLM/Maritaca/mock."
        ),
    )
    belebele.add_argument(
        "--max-input-length",
        type=int,
        help=(
            "Fixed input padding length for the batched transformers runner. "
            "Required on XLA/TPU for shape-stable HLOs."
        ),
    )
    belebele.add_argument("--temperature", type=float, default=0.0)
    belebele.add_argument("--trust-remote-code", action="store_true")
    belebele.add_argument("--model-path")
    belebele.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    belebele.set_defaults(func=cmd_predict_belebele)

    belebele_repair = sub.add_parser("repair-belebele-predictions")
    belebele_repair.add_argument("--predictions", required=True)
    belebele_repair.add_argument("--output", required=True)
    belebele_repair.add_argument("--dataset-dir", default=str(load_assets()["belebele"].local_dir))
    belebele_repair.set_defaults(func=cmd_repair_belebele_predictions)

    token_metrics = sub.add_parser("attach-tokenizer-metrics")
    token_metrics.add_argument("--generations", required=True)
    token_metrics.add_argument("--output", required=True)
    token_metrics.add_argument("--model")
    token_metrics.add_argument("--model-path")
    token_metrics.add_argument("--trust-remote-code", action="store_true")
    token_metrics.set_defaults(func=cmd_attach_tokenizer_metrics)

    response_bt = sub.add_parser("backtranslate-responses")
    response_bt.add_argument("--generations", action="append", required=True)
    response_bt.add_argument("--output")
    response_bt.add_argument("--output-dir")
    response_bt.add_argument("--output-template", default="{stem}.backtranslated.jsonl")
    response_bt.add_argument("--engine", choices=["placeholder", "nllb"], default="placeholder")
    response_bt.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    response_bt.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    response_bt.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    response_bt.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    response_bt.set_defaults(func=cmd_backtranslate_responses)

    sr_prepare = sub.add_parser("prepare-strongreject-batch")
    sr_prepare.add_argument("--generations", action="append", required=True)
    sr_prepare.add_argument("--requests-output", required=True)
    sr_prepare.add_argument("--manifest-output", required=True)
    sr_prepare.add_argument("--scored-output-dir", required=True)
    sr_prepare.add_argument("--scored-output-template", default="{stem}.strongreject.jsonl")
    sr_prepare.add_argument("--judge-model", required=True)
    sr_prepare.add_argument("--backtranslate-engine", choices=["reuse", "placeholder", "nllb"], default="reuse")
    sr_prepare.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    sr_prepare.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    sr_prepare.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    sr_prepare.add_argument("--custom-id-prefix", default="strongreject")
    sr_prepare.add_argument("--force", action="store_true", help="Overwrite batch prep artifacts and scored outputs.")
    sr_prepare.set_defaults(func=cmd_prepare_strongreject_batch)

    submit_batch = sub.add_parser("submit-openai-batch")
    submit_batch.add_argument("--input", required=True)
    submit_batch.add_argument("--output", required=True)
    submit_batch.add_argument("--completion-window", default="24h")
    submit_batch.add_argument("--metadata", action="append")
    submit_batch.add_argument("--api-base", default="https://api.openai.com/v1")
    submit_batch.set_defaults(func=cmd_submit_openai_batch)

    shard_batch = sub.add_parser("shard-openai-batch-requests")
    shard_batch.add_argument("--input", required=True)
    shard_batch.add_argument("--output-dir", required=True)
    shard_batch.add_argument("--output-prefix")
    shard_batch.add_argument("--index-output")
    shard_batch.add_argument("--max-estimated-input-tokens", type=int, default=750_000)
    shard_batch.add_argument("--max-requests", type=int)
    shard_batch.add_argument("--force", action="store_true", help="Overwrite existing shard files and shard index.")
    shard_batch.set_defaults(func=cmd_shard_openai_batch_requests)

    fetch_batch = sub.add_parser("fetch-openai-batch")
    fetch_batch.add_argument("--batch-id", required=True)
    fetch_batch.add_argument("--output", required=True)
    fetch_batch.add_argument("--download-output")
    fetch_batch.add_argument("--download-errors")
    fetch_batch.add_argument("--api-base", default="https://api.openai.com/v1")
    fetch_batch.set_defaults(func=cmd_fetch_openai_batch)

    ingest_batch = sub.add_parser("ingest-strongreject-batch")
    ingest_batch.add_argument("--manifest", required=True)
    ingest_batch.add_argument("--batch-output", required=True)
    ingest_batch.add_argument("--batch-errors")
    ingest_batch.add_argument("--force", action="store_true", help="Overwrite scored outputs that already exist.")
    ingest_batch.set_defaults(func=cmd_ingest_strongreject_batch)

    model_status = sub.add_parser("model-status")
    model_status.add_argument("--model", default="sagui_7b")
    model_status.set_defaults(func=cmd_model_status)

    sr_score = sub.add_parser("score-strongreject")
    sr_score.add_argument("--generations", action="append", required=True)
    sr_score.add_argument("--output")
    sr_score.add_argument("--output-dir")
    sr_score.add_argument("--output-template", default="{stem}.strongreject.jsonl")
    sr_score.add_argument("--evaluator", default="strongreject_rubric")
    sr_score.add_argument("--backtranslate-engine", choices=["placeholder", "nllb"], default="placeholder")
    sr_score.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    sr_score.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    sr_score.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    sr_score.add_argument(
        "--judge-model",
        help=(
            "OpenAI model name passed to the strongreject_rubric judge "
            "(e.g. gpt-5.4, gpt-4o). Defaults to the upstream library's "
            "fallback (gpt-4o-mini). Use the same model that produced the "
            "audit decisions for cross-stage consistency."
        ),
    )
    sr_score.add_argument("--api-base", default="https://api.openai.com/v1")
    sr_score.add_argument("--max-retries", type=int, default=6)
    sr_score.add_argument("--retry-initial-delay", type=float, default=2.0)
    sr_score.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel normal-API scoring workers. Use with --resume for checkpointed long runs.",
    )
    sr_score.add_argument(
        "--resume",
        action="store_true",
        help="Append scored rows and resume from an existing partial output file.",
    )
    sr_score.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    sr_score.set_defaults(func=cmd_score_strongreject)

    sr_repair = sub.add_parser("repair-strongreject-scores")
    sr_repair.add_argument("--scored", action="append", required=True)
    sr_repair.add_argument("--output")
    sr_repair.add_argument("--output-dir")
    sr_repair.add_argument("--output-template", default="{stem}.repaired.jsonl")
    sr_repair.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite scored files in place. Requires --force.",
    )
    sr_repair.add_argument("--force", action="store_true", help="Overwrite repaired score outputs.")
    sr_repair.set_defaults(func=cmd_repair_strongreject_scores)

    spec = sub.add_parser("compute-spec")
    spec.add_argument("--predictions", required=True)
    spec.add_argument("--output", required=True)
    spec.set_defaults(func=cmd_compute_spec)

    benign = sub.add_parser("score-benign-controls")
    benign.add_argument("--generations", required=True)
    benign.add_argument("--output", required=True)
    benign.set_defaults(func=cmd_score_benign)

    dataset = sub.add_parser("build-dataset")
    dataset.add_argument("--scored", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--translations")
    dataset.add_argument("--spec-scores")
    dataset.add_argument("--allow-frozen-overwrite", action="store_true")
    dataset.set_defaults(func=cmd_build_dataset)

    reports = sub.add_parser("export-report-tables")
    reports.add_argument("--rows", required=True)
    reports.add_argument("--output-dir", required=True)
    reports.set_defaults(func=cmd_export_report_tables)

    glmm = sub.add_parser("fit-glmm")
    glmm.add_argument("--rows", required=True)
    glmm.add_argument("--output", required=True)
    glmm.set_defaults(func=cmd_fit_glmm)

    audit_export = sub.add_parser("export-audit-queue")
    audit_export.add_argument("--translations", required=True)
    audit_export.add_argument("--output", required=True)
    audit_export.add_argument("--audit-plan", choices=["plan_a", "plan_b"], default="plan_b")
    audit_export.add_argument("--include-passed", action="store_true")
    audit_export.set_defaults(func=cmd_export_audit_queue)

    audit_import = sub.add_parser("import-audit")
    audit_import.add_argument("--translations", required=True)
    audit_import.add_argument("--audit-csv", required=True)
    audit_import.add_argument("--output", required=True)
    audit_import.set_defaults(func=cmd_import_audit)

    audit_judge = sub.add_parser("judge-audit-queue")
    audit_judge.add_argument("--audit-queue", required=True)
    audit_judge.add_argument("--output", required=True)
    audit_judge.add_argument("--provider", choices=["openai", "mock"], default="openai")
    audit_judge.add_argument("--model", default="gpt-4o")
    audit_judge.add_argument("--sleep-seconds", type=float, default=0.0)
    audit_judge.add_argument("--limit", type=int)
    audit_judge.add_argument("--force", action="store_true", help="Re-judge rows that already have xsts_score and audit_decision.")
    audit_judge.set_defaults(func=cmd_judge_audit_queue)

    calibrate = sub.add_parser("calibrate-blaser")
    calibrate.add_argument("--audit-csv", action="append", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(func=cmd_calibrate_blaser)

    spbleu = sub.add_parser("reference-spbleu")
    spbleu.add_argument("--translations", required=True)
    spbleu.add_argument("--references", required=True)
    spbleu.add_argument("--output", required=True)
    spbleu.set_defaults(func=cmd_reference_spbleu)

    pilot = sub.add_parser("run-pilot")
    pilot.add_argument("--output-dir", default=str(OUTPUT_DIR / "pilot"))
    pilot.add_argument("--prompts", default=str(DATA_DIR / "raw" / "strongreject" / "pilot_prompts.jsonl"))
    pilot.add_argument("--prompt-limit", type=int, default=2)
    pilot.set_defaults(func=cmd_run_pilot)

    export = sub.add_parser("export-results")
    export.add_argument("--rows", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--allow-frozen-overwrite", action="store_true")
    export.set_defaults(func=cmd_export_results)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
