from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .discovery import discover_models, select_models, variants_for_model
from .hardware import bytes_label, capture_hardware
from .paths import (
    first_executable,
    llama_server_candidates,
    llama_supported_speculative_methods,
    mlx_python_candidates,
    model_registry_path,
    tater_home,
)
from .reporting import write_reports
from .runner import BenchmarkRunner, save_batch
from .scenarios import load_suite
from .version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tater-bench", description="Benchmark models installed by Tater")
    parser.add_argument("--tater-home", default=None, help="Tater home (default: ~/.taterassistant)")
    parser.add_argument("--version", action="version", version=f"Tater Bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="List discovered benchmark target models")
    models.add_argument("--json", action="store_true")

    hardware = sub.add_parser("hardware", help="Show the publish-safe hardware profile")
    hardware.add_argument("--json", action="store_true")
    hardware.add_argument("--include-hostname", action="store_true")

    sub.add_parser("doctor", help="Check Tater registry and engine availability")

    validate = sub.add_parser("validate-suite", help="Validate a suite without loading a model")
    validate.add_argument("suite", nargs="?", default="core")

    run = sub.add_parser("run", help="Run one or more installed models")
    run.add_argument("--model", action="append", default=[], help="Model id/name/path substring; repeatable")
    run.add_argument("--all", action="store_true", help="Benchmark every discovered target model")
    run.add_argument("--provider", choices=["llama_cpp", "mlx_lm"], default=None)
    run.add_argument("--suite", default="core")
    run.add_argument("--variant", action="append", choices=["baseline", "mtp", "dflash", "dspark"], default=[])
    run.add_argument("--baseline-only", action="store_true")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--context", type=int, default=8192)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--limit", type=int, default=0, help="Run only the first N scenarios")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", default="results")
    run.add_argument("--yes", action="store_true", help="Start without interactive confirmation")

    report = sub.add_parser("report", help="Regenerate GitHub and optional HTML reports")
    report.add_argument("--results", default="results")
    report.add_argument("--markdown", default="RESULTS.md")
    report.add_argument("--docs", default="docs")
    return parser


def _print_models(models: list[Any]) -> None:
    if not models:
        print("No benchmark target models were found.")
        return
    for model in models:
        modes = ", ".join(variant.name for variant in variants_for_model(model))
        print(f"{model.id}  {model.provider:10}  {model.label}")
        print(f"  modes: {modes} | quant: {model.quantization or 'unknown'} | path: {model.model_path}")


def _doctor(home_value: str | None) -> int:
    home = tater_home(home_value)
    registry = model_registry_path(home)
    llama = first_executable(llama_server_candidates(home))
    mlx_python = first_executable(mlx_python_candidates(home))
    spec_methods = llama_supported_speculative_methods(home)
    models = discover_models(home)
    checks = [
        ("Tater home", home.exists(), str(home)),
        ("Model registry", registry.is_file(), str(registry)),
        ("llama.cpp engine", llama is not None, str(llama or "not found")),
        ("MLX Python", mlx_python is not None, str(mlx_python or "not found")),
        ("llama.cpp decoders", bool(spec_methods), ", ".join(sorted(spec_methods)) or "none detected"),
        ("Benchmark targets", bool(models), f"{len(models)} discovered"),
    ]
    for label, ok, detail in checks:
        print(f"{'OK' if ok else 'MISSING':7} {label}: {detail}")
    return 0 if all(ok for _label, ok, _detail in checks[:2]) and bool(models) else 1


def _jobs(args: argparse.Namespace, models: list[Any]) -> list[tuple[Any, Any]]:
    selected = models if args.all else select_models(models, list(args.model or []))
    if args.provider:
        selected = [model for model in selected if model.provider == args.provider]
    requested_variants = set(args.variant or [])
    if args.baseline_only:
        requested_variants = {"baseline"}
    jobs: list[tuple[Any, Any]] = []
    for model in selected:
        for variant in variants_for_model(model, include_speculative=not args.baseline_only):
            if requested_variants and variant.name not in requested_variants:
                continue
            jobs.append((model, variant))
    return jobs


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    home = tater_home(args.tater_home)
    if args.command == "models":
        models = discover_models(home)
        if args.json:
            print(json.dumps([model.to_dict() for model in models], indent=2))
        else:
            _print_models(models)
        return 0
    if args.command == "hardware":
        profile = capture_hardware(include_hostname=args.include_hostname)
        if args.json:
            print(json.dumps(profile, indent=2))
        else:
            print(f"Hardware ID: {profile['hardware_id']}")
            print(f"CPU: {profile['cpu']} ({profile['architecture']}, {profile['logical_cores']} logical cores)")
            print(f"Memory: {bytes_label(profile['memory_bytes'])}")
            for gpu in profile.get("gpus") or []:
                print(f"GPU: {gpu.get('name')} ({gpu.get('backend')})")
        return 0
    if args.command == "doctor":
        return _doctor(args.tater_home)
    if args.command == "validate-suite":
        suite = load_suite(args.suite)
        print(f"OK {suite.get('name')} {suite.get('version')}: {len(suite.get('scenarios') or [])} scenarios")
        return 0
    if args.command == "report":
        outputs = write_reports(results_dir=args.results, markdown_path=args.markdown, docs_dir=args.docs)
        for label, path in outputs.items():
            print(f"{label}: {path}")
        return 0
    if args.command != "run":
        return 2
    if not args.all and not args.model:
        print("Choose at least one --model or use --all.", file=sys.stderr)
        return 2
    models = discover_models(home)
    jobs = _jobs(args, models)
    supported_spec = llama_supported_speculative_methods(home)
    unsupported_jobs = [
        (model, variant)
        for model, variant in jobs
        if variant.speculative and variant.speculative_method not in supported_spec
    ]
    if unsupported_jobs:
        for model, variant in unsupported_jobs:
            print(
                f"Skipping {model.label} [{variant.name}]: installed llama-server does not support "
                f"{variant.speculative_method}.",
                file=sys.stderr,
            )
        jobs = [job for job in jobs if job not in unsupported_jobs]
    if not jobs:
        print("No matching model/variant jobs were found. Use 'tater-bench models' to inspect discovery.", file=sys.stderr)
        return 2
    suite = load_suite(args.suite)
    print(f"Tater Bench will run {len(jobs)} model/variant job(s), one at a time:")
    for model, variant in jobs:
        print(f"  - {model.label} [{model.provider} / {variant.name}]")
    print("Close Tater before official runs so memory and speed measurements remain fair.")
    if not args.yes:
        if not sys.stdin.isatty():
            print("Use --yes for a non-interactive run.", file=sys.stderr)
            return 2
        answer = input("Start benchmark? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 0
    runner = BenchmarkRunner(
        home=home,
        context_size=args.context,
        repeats=args.repeat,
        max_tokens=args.max_tokens,
        seed=args.seed,
        progress=print,
    )
    batch = runner.run_batch(jobs, suite, limit=args.limit)
    output_path = save_batch(batch, args.output)
    print(f"Saved: {output_path}")
    outputs = write_reports(results_dir=args.output)
    print(f"Leaderboard: {outputs['markdown']}")
    failed = sum(run.get("status") != "complete" for run in batch.get("runs") or [])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
