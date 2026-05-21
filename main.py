"""
Orchestrator: connects to RunPod, runs all quantization methods,
benchmarks each with vLLM, and prints a comparison report.

Usage:
    python main.py                          # run all methods
    python main.py --methods awq gptq fp8   # run specific methods
    python main.py --skip-runpod            # run locally (no pod creation)
    python main.py --serve awq              # serve one method via API
"""

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALL_METHODS = ["awq", "gptq", "bnb", "fp8", "int8_w8a8", "aqlm", "squeezellm"]

QUANTIZER_MAP = {
    "awq":       ("quantize.awq",        "AWQQuantizer"),
    "gptq":      ("quantize.gptq",       "GPTQQuantizer"),
    "bnb":       ("quantize.bnb",        "BitsAndBytesQuantizer"),
    "fp8":       ("quantize.fp8",        "FP8Quantizer"),
    "int8_w8a8": ("quantize.int8_w8a8",  "INT8W8A8Quantizer"),
    "aqlm":      ("quantize.aqlm",       "AQLMQuantizer"),
    "squeezellm":("quantize.squeezellm", "SqueezeLLMQuantizer"),
}


def get_quantizer(method: str):
    module_path, class_name = QUANTIZER_MAP[method]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def run_quantization(methods: list[str]) -> dict[str, dict]:
    results = {}
    for method in methods:
        log.info("=" * 60)
        log.info("Quantizing with method: %s", method.upper())
        log.info("=" * 60)
        try:
            quantizer = get_quantizer(method)
            quantizer.quantize()
            results[method] = {"status": "success", "vllm_kwargs": quantizer.get_vllm_kwargs()}
        except Exception as exc:
            log.error("[%s] Quantization failed: %s", method, exc, exc_info=True)
            results[method] = {"status": "failed", "error": str(exc)}
    return results


def run_benchmarks(results: dict[str, dict]) -> dict[str, dict]:
    from serve.vllm_server import VLLMInferenceServer

    bench_results = {}
    for method, info in results.items():
        if info["status"] != "success":
            bench_results[method] = {"status": "skipped"}
            continue

        log.info("Benchmarking: %s", method.upper())
        try:
            server = VLLMInferenceServer(info["vllm_kwargs"])
            server.load()
            bench = server.benchmark()
            bench_results[method] = {"status": "success", **bench}
            log.info("  [%s] %s", method, bench)
        except Exception as exc:
            log.error("[%s] Benchmark failed: %s", method, exc)
            bench_results[method] = {"status": "failed", "error": str(exc)}

    return bench_results


def print_report(quant_results: dict, bench_results: dict):
    print("\n" + "=" * 70)
    print("QUANTIZATION + BENCHMARK REPORT")
    print("=" * 70)
    header = f"{'Method':<14} {'Quant':<10} {'Bench':<10} {'Avg Lat (s)':<14} {'Throughput':<12}"
    print(header)
    print("-" * 70)
    for method in ALL_METHODS:
        q = quant_results.get(method, {})
        b = bench_results.get(method, {})
        q_status = q.get("status", "not run")
        b_status = b.get("status", "not run")
        lat = f"{b.get('avg_latency_s', '-')}"
        tput = f"{b.get('throughput_req_per_s', '-')} req/s"
        print(f"{method:<14} {q_status:<10} {b_status:<10} {lat:<14} {tput:<12}")
    print("=" * 70)


def remote_run(methods: list[str], run_benchmarks_flag: bool):
    """
    Upload and execute this script on a RunPod pod via SSH.
    """
    from runpod_connect import RunPodSession

    script = textwrap.dedent(f"""
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "vllm", "autoawq", "gptqmodel", "bitsandbytes",
            "llmcompressor", "aqlm[gpu]", "datasets", "tqdm"])
        subprocess.check_call([
            sys.executable, "main.py",
            "--skip-runpod",
            {"'--methods'" if methods != ALL_METHODS else ""},
            {", ".join(f"'{m}'" for m in methods) if methods != ALL_METHODS else ""},
            {"'--benchmark'" if run_benchmarks_flag else ""},
        ])
    """)

    with RunPodSession() as session:
        log.info("Uploading project files to pod...")
        _upload_project(session)
        log.info("Running quantization on pod...")
        exit_code = session.exec_script(script)
        if exit_code != 0:
            log.error("Remote execution failed with exit code %d", exit_code)
            sys.exit(1)
        log.info("Downloading results from pod...")
        _download_results(session)


def _upload_project(session):
    import os
    sftp = session.ssh.open_sftp()
    root = Path(__file__).parent

    sftp.mkdir("/root/quantization_project")
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        remote = f"/root/quantization_project/{rel}"
        remote_dir = str(Path(remote).parent)
        try:
            sftp.mkdir(remote_dir)
        except OSError:
            pass
        sftp.put(str(path), remote)

    for fname in [".env", "requirements.txt"]:
        local = root / fname
        if local.exists():
            sftp.put(str(local), f"/root/quantization_project/{fname}")

    sftp.close()


def _download_results(session):
    import tarfile
    _, _, _ = session.exec("tar -czf /tmp/quantized_models.tar.gz -C /root/quantization_project quantized_models")
    sftp = session.ssh.open_sftp()
    local_tar = "/tmp/quantized_models.tar.gz"
    sftp.get("/tmp/quantized_models.tar.gz", local_tar)
    sftp.close()
    with tarfile.open(local_tar) as tar:
        tar.extractall(Path(__file__).parent)
    log.info("Results downloaded to ./quantized_models/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--skip-runpod", action="store_true", help="Run locally without creating a pod")
    parser.add_argument("--benchmark", action="store_true", help="Run vLLM inference benchmark after quantization")
    parser.add_argument("--serve", choices=ALL_METHODS, help="Serve a single method via OpenAI-compatible API")
    args = parser.parse_args()

    if args.serve:
        quantizer = get_quantizer(args.serve)
        from serve.vllm_server import serve_api
        serve_api(quantizer.get_vllm_kwargs())
        return

    if args.skip_runpod:
        quant_results = run_quantization(args.methods)
        bench_results = run_benchmarks(quant_results) if args.benchmark else {}
        print_report(quant_results, bench_results)

        report_path = Path("quantization_report.json")
        with open(report_path, "w") as f:
            json.dump({"quantization": quant_results, "benchmarks": bench_results}, f, indent=2)
        log.info("Report saved to %s", report_path)
    else:
        remote_run(args.methods, args.benchmark)


if __name__ == "__main__":
    main()
