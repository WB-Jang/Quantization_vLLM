import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class VLLMInferenceServer:
    """Loads a quantized model with vLLM and runs inference."""

    def __init__(self, vllm_kwargs: dict[str, Any], gpu_memory_utilization: float = 0.90):
        self.vllm_kwargs = vllm_kwargs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None

    def load(self):
        from vllm import LLM

        log.info("Loading model with vLLM: %s", self.vllm_kwargs)
        self.llm = LLM(
            **self.vllm_kwargs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )
        log.info("Model loaded successfully.")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        from vllm import SamplingParams

        if not self.llm:
            raise RuntimeError("Model not loaded. Call load() first.")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def benchmark(self, prompt: str = "안녕하세요. 한국어로 자기소개를 해주세요.", n_runs: int = 5) -> dict:
        """Run a quick latency/throughput benchmark."""
        if not self.llm:
            raise RuntimeError("Model not loaded. Call load() first.")

        log.info("Benchmarking with %d runs...", n_runs)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.generate([prompt], max_tokens=128)
            latencies.append(time.perf_counter() - t0)

        avg_latency = sum(latencies) / len(latencies)
        return {
            "avg_latency_s": round(avg_latency, 3),
            "min_latency_s": round(min(latencies), 3),
            "max_latency_s": round(max(latencies), 3),
            "throughput_req_per_s": round(1 / avg_latency, 2),
        }


def serve_api(vllm_kwargs: dict[str, Any], host: str = "0.0.0.0", port: int = 8000):
    """Launch vLLM's built-in OpenAI-compatible HTTP server."""
    import subprocess
    import sys

    model = vllm_kwargs["model"]
    quantization = vllm_kwargs.get("quantization", "")
    dtype = vllm_kwargs.get("dtype", "auto")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--dtype", dtype,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if "load_format" in vllm_kwargs:
        cmd += ["--load-format", vllm_kwargs["load_format"]]

    log.info("Launching vLLM API server: %s", " ".join(cmd))
    subprocess.run(cmd)
