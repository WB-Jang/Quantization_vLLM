import logging
import subprocess
import sys
from pathlib import Path

from .base import BaseQuantizer

log = logging.getLogger(__name__)


class SqueezeLLMQuantizer(BaseQuantizer):
    """
    SqueezeLLM: Sparse + INT4 quantization with dense-and-sparse decomposition.
    Preserves extreme outlier weights in sparse fp16, rest in INT4.
    Install: pip install git+https://github.com/SqueezeAILab/SqueezeLLM.git
    """

    method = "squeezellm"

    def quantize(self):
        self._ensure_output_dir()

        # SqueezeLLM requires generating a sensitivity file first
        sensitivity_path = self.output_dir / "sensitivity.pt"

        log.info("[SqueezeLLM] Step 1/2 — Computing weight sensitivity...")
        sensitivity_cmd = [
            sys.executable, "-m", "squeezellm.sparse_generate",
            "--model", self.cfg.model_id,
            "--output", str(sensitivity_path),
            "--nsamples", str(self.qcfg.calibration_samples),
            "--seqlen", str(self.qcfg.sequence_length),
            "--dataset", "wikitext2",
        ]
        if self.cfg.hf_token:
            sensitivity_cmd += ["--token", self.cfg.hf_token]

        result = subprocess.run(sensitivity_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError("SqueezeLLM sensitivity computation failed.")

        log.info("[SqueezeLLM] Step 2/2 — Running INT%d quantization...", self.qcfg.squeezellm_bits)
        quant_cmd = [
            sys.executable, "-m", "squeezellm.quant_sequential",
            "--model", self.cfg.model_id,
            "--wbits", str(self.qcfg.squeezellm_bits),
            "--sparse_threshold", "0.45",
            "--sensitivity", str(sensitivity_path),
            "--save", str(self.output_dir / "quantized_model.pt"),
            "--nsamples", str(self.qcfg.calibration_samples),
            "--seqlen", str(self.qcfg.sequence_length),
        ]
        if self.cfg.hf_token:
            quant_cmd += ["--token", self.cfg.hf_token]

        result = subprocess.run(quant_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError("SqueezeLLM quantization failed.")

        # Save tokenizer alongside
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, token=self.cfg.hf_token)
        tokenizer.save_pretrained(str(self.output_dir))

        log.info("[SqueezeLLM] Done. Saved to %s", self.output_dir)

    def get_vllm_kwargs(self) -> dict:
        return {
            "model": self.cfg.model_id,
            "quantization": "squeezellm",
            "dtype": "float16",
        }
