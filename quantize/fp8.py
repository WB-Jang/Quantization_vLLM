import logging
from .base import BaseQuantizer

log = logging.getLogger(__name__)


class FP8Quantizer(BaseQuantizer):
    """
    FP8 (E4M3) Weight+Activation quantization via llmcompressor.
    Best throughput with minimal quality loss.
    Requires NVIDIA Hopper GPU (H100) for maximum performance.
    Ampere (A100) runs FP8 in emulation mode.
    """

    method = "fp8"

    def quantize(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier

        self._ensure_output_dir()
        samples, tokenizer = self._load_calibration_data()

        log.info("[FP8] Loading model: %s", self.cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            device_map="auto",
            torch_dtype="auto",
            token=self.cfg.hf_token,
        )

        recipe = QuantizationModifier(
            targets="Linear",
            scheme=self.qcfg.fp8_scheme,       # "FP8_DYNAMIC" or "FP8"
            ignore=["lm_head"],
        )

        log.info("[FP8] Running one-shot FP8 calibration (%d samples)...", len(samples))
        oneshot(
            model=model,
            dataset=samples,
            recipe=recipe,
            max_seq_length=self.qcfg.sequence_length,
            num_calibration_samples=self.qcfg.calibration_samples,
        )

        log.info("[FP8] Saving to %s", self.output_dir)
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        log.info("[FP8] Done.")

    def get_vllm_kwargs(self) -> dict:
        return {
            "model": str(self.output_dir),
            "quantization": "fp8",
            "dtype": "bfloat16",
        }
