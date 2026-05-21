import logging
from .base import BaseQuantizer

log = logging.getLogger(__name__)


class INT8W8A8Quantizer(BaseQuantizer):
    """
    INT8 W8A8 (SmoothQuant) quantization via llmcompressor.
    Quantizes both weights and activations to INT8.
    Requires NVIDIA GPU with Compute Capability >= 7.5 (Turing+).
    """

    method = "int8_w8a8"

    def quantize(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

        self._ensure_output_dir()
        samples, tokenizer = self._load_calibration_data()

        log.info("[INT8 W8A8] Loading model: %s", self.cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            device_map="auto",
            torch_dtype="auto",
            token=self.cfg.hf_token,
        )

        # SmoothQuant migrates activation outliers to weights before INT8 quant
        recipe = [
            SmoothQuantModifier(smoothing_strength=self.qcfg.int8_smoothquant_alpha),
            QuantizationModifier(
                targets="Linear",
                scheme="W8A8",
                ignore=["lm_head"],
            ),
        ]

        log.info("[INT8 W8A8] Running one-shot INT8 calibration (%d samples)...", len(samples))
        oneshot(
            model=model,
            dataset=samples,
            recipe=recipe,
            max_seq_length=self.qcfg.sequence_length,
            num_calibration_samples=self.qcfg.calibration_samples,
        )

        log.info("[INT8 W8A8] Saving to %s", self.output_dir)
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        log.info("[INT8 W8A8] Done.")

    def get_vllm_kwargs(self) -> dict:
        return {
            "model": str(self.output_dir),
            "quantization": "compressed-tensors",
            "dtype": "bfloat16",
        }
