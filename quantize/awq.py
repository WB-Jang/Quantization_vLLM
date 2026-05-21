import logging
from .base import BaseQuantizer

log = logging.getLogger(__name__)


class AWQQuantizer(BaseQuantizer):
    """
    Activation-aware Weight Quantization (INT4 W4A16).
    Best quality INT4 format for vLLM. Requires calibration data.
    """

    method = "awq"

    def quantize(self):
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        self._ensure_output_dir()
        log.info("[AWQ] Loading model: %s", self.cfg.model_id)

        model = AutoAWQForCausalLM.from_pretrained(
            self.cfg.model_id,
            token=self.cfg.hf_token,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            token=self.cfg.hf_token,
        )

        quant_config = {
            "zero_point": self.qcfg.awq_zero_point,
            "q_group_size": self.qcfg.awq_group_size,
            "w_bit": self.qcfg.awq_bits,
            "version": "GEMM",
        }

        log.info("[AWQ] Running quantization (bits=%d, group_size=%d)...", self.qcfg.awq_bits, self.qcfg.awq_group_size)
        model.quantize(tokenizer, quant_config=quant_config)

        log.info("[AWQ] Saving to %s", self.output_dir)
        model.save_quantized(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        log.info("[AWQ] Done.")

    def get_vllm_kwargs(self) -> dict:
        return {
            "model": str(self.output_dir),
            "quantization": "awq",
            "dtype": "float16",
        }
