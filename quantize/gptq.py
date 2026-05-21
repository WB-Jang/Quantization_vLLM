import logging
from .base import BaseQuantizer

log = logging.getLogger(__name__)


class GPTQQuantizer(BaseQuantizer):
    """
    GPTQ: Layer-by-layer weight quantization (INT4 W4A16).
    High compatibility; slightly lower quality than AWQ in most benchmarks.
    """

    method = "gptq"

    def quantize(self):
        from gptqmodel import GPTQModel, QuantizeConfig

        self._ensure_output_dir()
        samples, tokenizer = self._load_calibration_data()

        quant_config = QuantizeConfig(
            bits=self.qcfg.gptq_bits,
            group_size=self.qcfg.gptq_group_size,
            damp_percent=self.qcfg.gptq_damp_percent,
            desc_act=self.qcfg.gptq_desc_act,
        )

        log.info("[GPTQ] Loading model: %s", self.cfg.model_id)
        model = GPTQModel.load(
            self.cfg.model_id,
            quant_config,
            token=self.cfg.hf_token,
        )

        log.info("[GPTQ] Preparing calibration data (%d samples)...", len(samples))
        tokenized = [
            tokenizer(
                s,
                return_tensors="pt",
                max_length=self.qcfg.sequence_length,
                truncation=True,
            )
            for s in samples
        ]

        log.info("[GPTQ] Running quantization (bits=%d, group_size=%d)...", self.qcfg.gptq_bits, self.qcfg.gptq_group_size)
        model.quantize(tokenized)

        log.info("[GPTQ] Saving to %s", self.output_dir)
        model.save(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        log.info("[GPTQ] Done.")

    def get_vllm_kwargs(self) -> dict:
        return {
            "model": str(self.output_dir),
            "quantization": "gptq",
            "dtype": "float16",
        }
