import logging
from .base import BaseQuantizer

log = logging.getLogger(__name__)


class BitsAndBytesQuantizer(BaseQuantizer):
    """
    BitsAndBytes: In-flight NF4 (4-bit) or INT8 quantization.
    No calibration data needed. Lowest memory requirement.
    vLLM loads the model quantized on-the-fly from the original checkpoint.
    """

    method = "bitsandbytes"

    def quantize(self):
        """
        BitsAndBytes does not require a separate quantization pass.
        The model is quantized in-flight when loaded by vLLM.
        This method saves the config so vLLM knows the quant settings.
        """
        import json
        import shutil
        from transformers import AutoTokenizer

        self._ensure_output_dir()

        log.info("[BnB] Saving tokenizer config to %s", self.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, token=self.cfg.hf_token)
        tokenizer.save_pretrained(str(self.output_dir))

        # Write a marker config so serve.py knows what original model to load
        meta = {
            "original_model_id": self.cfg.model_id,
            "quantization": "bitsandbytes",
            "load_in_4bit": self.qcfg.bnb_load_in_4bit,
            "load_in_8bit": self.qcfg.bnb_load_in_8bit,
            "bnb_4bit_quant_type": self.qcfg.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.qcfg.bnb_4bit_use_double_quant,
        }
        with open(self.output_dir / "bnb_quant_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info("[BnB] No weight file generation needed — vLLM quantizes in-flight.")
        log.info("[BnB] Done.")

    def get_vllm_kwargs(self) -> dict:
        # vLLM loads directly from the original model ID with bnb config
        return {
            "model": self.cfg.model_id,
            "quantization": "bitsandbytes",
            "dtype": "float16",
            "load_format": "bitsandbytes",
        }
