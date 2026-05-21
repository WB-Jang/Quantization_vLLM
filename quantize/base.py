import logging
from abc import ABC, abstractmethod
from pathlib import Path

from config import ModelConfig, QuantizationConfig, model_config, quant_config

log = logging.getLogger(__name__)


class BaseQuantizer(ABC):
    method: str = ""

    def __init__(self, cfg: ModelConfig = model_config, qcfg: QuantizationConfig = quant_config):
        self.cfg = cfg
        self.qcfg = qcfg
        self.output_dir = Path(cfg.output_base_dir) / f"{self._safe_model_name()}_{self.method}"

    def _safe_model_name(self) -> str:
        return self.cfg.model_id.replace("/", "--")

    def _ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_calibration_data(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        log.info("Loading calibration dataset: %s/%s", self.qcfg.calibration_dataset, self.qcfg.calibration_dataset_config)
        dataset = load_dataset(
            self.qcfg.calibration_dataset,
            self.qcfg.calibration_dataset_config,
            split="train",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, token=self.cfg.hf_token)

        samples = []
        for text in dataset["text"]:
            if not text.strip():
                continue
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=self.qcfg.sequence_length,
                truncation=True,
            )
            if enc["input_ids"].shape[1] >= self.qcfg.sequence_length:
                samples.append(text)
            if len(samples) >= self.qcfg.calibration_samples:
                break

        return samples, tokenizer

    @abstractmethod
    def quantize(self):
        ...

    @abstractmethod
    def get_vllm_kwargs(self) -> dict:
        """Return kwargs to pass to vllm.LLM() for this quantization."""
        ...
