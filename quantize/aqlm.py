import logging
from .base import BaseQuantizer

log = logging.getLogger(__name__)


class AQLMQuantizer(BaseQuantizer):
    """
    AQLM: Additive Quantization of Language Models (2~4 bit).
    Extreme compression via learned vector codebooks.
    High quality at very low bitrates but slow quantization.
    """

    method = "aqlm"

    def quantize(self):
        import subprocess
        import sys

        self._ensure_output_dir()
        samples, _ = self._load_calibration_data()

        # AQLM quantization is run via its own CLI tool
        # num_codebooks=1, nbits=16 → effective ~2 bits per weight
        cmd = [
            sys.executable, "-m", "aqlm.convert",
            "--model", self.cfg.model_id,
            "--output", str(self.output_dir),
            "--num_codebooks", str(self.qcfg.aqlm_num_codebooks),
            "--nbits_per_codebook", str(self.qcfg.aqlm_nbits_per_codebook),
            "--in_group_size", str(self.qcfg.aqlm_in_group_size),
            "--nsamples", str(self.qcfg.calibration_samples),
            "--seqlen", str(self.qcfg.sequence_length),
            "--dataset", "wikitext2",
            "--save", str(self.output_dir),
        ]
        if self.cfg.hf_token:
            cmd += ["--token", self.cfg.hf_token]

        log.info("[AQLM] Launching quantization CLI: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"AQLM quantization failed with exit code {result.returncode}")

        log.info("[AQLM] Done. Saved to %s", self.output_dir)

    def get_vllm_kwargs(self) -> dict:
        return {
            "model": str(self.output_dir),
            "quantization": "aqlm",
            "dtype": "float16",
        }
