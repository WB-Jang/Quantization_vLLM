import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    model_id: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    output_base_dir: str = "./quantized_models"
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))


@dataclass
class RunPodConfig:
    api_key: str = field(default_factory=lambda: os.getenv("RUNPOD_API_KEY", ""))
    # GPU options: "NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"
    gpu_type_id: str = "NVIDIA A100 80GB PCIe"
    gpu_count: int = 1
    container_disk_in_gb: int = 100
    volume_in_gb: int = 200
    image_name: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    pod_name: str = "llama-korean-quantization"
    # SSH connection timeout settings
    ssh_timeout: int = 300
    pod_ready_timeout: int = 600


@dataclass
class QuantizationConfig:
    # Calibration dataset
    calibration_dataset: str = "wikitext"
    calibration_dataset_config: str = "wikitext-2-raw-v1"
    calibration_samples: int = 512
    sequence_length: int = 2048

    # AWQ
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True

    # GPTQ
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_damp_percent: float = 0.01
    gptq_desc_act: bool = False

    # BitsAndBytes
    bnb_load_in_4bit: bool = True
    bnb_load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True

    # FP8
    fp8_scheme: str = "FP8_DYNAMIC"

    # INT8 W8A8
    int8_smoothquant_alpha: float = 0.5

    # AQLM
    aqlm_num_codebooks: int = 1
    aqlm_nbits_per_codebook: int = 16
    aqlm_in_group_size: int = 8

    # SqueezeLLM
    squeezellm_bits: int = 4


model_config = ModelConfig()
runpod_config = RunPodConfig()
quant_config = QuantizationConfig()
