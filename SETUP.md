# Quantization-vLLM 환경 설정 및 사용 가이드

Docker + docker-compose + Poetry를 사용해 GPU 가상환경과 이미지를 관리합니다.

---

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [빠른 시작](#빠른-시작)
3. [환경 변수 설정](#환경-변수-설정)
4. [Docker 이미지 빌드](#docker-이미지-빌드)
5. [양자화 실행](#양자화-실행)
6. [vLLM 서빙](#vllm-서빙)
7. [벤치마크 실행](#벤치마크-실행)
8. [RunPod 원격 실행](#runpod-원격-실행)
9. [로컬 개발 환경 (Poetry 직접 사용)](#로컬-개발-환경)
10. [파일 구조](#파일-구조)
11. [트러블슈팅](#트러블슈팅)

---

## 사전 요구사항

| 필수 소프트웨어 | 확인 방법 |
|---|---|
| Docker Engine 24+ | `docker --version` |
| Docker Compose v2 | `docker compose version` |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` |
| Poetry 1.8+ (로컬 개발 시) | `poetry --version` |

**NVIDIA Container Toolkit 설치 (미설치 시):**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 빠른 시작

```bash
# 1. 환경 변수 파일 준비
cp .env.example .env
# .env 파일을 열어 HF_TOKEN과 RUNPOD_API_KEY를 실제 값으로 수정

# 2. 이미지 빌드 (첫 빌드는 30~60분 소요 — CUDA 확장 컴파일 포함)
docker compose build

# 3. AWQ + GPTQ + FP8 양자화 실행 (로컬 GPU)
docker compose up quantize

# 4. 양자화 완료 후 vLLM API 서버 실행
docker compose up serve
```

---

## 환경 변수 설정

`.env` 파일을 생성합니다 (`.env.example` 참고):

```bash
cp .env.example .env
```

| 변수 | 설명 | 기본값 |
|---|---|---|
| `HF_TOKEN` | HuggingFace API 토큰 | 필수 |
| `RUNPOD_API_KEY` | RunPod API 키 (원격 실행 시 필수) | — |
| `METHODS` | 양자화 메서드 목록 (공백 구분) | `awq gptq fp8` |
| `METHOD` | 서빙할 메서드 | `awq` |
| `SERVE_PORT` | API 서버 포트 | `8000` |

---

## Docker 이미지 빌드

```bash
# 전체 빌드
docker compose build

# 캐시 없이 새로 빌드
docker compose build --no-cache

# 빌드 로그 상세 출력
docker compose build --progress=plain
```

> **참고:** 첫 빌드는 CUDA 확장(AWQ, GPTQ, BitsAndBytes 등) 컴파일로 인해 30~60분 소요됩니다.
> `pyproject.toml`/`poetry.lock`이 변경되지 않으면 이후 빌드는 레이어 캐시로 빠르게 완료됩니다.

---

## 양자화 실행

### 기본 (AWQ + GPTQ + FP8)

```bash
docker compose up quantize
```

### 특정 메서드만 실행

```bash
# 환경 변수로 메서드 지정
METHODS="awq fp8" docker compose up quantize

# 또는 .env 파일에 METHODS=awq fp8 로 설정 후 실행
docker compose up quantize
```

### 지원하는 양자화 메서드

| 메서드 | 설명 |
|---|---|
| `awq` | Activation-aware Weight Quantization (4-bit) |
| `gptq` | GPTQ (4-bit, calibration 기반) |
| `bnb` | BitsAndBytes (NF4/FP4) |
| `fp8` | FP8 Dynamic quantization |
| `int8_w8a8` | INT8 Weight+Activation (SmoothQuant) |
| `aqlm` | Additive Quantization of LLM Weights |
| `squeezellm` | SqueezeLLM (소스에서 별도 설치 필요) |

### 결과 확인

양자화된 모델은 `./quantized_models/` 디렉토리에 저장됩니다:
```
quantized_models/
└── MLP-KTLim--llama-3-Korean-Bllossom-8B_awq/
└── MLP-KTLim--llama-3-Korean-Bllossom-8B_gptq/
└── ...
```

---

## vLLM 서빙

### API 서버 시작

```bash
# AWQ 모델 서빙 (기본)
docker compose up serve

# GPTQ 모델 서빙
METHOD=gptq docker compose up serve

# 포트 변경
SERVE_PORT=9000 docker compose up serve
```

### API 호출 예시

```bash
# 상태 확인
curl http://localhost:8000/health

# 모델 목록
curl http://localhost:8000/v1/models

# 텍스트 생성 (OpenAI 호환)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    "prompt": "안녕하세요. 오늘 날씨는",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## 벤치마크 실행

양자화 + 벤치마크를 한 번에 실행합니다:

```bash
# AWQ + GPTQ + FP8 양자화 후 각 메서드별 추론 성능 측정
docker compose up benchmark

# 특정 메서드만
METHODS="awq gptq" docker compose up benchmark
```

벤치마크 결과는 `quantization_report.json`으로 저장되며, 터미널에도 표로 출력됩니다.

---

## RunPod 원격 실행

RunPod A100 Pod을 자동으로 생성하고 원격에서 양자화를 실행합니다.
`.env`에 `RUNPOD_API_KEY`가 설정되어 있어야 합니다.

```bash
# 컨테이너 내부에서 직접 실행 (--skip-runpod 플래그 없음 → RunPod 자동 생성)
docker compose run --rm quantize \
  poetry run python main.py --methods awq gptq

# 특정 메서드만 RunPod에서 실행
docker compose run --rm quantize \
  poetry run python main.py --methods fp8 int8_w8a8
```

RunPod 실행 흐름:
1. `RUNPOD_API_KEY`로 A100 Pod 생성
2. SSH로 연결 후 의존성 설치
3. 양자화 코드 업로드 및 실행
4. 결과(`quantized_models/`) 다운로드
5. Pod 자동 종료

---

## 로컬 개발 환경

Docker 없이 Poetry로 직접 가상환경을 관리합니다.

### Poetry 설치

```bash
curl -sSL https://install.python-poetry.org | python3
```

### 가상환경 생성 및 의존성 설치

```bash
cd /home/eeariorie/AI_ML_DL/Projects/Quantization_vLLM

# 의존성 설치 (.venv/ 디렉토리에 생성됨)
poetry install --no-root

# 가상환경 활성화
poetry shell
```

### 로컬 실행

```bash
# 가상환경 내에서 실행
poetry run python main.py --skip-runpod --methods awq

# 또는 shell 활성화 후
poetry shell
python main.py --skip-runpod --methods awq
```

### 의존성 추가/제거

```bash
# 패키지 추가
poetry add <패키지명>

# 개발용 패키지 추가
poetry add --group dev <패키지명>

# 패키지 제거
poetry remove <패키지명>

# lock 파일만 업데이트 (실제 설치 없이)
poetry lock

# lock 파일 기반으로 재설치
poetry install --no-root
```

### lock 파일 생성 (첫 설정 시)

```bash
# poetry.lock 파일 생성 (의존성 버전 고정)
poetry lock

# 이후 docker build 시 lock 파일 자동 사용
docker compose build
```

---

## 파일 구조

```
Quantization_vLLM/
├── Dockerfile              # CUDA 12.4.1 + Python 3.11 + Poetry 이미지
├── docker-compose.yml      # quantize / serve / benchmark 서비스 정의
├── pyproject.toml          # Poetry 의존성 관리 (CUDA 토치 소스 포함)
├── poetry.lock             # 고정된 의존성 버전 (git commit 권장)
├── .env                    # 실제 API 키 (git에 절대 커밋 금지)
├── .env.example            # 환경 변수 템플릿
├── .dockerignore           # Docker 빌드 컨텍스트 제외 목록
├── config.py               # 모델/RunPod/양자화 설정
├── main.py                 # 진입점 (양자화 + 서빙 오케스트레이터)
├── runpod_connect.py       # RunPod Pod 생성 및 SSH 관리
├── quantize/               # 양자화 메서드별 구현
│   ├── base.py
│   ├── awq.py
│   ├── gptq.py
│   ├── bnb.py
│   ├── fp8.py
│   ├── int8_w8a8.py
│   ├── aqlm.py
│   └── squeezellm.py
├── serve/                  # vLLM 추론 서버
│   └── vllm_server.py
└── quantized_models/       # 양자화 결과 저장 (자동 생성, git 제외)
```

---

## 트러블슈팅

### GPU를 인식하지 못할 때

```bash
# NVIDIA Container Toolkit 동작 확인
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# docker compose GPU 확인
docker compose run --rm quantize nvidia-smi
```

### `poetry.lock` 충돌 시

```bash
# lock 파일 재생성
poetry lock --no-update

# 또는 전체 재해석
rm poetry.lock
poetry lock
docker compose build --no-cache
```

### 빌드 중 CUDA 확장 컴파일 실패

```bash
# 빌드 로그 확인
docker compose build --progress=plain 2>&1 | tee build.log

# 메모리 부족 시 병렬 빌드 제한
MAX_JOBS=4 docker compose build
```

### vLLM 서버가 시작되지 않을 때

```bash
# 컨테이너 로그 확인
docker compose logs serve

# 직접 컨테이너에 접속해 디버깅
docker compose run --rm serve bash
poetry run python main.py --serve awq
```

### 포트 충돌

```bash
# 다른 포트로 서빙
SERVE_PORT=9000 docker compose up serve
```

---

## 메서드별 설정 변경

`config.py`에서 모델, 양자화 파라미터를 수정합니다:

```python
# 모델 변경
model_config = ModelConfig(model_id="meta-llama/Meta-Llama-3-8B")

# AWQ 비트 수 변경 (4-bit → 3-bit)
quant_config = QuantizationConfig(awq_bits=3)
```

변경 후 이미지를 다시 빌드하지 않아도 됩니다 — `quantized_models/` 볼륨 마운트 덕분에 결과가 로컬에 저장됩니다. 단, `poetry.lock`이나 `pyproject.toml`을 수정하면 `docker compose build`가 필요합니다.
