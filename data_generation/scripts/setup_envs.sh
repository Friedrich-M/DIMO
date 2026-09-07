#!/usr/bin/env bash
# Create the extra conda environments the data-generation pipeline needs.
#
#   bash scripts/setup_envs.sh [wan|sv4d|captioner|all] [--prefix DIR]
#
# Most stages run in the `dimo` training environment (plus `pip install -r requirements.txt`).
# Separate environments are only created where the dependencies genuinely conflict:
#
#   wan        stage 2 (required)  Wan2.2 TI2V-5B: torch 2.6 + flash-attn. Its `diffusers` also
#                                  covers the CogVideoX backend (video.backend=cogvideox).
#   sv4d       stage 4 (required)  SV4D 2.0: torch 2.6 + xformers
#   captioner  stage 1 (optional)  only for `llm.backend=local`, which needs a transformers release
#                                  new enough for Qwen3.5 and so cannot share dimo's pinned 4.33.
#                                  Not needed when the captioner is reached over an API
#                                  (`llm.backend=openai`, the default).
#
# Versions below are the ones this pipeline was tested with.
set -euo pipefail

WHICH=${1:-all}
PREFIX_DIR=${CONDA_ENV_DIR:-$HOME/.conda/envs}
[ "${2:-}" = "--prefix" ] && PREFIX_DIR=${3:?missing prefix dir}

eval "$(conda shell.bash hook)"
make_env() { conda create -y -p "$PREFIX_DIR/$1" "python=$2" >/dev/null && conda activate "$PREFIX_DIR/$1"; }

if [ "$WHICH" = "wan" ] || [ "$WHICH" = "all" ]; then
  echo "=== wan (stage 2: Wan2.2 TI2V-5B) ==="
  make_env wan22 3.11
  pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
  pip install -q "transformers>=4.49.0,<=4.51.3" tokenizers accelerate diffusers "numpy<2" \
                 opencv-python tqdm imageio imageio-ffmpeg easydict ftfy einops safetensors
  # wan/__init__.py imports every task, so the s2v / animate dependencies are needed too.
  pip install -q decord peft librosa dashscope
  pip install -q sentencepiece   # T5 tokenizer, needed by the CogVideoX backend
  # Required: the transformer calls flash_attention() directly, bypassing Wan's PyTorch fallback.
  # The wheel must match the torch build; the 2.8.x wheels for torch 2.6 fail to import
  # (they reference a libc10 symbol torch 2.6.0 does not export).
  pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
  python -c "import torch, flash_attn; print('wan22: torch', torch.__version__, 'flash_attn', flash_attn.__version__)"
  conda deactivate
fi

if [ "$WHICH" = "sv4d" ] || [ "$WHICH" = "all" ]; then
  echo "=== sv4d (stage 4: SV4D 2.0) ==="
  make_env sv4d 3.11
  pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  # xformers is required: the SV4D VAE calls its memory-efficient attention directly.
  pip install -q xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124
  pip install -q "numpy<2" omegaconf einops fire imageio imageio-ffmpeg opencv-python-headless \
                 safetensors pytorch-lightning kornia scipy tqdm open-clip-torch rembg onnxruntime \
                 pillow "transformers<5" "git+https://github.com/openai/CLIP.git"
  python -c "import torch, xformers, open_clip; print('sv4d: torch', torch.__version__, 'xformers', xformers.__version__)"
  # The OpenCLIP image conditioner is fetched on first use; warm it while the network is available.
  python -c "import open_clip; open_clip.create_model_and_transforms('ViT-H-14', device='cpu', pretrained='laion2b_s32b_b79k')"
  conda deactivate
fi

if [ "$WHICH" = "captioner" ]; then
  echo "=== captioner (stage 1 with llm.backend=local) ==="
  make_env datagen 3.11
  pip install -q torch torchvision
  pip install -q "git+https://github.com/huggingface/transformers.git" accelerate
  pip install -q "numpy<2" pillow opencv-python-headless imageio imageio-ffmpeg omegaconf \
                 rembg onnxruntime openai fire
  python -c "import torch, transformers; print('datagen: torch', torch.__version__, 'transformers', transformers.__version__)"
  python -c "import rembg; rembg.new_session('u2net')"
  conda deactivate
fi

echo
echo "Done. Point the config at the interpreters, e.g.:"
echo "  wan.python=$PREFIX_DIR/wan22/bin/python  sv4d.python=$PREFIX_DIR/sv4d/bin/python"
