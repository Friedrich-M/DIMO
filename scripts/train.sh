#!/usr/bin/env bash
# Train DIMO on the example data. Run from the repository root:  bash scripts/train.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

input_folder=${1:-data/trump_n51_step20}    # data folder
scene_name=$(basename "$input_folder")
save_path=outputs/${scene_name}             # checkpoint folder (reused if it exists)

python train.py \
    input_folder="$input_folder" save_path="$save_path" \
    num_frames=21 ref_size=512 batch_size=2 \
    num_cpts=512 latent_code_dim=32 \
    iters_s1=2800 iters_s2=10000 \
    density_start_iter=200 density_end_iter=2000 densification_interval=100 \
    densify_opacity_threshold_s1=0.02 densify_grad_threshold=0.02 \
    arap_start_iter_s1=2000 arap_end_iter_s2=5000
