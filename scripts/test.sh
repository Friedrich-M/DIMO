#!/usr/bin/env bash
# Render a trained model. Run from the repository root:  bash scripts/test.sh [mode] [extra key=value ...]
#   bash scripts/test.sh                                   # render 11-walk (edit render_videos below)
#   bash scripts/test.sh render render_videos=null         # render all motions (slow)
#   bash scripts/test.sh interpolation
#   bash scripts/test.sh language test_text_prompt="Trump is walking" text_encoder_ckpt=<projector.pth>
#       (the language mode needs a text -> latent projector; the released checkpoint has none, so
#        train one first with data_generation/train_text_projector.py -- see README.md)
#   bash scripts/test.sh fit_motion test_motion_data=data/new_motion
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mode=${1:-render}
shift || true

input_folder=data/trump_n51_step20          # data folder
scene_name=$(basename "$input_folder")
save_path=ckpts/${scene_name}               # checkpoint folder
video_save_dir=vis/${scene_name}/${mode}    # rendering results

python test.py mode="$mode" \
    input_folder="$input_folder" save_path="$save_path" video_save_dir="$video_save_dir" \
    test_stage=s2 test_azi=0 num_frames=21 ref_size=512 \
    render_videos=11-walk \
    "$@"
