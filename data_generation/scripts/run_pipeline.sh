#!/usr/bin/env bash
# Run the five data-generation stages for one object.
#   bash scripts/run_pipeline.sh <object_name> <reference_image> [extra key=value overrides...]
# Set the interpreters / checkpoints of the Wan2.2 and SV4D environments below or via overrides.
set -euo pipefail
cd "$(dirname "$0")/.."

name=${1:?object name}
image=${2:?reference image}
shift 2

WAN_PYTHON=${WAN_PYTHON:-python}
WAN_CKPT=${WAN_CKPT:-/models/Wan2.2-TI2V-5B}
SV4D_PYTHON=${SV4D_PYTHON:-python}

common=(object.name="$name" "$@")
python generate_captions.py  "${common[@]}" object.image="$image"
python generate_videos.py    "${common[@]}" wan.python="$WAN_PYTHON" wan.ckpt_dir="$WAN_CKPT"
python filter_videos.py      "${common[@]}"
python generate_multiview.py "${common[@]}" sv4d.python="$SV4D_PYTHON"
python build_dataset.py      "${common[@]}"
