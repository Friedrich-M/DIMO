"""Image-to-video generation with CogVideoX, the model the DIMO paper used for motion distillation.

Run inside an environment that has ``diffusers`` and ``transformers`` (see data_generation/README.md)::

    python generate_video.py --model zai-org/CogVideoX-5b-I2V --image ref.png --prompt "..." \
        --save_file out.mp4 [--num_frames 49] [--steps 50] [--guidance_scale 6.0] [--seed 42] [--offload]

The video is written at the model's native resolution; the caller restores the square aspect ratio of
the reference image (``datagen.video_gen``). Defaults match the settings used for the paper's data.
"""

import argparse

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="zai-org/CogVideoX-5b-I2V")
    parser.add_argument("--image", required=True, help="conditioning image (first frame)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--save_file", required=True)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--offload", action="store_true",
                        help="sequential CPU offload + VAE tiling/slicing, for GPUs under ~24 GB")
    args = parser.parse_args()

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    if args.offload:
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    else:
        pipe.to("cuda")

    frames = pipe(
        prompt=args.prompt,
        image=load_image(args.image),
        num_videos_per_prompt=1,
        num_inference_steps=args.steps,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]
    export_to_video(frames, args.save_file, fps=args.fps)
    print(f"[INFO] wrote {len(frames)} frames to {args.save_file}")


if __name__ == "__main__":
    main()
