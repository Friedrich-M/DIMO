# Data generation for DIMO

This folder distills the training data DIMO needs from a **single image** of an object, following Sec. 3.1 of the paper ("Motion Priors Distillation from Video Models"):

| Stage | Script | Model | What it does |
|---|---|---|---|
| 1 | `generate_captions.py` | [GPT-5](https://platform.openai.com/docs/models) (default) or [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), served or in-process | Structured description of the object ("meta" prompt), then ≥50 diverse motion captions that share the same appearance text and differ only after *"As time progresses"*, plus a short phrase per motion ("lift the right hand") for language-guided generation. |
| 2 | `generate_videos.py` | [Wan2.2](https://github.com/Wan-Video/Wan2.2) TI2V-5B (default) or [CogVideoX](https://github.com/zai-org/CogVideo)-5B-I2V | One image-to-video clip per caption, all starting from the same reference frame. |
| 3 | `filter_videos.py` | RAFT optical flow (+ optional VLM judge) | Extracts `clips.num_frames` (41 by default) object-centred RGBA frames per clip, removes clips with too little / too much motion and, optionally, low visual quality, identity drift or prompt mismatch. |
| 4 | `generate_multiview.py` | [SV4D 2.0](https://github.com/Stability-AI/generative-models) (default) or SV4D 1.0 | 8 novel views x 41 frames per accepted clip (view 0 = input), written directly as `view_XX/FF.png`. |
| 5 | `build_dataset.py` | rembg or a white-background matte | Assembles `<motion>/view_XX/FF.png`, `FF_mask.npy`, `info.json` and `captions.json` for `train.py`. |
| 6 | `train_text_projector.py` (optional) | BERT | Trains the text -> latent projector (`mlp_encoder.pth`) for `test.py mode=language` from a trained DIMO model. |

Every stage is resumable: existing outputs are skipped unless `force=True`, and stages can be re-run after adding motions. All options live in `configs/default.yaml` and are overridden as `section.key=value` on the command line. Intermediate results are stored under `<workdir>/<object.name>/`:

```
work/<object>/
  reference.png, reference_white.png   object-centred reference canvas (RGBA / on white)
  description.json                     structured object description
  motions.json                         motion captions: name, motion_type, short phrase, caption
  videos/<motion>.mp4 (+ .json)        Wan2.2 clips (raw outputs under videos/raw/)
  clips/<motion>/FF.png                `clips.num_frames` RGBA frames per clip, 576x576
  filter_report.json                   motion scores, judge scores, accepted / rejected
  multiview/<model>/<motion>/view_XX/FF.png   SV4D output (per model, so layouts cannot be mixed);
                                              multiview/<model>/info.json holds the camera layout
```

## Environments

Most stages run in the **`dimo`** training environment from the main README, plus `pip install -r data_generation/requirements.txt`. Two extra environments are unavoidable because their dependencies conflict with it, and one more is optional:

| Environment | Stages | Why separate |
|---|---|---|
| `dimo` | 1, 3, 5, 6 | captions over an API, clip extraction, filtering, dataset assembly, text projector |
| `wan22` | 2 | Wan2.2 needs torch >= 2.4 and flash-attn; it also has the `diffusers` needed by the CogVideoX backend |
| `sv4d` | 4 | SV4D needs its own torch and xformers |
| `datagen` | 1 (optional) | only for `llm.backend=local`: Qwen3.5 needs a transformers release newer than dimo's pinned 4.33 |

None of the extra environments is needed for the default caption path, which reaches GPT-5 over the API from the `dimo` environment. `bash scripts/setup_envs.sh all` creates the required two; add `bash scripts/setup_envs.sh captioner` for the optional `datagen` one (the selector is `captioner`, the environment it creates is named `datagen`). All are pinned to the versions this pipeline was tested against. The `sv4d.python` and `wan.python` config keys point the pipeline at those interpreters, so you never have to switch environments by hand.

On a cluster whose compute nodes have no network access, warm the caches once from a login node (the setup script does this for the environments it creates):
```bash
python -c "import rembg; rembg.new_session('u2net')"                 # ~/.u2net/u2net.onnx
python -c "from torchvision.models.optical_flow import raft_small, Raft_Small_Weights; raft_small(weights=Raft_Small_Weights.DEFAULT)"
python -c "import lpips; lpips.LPIPS(net='vgg')"                     # used by train.py / test.py
python -c "from transformers import AutoModel, AutoTokenizer; [c.from_pretrained('bert-base-cased', cache_dir='../ckpts/hf_cache') for c in (AutoTokenizer, AutoModel)]"
```
Pass `projector.bert_cache_dir=ckpts/hf_cache` to `train_text_projector.py` (and `bert_cache_dir=` to `test.py mode=language`) so stage 6 reads BERT from that cache instead of trying to download it, and export `HF_HUB_OFFLINE=1` so transformers does not spend half a minute retrying the hub before falling back to the cache.

Stage 5 precomputes the `FF_mask.npy` masks so training never has to. `dataset.mask_method=white_bg` derives them from the distance to white instead of running the matting network, which is about 30x faster and exact for these renders, since they all sit on a pure white background; it must match `mask_method` in the DIMO training config, so that a mask cached here and one computed at training time agree. The matting model is pinned to `u2net` (recent rembg releases changed the default); override it with `$DIMO_REMBG_MODEL` in both this pipeline and `dimo`.

### Video models (stage 2)

`video.backend` selects the image-to-video model:

| Backend | Model | Output | Notes |
|---|---|---|---|
| `wan` (default) | Wan2.2 TI2V-5B | 121 frames @ 24 fps, 1280x704 | needs a checkout of the Wan2.2 repo and flash-attn |
| `cogvideox` | CogVideoX-5B-I2V | 49 frames @ 8 fps, 480x720 | what the paper used; pure `diffusers` (plus `sentencepiece` for its T5 tokenizer), no repo checkout |

Both take the same square reference canvas and return a square, object-centred clip: Wan's frames are padded and un-padded around its 1280:704 aspect, CogVideoX's are squashed and restored.

```bash
python generate_videos.py object.name=trump video.backend=cogvideox \
    cogvideox.python=/envs/wan22/bin/python cogvideox.model=ckpts/CogVideoX-5b-I2V
```

### Checkpoints

```bash
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ckpts/Wan2.2-TI2V-5B            # ~32 GB, 24 GB GPU
hf download zai-org/CogVideoX-5b-I2V --local-dir ckpts/CogVideoX-5b-I2V       # ~20 GB, only for video.backend=cogvideox
hf download stabilityai/sv4d2.0 sv4d2_8views.safetensors --local-dir ckpts/sv4d   # 12 GB, 8 novel views
hf download stabilityai/sv4d2.0 sv4d2.safetensors        --local-dir ckpts/sv4d   # 12 GB, 4 novel views
```
SV4D 2.0 generates the novel views directly, so no SV3D is involved. SV4D 1.0 (`sv4d.model=sv4d`) is still supported but needs two extra checkpoints, and both are gated on the Hub, so accept the licences on their model pages first:
```bash
hf download stabilityai/sv4d sv4d.safetensors    --local-dir ckpts/sv4d   # huggingface.co/stabilityai/sv4d
hf download stabilityai/sv3d sv3d_u.safetensors  --local-dir ckpts/sv4d   # huggingface.co/stabilityai/sv3d
```

### Captions: GPT-5 or Qwen3.5-9B

Stage 1 (and the optional judge in stage 3) needs a vision-language model. Three setups work, all through the same client; whichever you pick, run `python check_llm.py` first, which sends three short requests and reports what the endpoint accepted.

**GPT-5 through the OpenAI API** (the default). Export `OPENAI_API_KEY`; the shipped config already selects it:
```bash
export OPENAI_API_KEY=...
python check_llm.py                                   # llm.model=gpt-5, llm.base_url=null
```
`llm.reasoning_effort` (`minimal`, `low`, `medium`, `high`) trades cost for care; `low` is the default and is enough for these prompts. Any GPT-5 or GPT-4o model works, e.g. `llm.model=gpt-5-mini` for a cheaper run.

**Qwen3.5-9B served locally** (fully open, no API key):
```bash
vllm serve Qwen/Qwen3.5-9B --port 8000 --max-model-len 32768 --reasoning-parser qwen3
python check_llm.py llm.model=Qwen/Qwen3.5-9B llm.base_url=http://localhost:8000/v1
```
**Qwen3.5-9B in-process**, with no server: `llm.backend=local llm.model=Qwen/Qwen3.5-9B`, from the optional `datagen` environment.

The sampling parameters are *negotiated* rather than configured, because the three targets disagree about them: GPT-5 requires `max_completion_tokens` instead of `max_tokens`, rejects `top_p` and the penalties, and accepts only `temperature=1`, while GPT-4o rejects `reasoning_effort`. The client guesses from the model name, then drops whatever the endpoint returns a 400 for and remembers the drop, so no per-model configuration is needed. `llm.temperature`/`llm.top_p` default to Qwen3.5's recommended non-thinking values (0.7 / 0.8, with `top_k=20` and `presence_penalty=1.5` added for Qwen) and are simply ignored where they are not supported. One consequence worth knowing: on GPT-5, `llm.max_tokens` is a budget that also pays for hidden reasoning tokens, so the client raises it automatically if reasoning consumes the lot.

## Running

```bash
cd data_generation
python generate_captions.py  object.name=trump object.image=/path/trump.png
python generate_videos.py    object.name=trump wan.python=/envs/wan/bin/python wan.ckpt_dir=/models/Wan2.2-TI2V-5B
python filter_videos.py      object.name=trump                      # add filter.judge=vlm to use the VLM judge
python generate_multiview.py object.name=trump sv4d.python=/envs/sv4d/bin/python
python build_dataset.py      object.name=trump                      # -> ../data/trump
cd .. && python train.py input_folder=data/trump save_path=outputs/trump
cd data_generation && python train_text_projector.py object.name=trump projector.checkpoint_dir=../outputs/trump/s2
```
`bash scripts/run_pipeline.sh trump /path/trump.png [key=value ...]` chains the five data stages in one command (set `WAN_PYTHON`, `WAN_CKPT` and `SV4D_PYTHON` in the environment to point at the other interpreters and checkpoints). Stages 2 and 4 need a GPU (24 GB is enough for `ti2v-5B` and SV4D at the default batch sizes); stages 1 and 3 need one only for a local VLM or the judge.

## Design notes

- **Prompts.** Stage 1 reproduces the auto-prompting recipe used for the paper: a structured "meta" description of appearance / expression / initial state seeds a template caption, and the captioner is asked for batches of 20 captions that keep the text before *"As time progresses"* identical (so every video starts from the same object state) while listing the motions already produced to avoid repeats. Set `captions.cluster_to` to over-generate and keep k-means representatives in BERT space, as in the paper's language experiments.
- **Camera layout.** SV4D 2.0 (`sv4d2_8views`, the default) generates the 8 novel views directly at azimuths `[30, 75, 120, 165, 210, 255, 300, 330]` degrees relative to the input view, elevation 0, giving the 9-view layout DIMO trains on. `sv4d2` gives 4 novel views at `[60, 120, 180, 240]`, so train with `num_views=5`. SV4D 1.0 instead renders a 21-view SV3D orbit of the reference frame once per object (cached in `multiview/canonical/`) and uses views `[0, 2, 5, 7, 9, 12, 14, 16, 19]` of it, i.e. azimuths `[0, 34.3, 85.7, 120, 154.3, 205.7, 240, 274.3, 325.7]`, which is the layout of the released Trump data. DIMO reads whichever layout `info.json` declares.
- **Frame counts.** Stage 3 subsamples the generated clip down to `clips.num_frames`, spreading the frames over the whole clip (first and last included): Wan's 121 frames become 41 by taking every 3rd. SV4D 2.0 is autoregressive, so longer sequences work too, but the count must suit both the model and the clip. `sv4d2_8views` renders 5 frames per window stepping 4, so it needs `4k+1` frames (a bad value would leave the tail unrendered, and `sample_multiview.py` refuses it); `sv4d2` renders 12 per window and accepts any count from 12 up. **SV4D 1.0 (`sv4d`) is fixed at exactly 21 frames**, because its anchor pass conditions on 5 motion frames spaced 4 apart, which only spans a 21-frame clip; run it as `sv4d.model=sv4d clips.num_frames=21`. The count should also divide the source clip evenly, otherwise the steps alternate (49 to 21 alternates 2 and 3 frames) and the timestamps DIMO assumes drift slightly from the real ones. The defaults match the backend: **41 frames for Wan** (every 3rd frame, so 8 fps over the whole 5 s clip, the same temporal resolution the paper's data had) and **25 for CogVideoX** (`clips.num_frames=25`, every 2nd frame, 4 fps over 6 s). Other evenly spaced options are 21, 25, 61, 121 for Wan and 13, 17, 49 for CogVideoX. Set `clips.frame_stride` instead to take a shorter dense window. Host memory for the cached dataset grows with the count: at 576 px and 9 views it is roughly 0.5 GB per motion at 41 frames, so about 25 GB for 50 motions.
- **Filtering.** The motion score is the paper's motion-amplitude metric: the mean of the largest 20% optical-flow magnitudes (RAFT) inside the object mask, averaged over consecutive frame pairs and expressed in pixels at the clip resolution. It is measured between consecutive *kept* frames, so it scales with `clips.num_frames`: the default 41-from-121 keeps every 3rd frame (8 fps), the rate the released data was sampled at, which is what makes the numbers below comparable; at 21 frames (4 fps) the same clip scores about twice as much. Over the 51 released Trump motions it spans 0.5 px (a barely perceptible shift) to 23.5 px (a full step), with a median of 5.2, so the default thresholds admit everything the authors kept and only drop frozen or runaway clips. They are object dependent, so `filter_videos.py` prints the score percentiles of your own clips to help tune `filter.min_motion` / `filter.max_motion`. The judge asks the VLM for 1-5 scores on visual quality, consistency and prompt alignment from a strip of six frames.
- **Switching models.** Each `sv4d.model` renders into its own `multiview/<model>/` folder, so you can generate both layouts for one object without them colliding; give the resulting datasets different names with `dataset.name=trump_sv4d2` when building them.
- **Naming and latent-code identity.** Motions are named `<index>-<short phrase>` (e.g. `003-raise_right_hand`) instead of generation timestamps, and `captions.json` keeps the full caption and short phrase per motion. Row *i* of the trained latent table belongs to the *i*-th motion `train.py` saw, which is the dataset's `input_videos` order only when training did not override `input_videos`. So `train.py` writes the order it actually used to `motion_order.json` beside the codes, and `train_text_projector.py` reads that file to pair each phrase with its code. Matching counts alone is not enough: a reordered list would regress every phrase onto the wrong code, converge to a low loss and report nothing wrong.
- **Caption diversity does not come from the sampling temperature.** GPT-5 accepts only `temperature=1`, so the prompt carries the work instead: it names six morphology-agnostic motion categories and asks for coverage across them, asks for an explicit spread of amplitudes (roughly a third large, a third medium, a third small), and each round is given the motions already produced plus the categories still thin. Without that, captioners collapse onto a run of near-identical head motions, and "simple, not exaggerated" phrasing collapses onto blinks and sways that the flow filter then rejects as static. Verified on GPT-5 with the released Trump reference: 24 requested motions came back as 24 distinct physical motions spread over all six categories, from a full-body crouch and a 90-degree pivot down to a wrist rotation and a single foot tap.
