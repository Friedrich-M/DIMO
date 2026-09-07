"""Stage 1: describe the reference image and generate diverse motion captions.

    python generate_captions.py object.name=trump object.image=inputs/trump.png [llm.backend=openai llm.model=gpt-4o]

Writes ``<workdir>/<object>/reference.png`` (object-centred RGBA canvas) and ``<workdir>/<object>/motions.json``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datagen.captions import build_motion_records, default_template, describe_object, generate_motion_captions, summarize_motions
from datagen.config import load_config, object_dir
from datagen.image import composite_on_white, prepare_reference
from datagen.io import ensure_dir, read_json, write_json
from datagen.llm import build_client


def main():
    cfg = load_config(description=__doc__)
    out_dir = ensure_dir(object_dir(cfg))
    motions_path = os.path.join(out_dir, "motions.json")
    if os.path.exists(motions_path) and not cfg.force:
        print(f"[INFO] {motions_path} exists, nothing to do (pass force=True to regenerate)")
        return

    if not cfg.object.image:
        raise SystemExit("`object.image` must point to the reference image")
    reference = prepare_reference(cfg.object.image, image_ratio=cfg.object.image_ratio,
                                  object_centered=cfg.object.object_centered, remove_bg=cfg.object.remove_bg)
    reference.save(os.path.join(out_dir, "reference.png"))
    reference_rgb = composite_on_white(reference)
    reference_rgb.save(os.path.join(out_dir, "reference_white.png"))

    client = build_client(cfg.llm)
    meta_path = os.path.join(out_dir, "description.json")
    if os.path.exists(meta_path) and not cfg.force:
        meta = read_json(meta_path)
    else:
        meta = describe_object(client, reference_rgb)
        write_json(meta_path, meta)
    print(f"[INFO] object: {meta['object']}")

    template = cfg.captions.template or default_template(meta)
    print(f"[INFO] seed caption: {template}")
    motions = generate_motion_captions(client, reference_rgb, template, cfg.captions.num_motions,
                                       per_request=cfg.captions.per_request, max_rounds=cfg.captions.max_rounds,
                                       temperature=cfg.captions.temperature)
    if cfg.captions.cluster_to:
        from datagen.text import bert_embeddings, kmeans_representatives

        features = bert_embeddings([m["motion_type"] for m in motions], cache_dir=cfg.projector.bert_cache_dir).numpy()
        keep = kmeans_representatives(features, cfg.captions.cluster_to, seed=cfg.seed)
        motions = [motions[i] for i in keep]
        print(f"[INFO] kept {len(motions)} motions after clustering in BERT space")

    summaries = summarize_motions(client, [m["caption"] for m in motions])
    records = build_motion_records(motions, summaries)
    write_json(motions_path, {"object": cfg.object.name, "reference_image": os.path.abspath(cfg.object.image),
                              "template": template, "motions": records})
    print(f"[INFO] {len(records)} motion captions written to {motions_path}")
    for r in records[:5]:
        print(f"    {r['name']}: {r['caption'][:100]}...")


if __name__ == "__main__":
    main()
