"""Stage 1: structured object description and diverse motion captions from one reference image."""

import re
from typing import Dict, List

from PIL import Image

from datagen.io import motion_name
from datagen.llm import ChatClient
from datagen.prompts import (
    AVOID_TEMPLATE,
    CATEGORY_HINT_TEMPLATE,
    DIVERSE_MOTIONS_SYSTEM,
    META_PROMPT,
    MOTION_CATEGORIES,
    SHORT_SUMMARY_SYSTEM,
    category_names,
    seed_caption,
)


def describe_object(client: ChatClient, image: Image.Image) -> Dict:
    """The "meta" prompt: appearance, expression and initial action state of the object."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": META_PROMPT}]}]
    meta = client.chat_json(messages, temperature=0.2)
    if isinstance(meta, list):
        meta = meta[0]
    for key in ("object", "appearance", "initial_state"):
        if not meta.get(key):
            raise ValueError(f"object description is missing {key!r}: {meta}")
    return meta


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower()).strip()


def generate_motion_captions(client: ChatClient, image: Image.Image, template: str, num_motions: int,
                             per_request: int = 20, max_rounds: int = 10, temperature: float = 1.0) -> List[Dict]:
    """Ask for ``per_request`` captions at a time until ``num_motions`` distinct motions are collected.

    Each round tells the model which motions it already produced and which categories are still
    thin, which is what stops it from returning twenty variations of the same head turn.
    """
    motions: List[Dict] = []
    seen = set()
    categories = category_names()
    for _ in range(max_rounds):
        if len(motions) >= num_motions:
            break
        count = min(per_request, num_motions - len(motions))
        avoid = ""
        if motions:
            avoid = AVOID_TEMPLATE.format(motions="\n".join(f"- {m['motion_type']}" for m in motions))
            covered = {m.get("category") for m in motions if m.get("category")}
            missing = [c for c in categories if c not in covered]
            if missing and covered:
                avoid += CATEGORY_HINT_TEMPLATE.format(covered=", ".join(sorted(covered)), missing=", ".join(missing))
        system = DIVERSE_MOTIONS_SYSTEM.format(count=count, avoid=avoid,
                                               categories="\n".join(f"- {c}" for c in MOTION_CATEGORIES))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": template}, {"type": "image", "image": image}]},
        ]
        try:
            result = client.chat_json(messages, temperature=temperature)
        except RuntimeError as e:  # unparsable answer: try again with a fresh request
            print(f"[WARN] caption request failed ({e}); retrying")
            continue
        items = result if isinstance(result, list) else [result]
        for item in items:
            if not isinstance(item, dict) or not item.get("video_caption") or not item.get("motion_type"):
                continue
            key = _normalise(item["motion_type"])
            if key in seen:
                continue
            seen.add(key)
            category = str(item.get("category", "")).split(":", 1)[0].strip().lower()
            motions.append({"motion_type": item["motion_type"].strip(), "caption": item["video_caption"].strip(),
                            "category": category if category in categories else ""})
    if len(motions) < num_motions:
        print(f"[WARN] collected {len(motions)} distinct motions, fewer than the {num_motions} requested")
    motions = motions[:num_motions]
    spread = {c: sum(1 for m in motions if m["category"] == c) for c in categories}
    print("[INFO] motion categories: " + ", ".join(f"{c.split()[0]} {n}" for c, n in spread.items() if n))
    return motions


def summarize_motions(client: ChatClient, captions: List[str], chunk: int = 25) -> List[str]:
    """Short imperative phrases (e.g. "lift the right hand") used for language-guided generation."""
    summaries: List[str] = []
    for start in range(0, len(captions), chunk):
        block = captions[start:start + chunk]
        text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(block))
        result = client.chat_json([{"role": "system", "content": SHORT_SUMMARY_SYSTEM}, {"role": "user", "content": text}], temperature=0.2)
        result = result if isinstance(result, list) else [result]
        if len(result) != len(block):
            raise RuntimeError(f"expected {len(block)} summaries, got {len(result)}")
        summaries += [str(s).strip().rstrip(".") for s in result]
    return summaries


def build_motion_records(motions: List[Dict], summaries: List[str]) -> List[Dict]:
    records = []
    for i, (motion, short) in enumerate(zip(motions, summaries)):
        records.append({
            "name": motion_name(i, short),
            "motion_type": motion["motion_type"],
            "category": motion.get("category", ""),
            "short": short,
            "caption": motion["caption"],
        })
    return records


def default_template(meta: Dict) -> str:
    """Seed caption built from the description and the first suggested motion."""
    candidates = meta.get("possible_motions") or ["starts to move gently"]
    return seed_caption(meta, candidates[0])
