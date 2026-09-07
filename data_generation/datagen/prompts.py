"""Prompt templates for the caption stage and the video-quality judge (paper Sec. 3.1).

The diverse-motion prompt follows the auto-prompting recipe used for the paper: the model sees the
reference image and a "seed" caption, and returns captions that keep the appearance description
identical and vary only the motion after "As time progresses". Two things the prompt has to fight:

* **Collapse onto one body part.** Asked plainly for "diverse motions", captioners return a run of
  near-identical head or torso motions. The prompt therefore names motion categories and asks for
  coverage across them, and later rounds are told which categories are still thin.
* **Collapse onto tiny motions.** "Simple, not exaggerated" phrasing pushes every caption towards
  blinks and sways, which the flow filter then rejects as static. The prompt asks for an explicit
  spread of amplitudes instead, since the trained latent space needs both small and large motions.

The constraints that remain hard are the ones the 3D stages depend on: one continuous motion that
completes inside the clip, the object staying in place and fully in frame, a pure white background,
no camera motion, and nothing entering the scene.
"""

# Coverage targets for the caption stage. Deliberately morphology-agnostic: the captioner maps them
# onto whatever the object actually has (arms, wings, branches, wheels, ...) and skips ones that
# cannot apply.
MOTION_CATEGORIES = [
    "locomotion: the whole object travels or turns in place (step, walk, hop, pivot, roll)",
    "appendage: one limb, arm, wing, branch or similar part moves on its own",
    "head or top part: the upper part turns, tilts, nods or looks around",
    "body axis: the body leans, twists, bends, stretches or sways as a whole",
    "posture change: the object rises, crouches, settles, spreads out or folds in",
    "fine detail: a small localised motion such as a blink, a breath, a finger or a flick",
]

META_PROMPT = """You are given the first frame of a video: a single object centred on a white background.
Describe it as a structured JSON object with these fields:
- "object": the object category in a few words (e.g. "cartoon man in a suit", "grey tabby cat").
- "appearance": one or two complete sentences describing the appearance in detail (shape, colours, clothing, materials).
- "expression": one sentence describing the facial expression or demeanour, or an empty string if not applicable.
- "initial_state": one sentence describing the current pose / action state in this first frame.
- "movable_parts": a list of the parts of THIS object that could plausibly move, named as they appear in the image (e.g. ["head", "left arm", "right arm", "legs", "coat tails"]). Only list parts that are actually visible.
- "possible_motions": a list of 10 short third-person clauses (4-8 words) describing plausible motions this object could perform next, each starting with a subject and a present-tense verb, e.g. "he raises his right hand slowly" or "it turns its head to the left". Spread them over different parts and different sizes of motion, from a full step to a small gesture.
Answer with the JSON object only."""

DIVERSE_MOTIONS_SYSTEM = """You write video captions that will be used to generate short videos of one object, and then to reconstruct that object's motion in 3D. Give exactly {count} captions.

**Input**: the first frame of the video (a single object centred on a pure white background) and one example caption from the user. The example has two parts: a description of the object and its starting state, then a motion introduced by "As time progresses".

**What to keep identical**: copy the user's text before "As time progresses" word for word, unchanged. Every caption must start from the object exactly as it appears in this frame.

**What to vary**: only the motion after "As time progresses".

**Diversity is the point.** The {count} motions must differ in *what physically moves*, not in wording. Spread them across these categories, using the ones that suit this object and skipping the ones that do not:
{categories}
Also vary the size of the motion: aim for roughly a third large and clearly visible (a step, a full turn, a raised arm), a third medium, and a third small and subtle. Two captions describing the same part moving the same way in the same direction count as duplicates even if the sentences differ.

**Hard requirements**, because the video is later lifted to 3D:
1. One single continuous motion per caption, begun and finished within about five seconds.
2. The object stays in place and stays completely inside the frame. It may step, turn or lean, but it must not walk out of view or move so far that it leaves the centre of the frame.
3. The background stays pure white and empty. Nothing else enters the scene, and the object does not pick up, hold or interact with any object that is not already visible.
4. No camera motion, no zoom, no perspective change, no cuts, no scene changes.
5. The motion must be physically plausible for this object's shape, and must be visible from this single viewpoint. Avoid motions that happen only on the side facing away from the camera.
6. Describe only motion. Do not add lighting changes, colour changes, weather, mood shifts or story.

**Style**: complete sentences, concrete and specific about which part moves and in which direction. Do not name the categories in the caption text.{avoid}

**Output**: a JSON list of exactly {count} objects, no other text:

{{
    "category": "one of the category names above (the word before the colon)",
    "motion_type": "the motion as a short imperative phrase, 3-6 words, e.g. 'raise the right hand'",
    "video_caption": "the full caption: the user's unchanged opening text, then 'As time progresses, ...'"
}}

user input:"""

AVOID_TEMPLATE = """

**Already generated, do not repeat these or produce near-variants** (a different phrasing of the same physical motion is a repeat):
{motions}"""

CATEGORY_HINT_TEMPLATE = """
So far the categories {covered} are covered. Prefer motions from {missing}, if they suit this object."""

SHORT_SUMMARY_SYSTEM = """You will receive a list of detailed video captions. For each caption, summarise the motion that happens after "As time progresses" into a simple imperative phrase of 2 to 5 words, such as "lift the right hand", "walk forward" or "shake head". Name the part that moves and the direction where it matters. Keep the input order. Answer with a JSON list of strings only, one per caption."""

QUALITY_JUDGE_SYSTEM = """You are a strict video quality assessor. You will see the reference image of an object and a sequence of frames sampled from a generated video that should start from that image. Rate the video with integer scores from 1 (very bad) to 5 (excellent) for:
- "visual_quality": sharpness and absence of artifacts, deformations or extra limbs/objects.
- "consistency": the object keeps its identity, shape and colours across frames, stays centred on a white background, and no camera motion or cuts occur.
- "motion_alignment": the motion described by the prompt is clearly visible and plausible.
Also give a one-sentence "comment". Answer with a JSON object only."""


def category_names() -> list:
    """Short names of :data:`MOTION_CATEGORIES` (the part before the colon)."""
    return [c.split(":", 1)[0] for c in MOTION_CATEGORIES]


def seed_caption(meta: dict, motion: str) -> str:
    """Compose the template caption from the structured description and one motion phrase."""
    parts = [meta.get("appearance", "").strip(), meta.get("expression", "").strip(), meta.get("initial_state", "").strip()]
    prefix = " ".join(p if p.endswith(".") else p + "." for p in parts if p)
    return f"{prefix} The background is pure white. As time progresses, {motion.strip().rstrip('.')}. The background remains pure white."
