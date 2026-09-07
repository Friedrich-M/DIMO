"""Check the vision-language model setup before launching a long generation job.

    python check_llm.py                                   # whatever configs/default.yaml selects
    python check_llm.py llm.model=gpt-5-mini
    python check_llm.py llm.model=Qwen/Qwen3.5-9B llm.base_url=http://localhost:8000/v1
    python check_llm.py llm.backend=local

Sends two short requests: one plain, one with an image, and one structured-JSON request of the shape
the caption stage uses. Reports which sampling parameters the endpoint accepted, since the client
negotiates them at runtime (GPT-5 rejects ``max_tokens``, ``top_p`` and any temperature but 1, while
GPT-4o rejects ``reasoning_effort``). Costs a few hundred tokens, versus a whole run that fails on
its last motion.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datagen.config import load_config
from datagen.llm import build_client

JSON_PROBE = ("Answer with a JSON list of exactly two objects and no other text, each with the keys "
              '"category" and "motion_type", describing two different motions a cat could make.')


def main():
    cfg = load_config(description=__doc__, require_object=False)
    llm = cfg.llm
    where = llm.base_url or ("in-process" if llm.backend == "local" else "https://api.openai.com/v1")
    print(f"[INFO] backend={llm.backend} model={llm.model} endpoint={where}")
    if llm.backend == "openai" and not llm.base_url and not os.environ.get(llm.api_key_env):
        raise SystemExit(f"${llm.api_key_env} is not set. Export it, or point llm.base_url at a local server, "
                         f"or use llm.backend=local.")

    client = build_client(llm)

    info = client.preflight(image=False)
    print(f"[ OK ] text request -> {info['reply']!r}")
    info = client.preflight(image=True)
    print(f"[ OK ] image request -> {info['reply']!r} (a red square; the answer should say so)")

    result = client.chat_json([{"role": "user", "content": JSON_PROBE}])
    if not isinstance(result, list) or len(result) != 2 or not all(isinstance(r, dict) for r in result):
        raise SystemExit(f"the model did not return the requested JSON list of 2 objects: {result!r}")
    print(f"[ OK ] structured JSON -> {result}")

    # Read off the client after the last request, not from the preflight result: the budget can grow
    # mid-run, and the number worth reporting is the one that ended up working.
    effort = client.reasoning_effort if "reasoning_effort" not in client.dropped else None
    rejected = sorted(client.rejected)
    print(f"[INFO] negotiated: token budget sent as {client.token_param}={client.budget}"
          + (f", reasoning_effort={effort}" if effort else "")
          + (f", parameters this endpoint refused: {', '.join(rejected)}" if rejected else
             ", no parameter had to be dropped"))
    if client.budget > int(llm.max_tokens):
        print(f"[INFO] llm.max_tokens={llm.max_tokens} was too small for this model's reasoning; set it to "
              f"at least {client.budget} (or lower llm.reasoning_effort) to avoid the retries")
    print("[INFO] setup looks good; `python generate_captions.py` will work with this config")


if __name__ == "__main__":
    main()
