"""Vision-language model clients used for captioning and quality judging.

Two backends share one interface (``chat(messages) -> str`` / ``chat_json(messages) -> Any``):

* ``openai``: any OpenAI-compatible chat-completions endpoint. This covers the OpenAI API (GPT-5 and
  the GPT-4o family) and a locally served Qwen3.5-9B (``vllm serve Qwen/Qwen3.5-9B
  --reasoning-parser qwen3``). Images are sent inline as base64 data URLs.
* ``local``: Qwen3.5 loaded in-process with ``transformers`` (needs a recent transformers and a GPU).

Messages use the OpenAI format: ``{"role": ..., "content": str | [{"type": "text"|"image", ...}]}``
where image parts are ``{"type": "image", "image": PIL.Image}``.

Why the request parameters are negotiated instead of configured
---------------------------------------------------------------
The three targets disagree about the sampling parameters, so one hard-coded set cannot serve all of
them. Probed against the live OpenAI API on 2026-09-06 (``python check_llm.py`` reruns this):

============================  ==================================  =====================
parameter                     GPT-5 family (incl. nano/mini)      GPT-4o family
============================  ==================================  =====================
``max_tokens``                400, "use max_completion_tokens"    accepted
``max_completion_tokens``     required                            accepted
``temperature``               only ``1``; 0.7 is a 400            accepted
``top_p``                     400, unsupported                    accepted
``presence_penalty``          400, unsupported                    accepted
``reasoning_effort``          accepted                            400, unrecognised
``verbosity``                 accepted                            400, unrecognised
images / data URLs            accepted                            accepted
============================  ==================================  =====================

So the client starts from a guess based on the model name and then *learns*: a 400 that names an
offending parameter is turned into a retry without it, and the drop is remembered for the rest of
the run. That keeps one code path working across GPT-5, GPT-4o, a vLLM-served Qwen3.5 and whatever
parameter rules a future model arrives with, without a per-model table to maintain.

Two consequences of GPT-5's reasoning worth knowing:

* ``max_completion_tokens`` covers *reasoning* tokens as well as the visible answer, so too small a
  budget returns an empty string with ``finish_reason="length"``. The client detects exactly that and
  retries with a larger budget rather than reporting an empty response.
* Temperature cannot be used to trade determinism for diversity. The caption stage therefore does not
  rely on it: diversity comes from the prompt's category coverage and the "already generated" list
  (see :mod:`datagen.prompts`), and structured extraction relies on the JSON instruction, both of
  which work at the fixed temperature of 1.
"""

import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

from PIL import Image

# Models that reason before answering, and so follow the right-hand column of the table above.
# Only used for the *first* request; `ChatClient._adapt` corrects a wrong guess from the API's reply.
REASONING_MODELS = re.compile(r"^(?:gpt-[5-9]|gpt-\d\d|o[1-9])", re.I)

# Parameters the client is allowed to give up on when the endpoint rejects them.
NEGOTIABLE = ("temperature", "top_p", "top_k", "presence_penalty", "reasoning_effort", "verbosity",
              "chat_template_kwargs")

# Qwen3.5's own recommendation for non-thinking mode, general tasks (model card, 2026-09-06):
# temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5. Applied only to Qwen, since `top_k` is
# not an OpenAI parameter and the OpenAI models reject the penalties.
QWEN_SAMPLING = {"top_k": 20, "presence_penalty": 1.5}


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


def extract_json(text: str) -> Any:
    """Parse the JSON payload of a model response.

    Used instead of the API's ``response_format={"type": "json_object"}`` because several prompts ask
    for a JSON *list*, which that mode cannot return, and because it has to work for the local
    backend too. It therefore has to survive what instruction-following models actually emit:

    * reasoning traces, including ones whose opening ``<think>`` was prefilled or truncated away, so
      everything up to the last ``</think>`` is discarded rather than only balanced pairs;
    * markdown code fences;
    * trailing commentary such as ``"That is 8 captions."`` after the payload;
    * a run of bare objects with no enclosing brackets, which is collected into a list.

    Only *contiguous* JSON is taken: decoding stops at the first byte that does not continue the
    payload, so trailing prose cannot be mistaken for another item. An earlier version scanned the
    whole response and returned every decodable value, which turned ``[{...}]\nThat is 1 caption.``
    into ``[[{...}], 1]`` and silently emptied a caption round.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON found in response: {text[:200]!r}")

    decoder = json.JSONDecoder()
    try:
        first, pos = decoder.raw_decode(text, start)
    except json.JSONDecodeError as e:
        raise ValueError(f"unparsable JSON in response: {text[start:start + 200]!r}") from e
    if not isinstance(first, dict):
        return first  # a list (or scalar) is the whole payload

    # A bare run of objects: keep taking them while only whitespace or a comma separates them.
    items = [first]
    while pos < len(text):
        gap = re.match(r"[\s,]*", text[pos:])
        pos += gap.end()
        if pos >= len(text) or text[pos] != "{":
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        items.append(obj)
    return items[0] if len(items) == 1 else items


class ChatClient:
    """OpenAI-compatible chat completions with optional image inputs.

    Sampling parameters are negotiated with the endpoint rather than assumed; see the module
    docstring. ``max_tokens`` is the *budget*, which for a reasoning model also pays for the hidden
    reasoning tokens, so it is raised automatically (up to ``budget_ceiling``) if reasoning uses it
    all up.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.7, top_p: float = 0.8, max_tokens: int = 8192,
                 reasoning_effort: Optional[str] = None, disable_thinking: bool = True,
                 qwen_sampling: Optional[bool] = None, retries: int = 3, retry_wait: float = 5.0,
                 budget_ceiling: int = 32768, verbose: bool = True):
        from openai import OpenAI

        api_key = os.environ.get(api_key_env) or ("EMPTY" if base_url else None)
        if api_key is None:
            raise EnvironmentError(
                f"no API key: set ${api_key_env} for the OpenAI API, or point `llm.base_url` at a "
                f"local OpenAI-compatible server (e.g. vllm serve {model}), or use `llm.backend=local`")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.budget = max_tokens
        self.budget_ceiling = max(max_tokens, budget_ceiling)
        self.retries = max(1, retries)
        self.retry_wait = retry_wait
        self.verbose = verbose

        # A local server means Qwen3.5 unless told otherwise: its recommended sampling knobs apply,
        # and its chat template takes the switch that turns thinking off.
        local = bool(base_url)
        self.qwen_sampling = local if qwen_sampling is None else qwen_sampling
        self.chat_template_kwargs = {"enable_thinking": False} if (disable_thinking and local) else None
        self.reasoning_effort = reasoning_effort or None

        # First guess at the parameter rules, corrected by `_adapt` on the first 400.
        self.reasoning = bool(REASONING_MODELS.match(os.path.basename(model)))
        self.token_param = "max_completion_tokens" if self.reasoning else "max_tokens"
        # `dropped` is what is left out of the request; `rejected` is the subset the endpoint itself
        # refused at runtime. The rest is the opening guess, so reporting only `rejected` keeps
        # `check_llm.py` from claiming a parameter was refused when it was never sent.
        self.dropped: Set[str] = set()
        self.rejected: Set[str] = set()
        if self.reasoning:
            self.dropped |= {"temperature", "top_p", "top_k", "presence_penalty"}
        else:
            self.dropped |= {"reasoning_effort", "verbosity"}

    # -- request construction ----------------------------------------------------------------------

    @staticmethod
    def _convert(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part["type"] == "image":
                        parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(part["image"])}})
                    else:
                        parts.append(part)
                content = parts
            out.append({"role": message["role"], "content": content})
        return out

    def _build_kwargs(self, messages: List[Dict[str, Any]], temperature: Optional[float]) -> Dict[str, Any]:
        """Assemble the request, leaving out everything the endpoint has rejected so far."""
        wanted: Dict[str, Any] = {"temperature": self.temperature if temperature is None else temperature,
                                  "top_p": self.top_p}
        if self.qwen_sampling:
            wanted.update(QWEN_SAMPLING)
        if self.reasoning_effort:
            wanted["reasoning_effort"] = self.reasoning_effort
        if self.chat_template_kwargs:
            wanted["chat_template_kwargs"] = self.chat_template_kwargs

        kwargs: Dict[str, Any] = {"model": self.model, "messages": self._convert(messages),
                                  self.token_param: self.budget}
        # `top_k` and the chat-template switch are not OpenAI parameters; vLLM reads them from
        # extra_body. Everything else is a first-class field.
        extra_body = {}
        for key, value in wanted.items():
            if key in self.dropped:
                continue
            if key in ("top_k", "chat_template_kwargs"):
                extra_body[key] = value
            else:
                kwargs[key] = value
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    # -- learning from errors ----------------------------------------------------------------------

    @staticmethod
    def _offending_param(error: Exception) -> Optional[str]:
        """The parameter an OpenAI-style 400 is complaining about, if it names one."""
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            param = (body.get("error") or {}).get("param")
            if param:
                return str(param).split(".")[0]
        message = str(error)
        for pattern in (r"(?:Unsupported|Unknown|Unrecognized|Invalid|Extra inputs)[^']*'([A-Za-z_]+)'",
                        r"argument supplied:\s*([A-Za-z_]+)",
                        r"'([A-Za-z_]+)' is not supported"):
            match = re.search(pattern, message, re.I)
            if match:
                return match.group(1)
        return None

    def _adapt(self, error: Exception) -> Optional[str]:
        """Drop or rename whatever the endpoint objected to. Returns a note, or None if unfixable."""
        if getattr(error, "status_code", None) not in (400, 422) and "400" not in str(error)[:32]:
            return None
        param = self._offending_param(error)
        if param is None:
            return None
        message = str(error)
        # The API tells us which of the two token parameters it wants.
        if param in ("max_tokens", "max_completion_tokens"):
            other = "max_completion_tokens" if param == "max_tokens" else "max_tokens"
            if other in message and self.token_param != other:
                self.token_param = other
                return f"{param} -> {other}"
            return None
        if param in NEGOTIABLE and param not in self.dropped:
            self.dropped.add(param)
            self.rejected.add(param)
            return f"dropped {param}"
        return None

    def _grow_budget(self) -> bool:
        """Reasoning ate the whole budget; give it more room. False when already at the ceiling."""
        if self.budget >= self.budget_ceiling:
            return False
        self.budget = min(self.budget * 4, self.budget_ceiling)
        return True

    # -- the call ----------------------------------------------------------------------------------

    def chat(self, messages: List[Dict[str, Any]], temperature: Optional[float] = None) -> str:
        last_error: Optional[Exception] = None
        attempt = 0
        # Parameter negotiation and budget growth are corrections, not failures, so they get their
        # own attempts on top of `self.retries` (which is for network errors and rate limits).
        max_attempts = self.retries + len(NEGOTIABLE) + 4
        while attempt < max_attempts:
            attempt += 1
            try:
                response = self.client.chat.completions.create(**self._build_kwargs(messages, temperature))
                choice = response.choices[0]
                text = choice.message.content
                if text and text.strip():
                    return text
                if choice.finish_reason == "length" and self._grow_budget():
                    if self.verbose:
                        print(f"[WARN] {self.model}: reasoning used the whole token budget; "
                              f"retrying with {self.token_param}={self.budget}")
                    continue
                last_error = RuntimeError(
                    f"empty response (finish_reason={choice.finish_reason}); "
                    f"raise llm.max_tokens (now {self.budget}) or lower llm.reasoning_effort")
                if choice.finish_reason == "length":
                    break
            except Exception as e:
                note = self._adapt(e)
                if note:
                    if self.verbose:
                        print(f"[INFO] {self.model} rejected a request parameter: {note}")
                    continue  # corrected, so this attempt does not count as a failure
                last_error = e
                if getattr(e, "status_code", None) in (401, 403, 404):
                    break  # bad key or unavailable model: retrying cannot help
                if attempt < max_attempts:
                    time.sleep(self.retry_wait * min(attempt, 4))
        raise RuntimeError(f"LLM request to {self.model!r} failed: {last_error}")

    def chat_json(self, messages: List[Dict[str, Any]], temperature: Optional[float] = None) -> Any:
        last_error: Optional[Exception] = None
        for _ in range(self.retries):
            text = self.chat(messages, temperature=temperature)
            try:
                return extract_json(text)
            except ValueError as e:
                last_error = e
        raise RuntimeError(f"could not get valid JSON from {self.model!r}: {last_error}")

    # -- setup check -------------------------------------------------------------------------------

    def preflight(self, image: bool = True) -> Dict[str, Any]:
        """One cheap round trip that settles the parameter negotiation and proves images work.

        Called by ``check_llm.py`` so a misconfigured endpoint fails in seconds instead of part-way
        through a 50-motion run.
        """
        content: Any = "Reply with the single word: ok"
        if image:
            content = [{"type": "text", "text": "Reply with the single word: ok"},
                       {"type": "image", "image": Image.new("RGB", (64, 64), (200, 30, 30))}]
        reply = self.chat([{"role": "user", "content": content}])
        return {"model": self.model, "reply": reply.strip()[:60], "images": image,
                "token_param": self.token_param, "budget": self.budget,
                "rejected": sorted(self.rejected),
                "reasoning_effort": self.reasoning_effort if "reasoning_effort" not in self.dropped else None}


class LocalQwenClient(ChatClient):
    """Qwen3.5 (or any image-text-to-text model) run in-process with transformers.

    Generation follows the model card's non-thinking recommendation: temperature 0.7, top_p 0.8,
    top_k 20. (``presence_penalty`` has no ``generate`` equivalent and is left out.)
    """

    def __init__(self, model: str = "Qwen/Qwen3.5-9B", temperature: float = 0.7, top_p: float = 0.8,
                 max_tokens: int = 8192, disable_thinking: bool = True, retries: int = 3,
                 device_map: str = "auto", dtype: str = "bfloat16", verbose: bool = True, **_):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model)
        self.model_obj = AutoModelForImageTextToText.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map=device_map).eval()
        self.model = model
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.budget = self.budget_ceiling = max_tokens
        self.top_k = QWEN_SAMPLING["top_k"]
        self.disable_thinking = disable_thinking
        self.retries, self.retry_wait, self.verbose = max(1, retries), 0.0, verbose
        self.token_param, self.reasoning_effort = "max_new_tokens", None
        self.dropped, self.rejected = set(), set()

    def chat(self, messages: List[Dict[str, Any]], temperature: Optional[float] = None) -> str:
        import torch

        converted, images = [], []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part["type"] == "image":
                        images.append(part["image"].convert("RGB"))
                        parts.append({"type": "image"})
                    else:
                        parts.append(part)
                content = parts
            converted.append({"role": message["role"], "content": content})
        text = self.processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=not self.disable_thinking)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.model_obj.device)
        temperature = self.temperature if temperature is None else temperature
        with torch.no_grad():
            output = self.model_obj.generate(**inputs, max_new_tokens=self.max_tokens,
                                             do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                             top_p=self.top_p, top_k=self.top_k)
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)

    def preflight(self, image: bool = True) -> Dict[str, Any]:
        reply = self.chat([{"role": "user", "content": (
            [{"type": "text", "text": "Reply with the single word: ok"},
             {"type": "image", "image": Image.new("RGB", (64, 64), (200, 30, 30))}] if image
            else "Reply with the single word: ok")}])
        return {"model": self.model, "reply": reply.strip()[:60], "images": image,
                "token_param": "max_new_tokens", "budget": self.max_tokens, "rejected": [],
                "reasoning_effort": None}


def build_client(cfg) -> ChatClient:
    """``cfg`` is the ``llm`` section of the config."""
    common = dict(model=cfg.model, temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens,
                  disable_thinking=cfg.disable_thinking, retries=cfg.retries)
    if cfg.backend == "openai":
        return ChatClient(base_url=cfg.base_url or None, api_key_env=cfg.api_key_env,
                          reasoning_effort=cfg.get("reasoning_effort") or None, **common)
    if cfg.backend == "local":
        return LocalQwenClient(**common)
    raise ValueError(f"unknown llm.backend {cfg.backend!r} (expected 'openai' or 'local')")
