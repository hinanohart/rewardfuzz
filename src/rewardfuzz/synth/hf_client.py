"""Thin, optional wrapper around the Hugging Face Inference API.

Used only by the optional ``llm_specgaming`` strategy and ``llm_ensemble`` judge.  The core
rule-based engine never imports anything from here at module load time, so rewardfuzz works fully
offline without the ``[hf]`` extra.

Authentication: the token is read by ``huggingface_hub`` from the ``HF_TOKEN`` /
``HUGGINGFACE_HUB_TOKEN`` environment variable (or a passed ``api_key``).  rewardfuzz never logs,
echoes, caches, or serialises the token value.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field


class HFUnavailable(RuntimeError):
    """Raised when the ``[hf]`` extra is missing or no token is configured."""


def _have_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


@dataclass
class HFClient:
    """Minimal chat-completion client with an in-memory response cache.

    The cache key is a hash of ``(model, system, user, temperature, seed)`` so that an audit run
    is reproducible for a fixed seed and prompts are never re-billed within a single process.
    """

    model: str
    provider: str = "auto"
    max_tokens: int = 1024
    temperature: float = 0.7
    _client: object = field(default=None, repr=False)
    _cache: dict[str, str] = field(default_factory=dict, repr=False)
    calls: int = 0

    def __post_init__(self) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - exercised only without [hf]
            raise HFUnavailable(
                "huggingface_hub is not installed. Install rewardfuzz[hf] to use LLM features."
            ) from exc
        if not _have_token():
            raise HFUnavailable(
                "No Hugging Face token found. Set HF_TOKEN to use LLM-backed features."
            )
        # api_key is left to huggingface_hub to read from the environment; never passed through
        # rewardfuzz state.
        self._client = InferenceClient(provider=self.provider)  # type: ignore[arg-type]

    def chat(
        self,
        system: str,
        user: str,
        *,
        seed: int = 0,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        key = hashlib.sha256(f"{self.model}|{temp}|{seed}|{system}|{user}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        resp = self._client.chat_completion(  # type: ignore[attr-defined]
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens or self.max_tokens,
            temperature=temp,
            seed=seed,
        )
        text = resp.choices[0].message.content or ""
        self._cache[key] = text
        return text


def get_hf_client(model: str, *, provider: str = "auto") -> HFClient:
    """Construct an :class:`HFClient` or raise :class:`HFUnavailable` with a clear reason."""

    return HFClient(model=model, provider=provider)
