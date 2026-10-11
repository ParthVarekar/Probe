"""Optional AI provider port (milestone M10; handover §12 Part F boundary (2), Part M).

The product works with no key and no network: the default provider is
:class:`NullProvider`. AI output is only ever a *proposal*: it never enters a
likelihood, a decision or a number. Every call is recorded as an ``LLMCall``
provenance dict (request hash, provider, model, template, output, validation
errors), and **never** contains a key.

Providers:

* :class:`NullProvider`: assist disabled (returns nothing).
* :class:`ReplayProvider`: answers from a recorded cassette keyed by request hash,
  so tests and replays never call a model.
* :class:`RecordingProvider`: wraps another provider and appends to a cassette.
* :class:`AnthropicProvider`: Claude via the official ``anthropic`` SDK (optional
  extra, imported lazily; the key is read by the SDK from the environment).

Tasks are validated deterministically. :func:`paraphrase_claim` accepts a
paraphrase only if it passes the H.9 lexicon check for the claim's level *and*
keeps every number of the template text; otherwise the template text stands.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from probe.claims import check_wording
from probe.determinism import content_hash

__all__ = [
    "AnthropicProvider",
    "NullProvider",
    "Provider",
    "RecordingProvider",
    "ReplayProvider",
    "paraphrase_claim",
]

DEFAULT_MODEL = "claude-opus-5-5"
TEMPLATES = {
    "paraphrase_claim@v1": (
        "You rewrite one scientific claim for a non-specialist reader. Keep every number "
        "exactly as written, keep the scope phrase (e.g. 'in simulator world ...') and do "
        "not strengthen the claim: no causal words (causes, reduces, leads to, effect of, "
        "because, impact). Reply with the rewritten claim only."
    ),
}


@dataclass(frozen=True)
class Completion:
    text: str | None
    provider: str
    model: str | None
    usage: Mapping[str, Any]
    stop_reason: str | None = None


class Provider(Protocol):
    name: str

    def complete(self, template_id: str, user: str) -> Completion: ...


def request_hash(template_id: str, user: str, model: str | None) -> str:
    return content_hash(
        {"template": template_id, "system": TEMPLATES[template_id], "user": user, "model": model}
    )


class NullProvider:
    name = "null"

    def complete(self, template_id: str, user: str) -> Completion:
        return Completion(None, self.name, None, {})


class ReplayProvider:
    """Answers from ``{request_hash: completion}``; unknown requests return nothing."""

    name = "replay"

    def __init__(self, cassette: Path, model: str | None = DEFAULT_MODEL) -> None:
        self.model = model
        self._data: dict[str, Any] = (
            json.loads(Path(cassette).read_text(encoding="utf-8"))
            if Path(cassette).exists()
            else {}
        )

    def complete(self, template_id: str, user: str) -> Completion:
        hit = self._data.get(request_hash(template_id, user, self.model))
        if hit is None:
            return Completion(None, self.name, self.model, {}, "not_recorded")
        return Completion(
            hit["text"], self.name, self.model, hit.get("usage", {}), hit.get("stop_reason")
        )


class RecordingProvider:
    name = "recording"

    def __init__(self, inner: Any, cassette: Path) -> None:
        self.inner = inner
        self.cassette = Path(cassette)

    def complete(self, template_id: str, user: str) -> Completion:
        out: Completion = self.inner.complete(template_id, user)
        data = (
            json.loads(self.cassette.read_text(encoding="utf-8")) if self.cassette.exists() else {}
        )
        data[request_hash(template_id, user, out.model)] = {
            "text": out.text,
            "usage": dict(out.usage),
            "stop_reason": out.stop_reason,
        }
        self.cassette.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        return out


class AnthropicProvider:
    """Claude through the official SDK (``pip install probe[ai]``). Credentials come
    from the SDK's own resolution (``ANTHROPIC_API_KEY`` or an ``ant auth`` profile);
    PROBE never reads, stores or logs them."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low") -> None:
        try:
            import anthropic
        except ImportError as e:  # optional extra
            raise RuntimeError("AI assist needs the optional extra: pip install 'probe[ai]'") from e
        self._client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def complete(self, template_id: str, user: str) -> Completion:
        resp = self._client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=TEMPLATES[template_id],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            return Completion(None, self.name, resp.model, {}, "refusal")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return Completion(text, self.name, resp.model, usage, resp.stop_reason)


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def paraphrase_claim(provider: Any, claim_text: str, level: str) -> dict[str, Any]:
    """A validated paraphrase proposal plus its ``LLMCall`` provenance record."""
    tid = "paraphrase_claim@v1"
    out: Completion = provider.complete(tid, claim_text)
    errors: list[str] = []
    text = (out.text or "").strip()
    if not text:
        errors.append("no output")
    else:
        errors += check_wording(text, level)
        missing = sorted(set(_NUM.findall(claim_text)) - set(_NUM.findall(text)))
        if missing:
            errors.append(f"numbers changed or dropped: {missing}")
    accepted = not errors
    return {
        "text": text if accepted else claim_text,
        "accepted": accepted,
        "epistemic_status": "proposed",
        "llm_call": {
            "request_hash": request_hash(tid, claim_text, out.model),
            "provider": out.provider,
            "model": out.model,
            "template": tid,
            "output": out.text,
            "usage": dict(out.usage),
            "stop_reason": out.stop_reason,
            "validation_errors": errors,
        },
    }
