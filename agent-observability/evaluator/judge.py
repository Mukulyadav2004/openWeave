"""Judge callers and output validation.

Two providers, one interface. `ollama` keeps the local setup from docker-compose
working; `openai` covers anything speaking the OpenAI chat-completions shape
(OpenAI, Together, Groq, vLLM, Ollama's own compat endpoint), which is most
things. Adding a third means adding a function here and nothing else.

The important part is not the HTTP call, it is `validate_output`: an LLM judge
that returns free text is not a judge, it is a chatbot. Every verdict is checked
against the evaluator version's declared output schema before it becomes a
Score, and a verdict that fails validation is an ERROR job, never a silent 0.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

logger = logging.getLogger("openweave.judge")

REQUEST_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "60"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class JudgeError(Exception):
    """The judge could not produce a usable verdict."""


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #
_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def render_prompt(template: str, variables: dict) -> str:
    """Substitute {{name}} placeholders.

    Double braces rather than str.format so a prompt can contain literal JSON
    (`{"score": 0.8}`) without every brace needing escaping — judge prompts are
    full of example JSON, so this comes up immediately.

    A missing variable renders as empty rather than raising: a trace with no
    output is a real case, and it should reach the judge as an empty field, not
    crash the worker.
    """
    def replace(match):
        value = variables.get(match.group(1), "")
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return "" if value is None else str(value)

    return _VAR.sub(replace, template)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
async def _call_ollama(model: str, prompt: str, params: dict) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "format": "json", "options": params or {}},
        )
        response.raise_for_status()
        return response.json().get("response", "")


async def _call_openai(model: str, prompt: str, params: dict) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                **(params or {}),
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


PROVIDERS = {"ollama": _call_ollama, "openai": _call_openai}


async def call_judge(provider: str, model: str, prompt: str,
                     params: dict | None = None) -> dict:
    """Run the judge and return its parsed verdict."""
    caller = PROVIDERS.get((provider or "ollama").lower())
    if caller is None:
        raise JudgeError(f"unknown provider {provider!r}; have {sorted(PROVIDERS)}")

    try:
        raw = await caller(model, prompt, params or {})
    except httpx.HTTPError as exc:
        raise JudgeError(f"{provider} request failed: {exc}") from exc

    if not raw or not raw.strip():
        raise JudgeError("judge returned an empty response")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Models wrap JSON in prose or fences more often than they should.
        # One salvage attempt, then give up honestly rather than guessing.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise JudgeError(f"judge did not return JSON: {raw[:200]!r}") from None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_output(verdict: dict, schema: dict | None) -> dict:
    """Check a verdict against the evaluator version's output schema.

    Uses `jsonschema` when it is installed and falls back to a required-keys and
    type check otherwise, so the eval worker has no hard dependency on it. The
    fallback covers the shape every judge schema actually uses.
    """
    if not schema:
        return verdict

    try:
        import jsonschema
    except ImportError:
        jsonschema = None

    if jsonschema is not None:
        try:
            jsonschema.validate(verdict, schema)
            return verdict
        except jsonschema.ValidationError as exc:
            raise JudgeError(f"verdict failed schema: {exc.message}") from None

    for key in schema.get("required", []):
        if key not in verdict:
            raise JudgeError(f"verdict missing required field {key!r}")

    types = {"number": (int, float), "integer": int, "string": str,
             "boolean": bool, "object": dict, "array": list}
    for key, spec in (schema.get("properties") or {}).items():
        if key not in verdict or "type" not in spec:
            continue
        expected = types.get(spec["type"])
        # bool is a subclass of int in Python; a boolean is not a number here.
        if expected and (not isinstance(verdict[key], expected)
                         or (spec["type"] in ("number", "integer")
                             and isinstance(verdict[key], bool))):
            raise JudgeError(
                f"field {key!r} should be {spec['type']}, got "
                f"{type(verdict[key]).__name__}"
            )
        if spec["type"] == "number":
            low, high = spec.get("minimum"), spec.get("maximum")
            if low is not None and verdict[key] < low:
                raise JudgeError(f"{key}={verdict[key]} below minimum {low}")
            if high is not None and verdict[key] > high:
                raise JudgeError(f"{key}={verdict[key]} above maximum {high}")
    return verdict
