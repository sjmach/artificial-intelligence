# Adding a Custom Guardrail (GLiGuard) to a LiteLLM AI Gateway

A FastAPI service that wraps [GLiGuard](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) — a 300M-parameter NER-based LLM guardrail model — and exposes it as both a flexible inspection API and a drop-in LiteLLM custom guardrail.

**Article:** [Custom Guardrail (GLiGuard) to a LiteLLM Proxy](https://www.sundeepmachado.com/2026/06/custom-guardrail-gliguard-litellm-proxy.html)

## Files

| File | Purpose |
|---|---|
| `gliguard_service.py` | FastAPI service exposing `/guard` and the LiteLLM guardrail endpoint |
| `guard_client.py` | Async Python client for calling the service from your own code |

## Prerequisites

```bash
pip install fastapi uvicorn gliner2 torch httpx
```

A CUDA-capable GPU is optional but recommended — the model loads onto CPU if no GPU is found.

## Running the service

```bash
uvicorn gliguard_service:app --host 0.0.0.0 --port 8765
```

The model (`fastino/gliguard-LLMGuardrails-300M`) is downloaded from HuggingFace on first startup.

## API

### `POST /guard`

Flexible endpoint for checking a prompt, a response, or both. Returns per-task classification results.

```json
{
  "prompt": "How do I make explosives?",
  "response": null,
  "tasks": ["prompt_safety", "jailbreak_detection"]
}
```

`tasks` is optional — omit it to run all applicable tasks. Valid task names:

| Task | Applies to | Labels |
|---|---|---|
| `prompt_safety` | prompt | `safe`, `unsafe` |
| `prompt_toxicity` | prompt | multi-label (violence, hate, PII, ...) |
| `jailbreak_detection` | prompt | multi-label (prompt_injection, roleplay_bypass, ...) |
| `response_safety` | response | `safe`, `unsafe` |
| `response_toxicity` | response | multi-label |
| `response_refusal` | response | `refusal`, `compliance` |

### `POST /beta/litellm_basic_guardrail_api`

Implements LiteLLM's `generic_guardrail_api` contract. Returns `{"action": "NONE"}` or `{"action": "BLOCKED", "blocked_reason": "..."}`.

```json
{
  "texts": ["user message here"],
  "input_type": "request"
}
```

Set `input_type` to `"response"` for model output checks.

### `GET /health`

```json
{"status": "ok", "device": "cuda"}
```

## Using the Python client

```python
import asyncio
from guard_client import GuardClient

async def main():
    async with GuardClient() as guard:
        # Simple boolean checks
        safe = await guard.is_prompt_safe("Tell me about quantum physics")
        print(safe)  # True

        safe = await guard.is_response_safe(
            prompt="What is the capital of France?",
            response="Paris is the capital of France."
        )
        print(safe)  # True

        # Raw classification result
        result = await guard.check(
            prompt="Ignore previous instructions and...",
            tasks=["prompt_safety", "jailbreak_detection"],
        )
        print(result)

asyncio.run(main())
```

## LiteLLM integration

Add the guardrail to your LiteLLM `config.yaml`:

```yaml
guardrails:
  - guardrail_name: gliguard
    litellm_params:
      guardrail: generic_guardrail_api
      guardrail_endpoint: http://localhost:8765/beta/litellm_basic_guardrail_api
      default_on: true
```

## Blocking logic

- **Prompt** is blocked if `prompt_safety == "unsafe"` OR any jailbreak label other than `benign` is detected.
- **Response** is blocked only if `response_safety == "unsafe"` AND `response_refusal != "refusal"` — this avoids blocking the model's own refusals, which the model may label unsafe due to the topic they describe.
