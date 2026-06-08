from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from gliner2 import GLiNER2
from pydantic import BaseModel

MODEL_ID = "fastino/gliguard-LLMGuardrails-300M"

SAFETY_LABELS = ["safe", "unsafe"]
REFUSAL_LABELS = ["refusal", "compliance"]

TOXICITY_LABELS = [
    "violence_and_weapons", "non_violent_crime", "sexual_content",
    "hate_and_discrimination", "self_harm_and_suicide", "pii_exposure",
    "misinformation", "copyright_violation", "child_safety",
    "political_manipulation", "unethical_conduct", "regulated_advice",
    "privacy_violation", "other", "benign",
]

JAILBREAK_LABELS = [
    "prompt_injection", "jailbreak_attempt", "policy_evasion",
    "instruction_override", "system_prompt_exfiltration", "data_exfiltration",
    "roleplay_bypass", "hypothetical_bypass", "obfuscated_attack",
    "multi_step_attack", "social_engineering", "benign",
]

# Lower threshold than the 0.5 default — these are multi-label tasks where
# missing a borderline category is worse than a false positive.
TOXICITY_TASK = {"labels": TOXICITY_LABELS, "multi_label": True, "cls_threshold": 0.4}
JAILBREAK_TASK = {"labels": JAILBREAK_LABELS, "multi_label": True, "cls_threshold": 0.4}

# Canonical task -> classify_text config, mirroring the model card so /guard
# can expose every task GLiGuard supports without inventing label sets.
PROMPT_TASKS = {
    "prompt_safety": SAFETY_LABELS,
    "prompt_toxicity": TOXICITY_TASK,
    "jailbreak_detection": JAILBREAK_TASK,
}
RESPONSE_TASKS = {
    "response_safety": SAFETY_LABELS,
    "response_toxicity": TOXICITY_TASK,
    "response_refusal": REFUSAL_LABELS,
}
ALL_TASKS = {**PROMPT_TASKS, **RESPONSE_TASKS}

model: GLiNER2 = None
device: str = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not per-request — the 300M encoder
    # fits comfortably on a 6GB card but reloading it would tank latency.
    global model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER2.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    yield


app = FastAPI(title="GLiGuard", lifespan=lifespan)


class GuardRequest(BaseModel):
    prompt: Optional[str] = None
    response: Optional[str] = None
    tasks: Optional[list[str]] = None


class LiteLLMGuardrailRequest(BaseModel):
    texts: list[str] = []
    input_type: str = "request"


@app.post("/guard")
async def guard(req: GuardRequest):
    if req.prompt is None and req.response is None:
        raise HTTPException(status_code=422, detail="Provide at least one of: prompt, response")

    requested = set(req.tasks) if req.tasks else set(ALL_TASKS)
    invalid = requested - set(ALL_TASKS)
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unknown tasks: {sorted(invalid)}")

    result = {}

    # Only run the prompt-side tasks the caller actually asked for, and only
    # if a prompt was supplied — keeps inference cost proportional to the request.
    prompt_tasks = {name: cfg for name, cfg in PROMPT_TASKS.items() if name in requested}
    if req.prompt is not None and prompt_tasks:
        result.update(model.classify_text(req.prompt, prompt_tasks))

    # Response-side tasks need the "Response: " prefix the model was trained on;
    # including the prompt as context lets it judge e.g. refusals correctly.
    response_tasks = {name: cfg for name, cfg in RESPONSE_TASKS.items() if name in requested}
    if req.response is not None and response_tasks:
        text = f"Prompt: {req.prompt}\nResponse: {req.response}" if req.prompt else f"Response: {req.response}"
        result.update(model.classify_text(text, response_tasks))

    return result


@app.post("/beta/litellm_basic_guardrail_api")
async def litellm_guardrail(req: LiteLLMGuardrailRequest):
    """Implements LiteLLM's generic_guardrail_api contract: {action: NONE | BLOCKED}."""
    text = "\n".join(req.texts).strip()
    if not text:
        return {"action": "NONE"}

    # Classify only the tasks that feed the block decision below — toxicity
    # is informational (logged elsewhere), not a blocking signal here, so we
    # skip it to halve the inference work per request.
    if req.input_type == "response":
        tasks = {"response_safety": SAFETY_LABELS, "response_refusal": REFUSAL_LABELS}
        result = model.classify_text(f"Response: {text}", tasks)
        # A response is unsafe only if it's flagged unsafe AND isn't a refusal —
        # the model often labels refusals "unsafe" because of the topic they describe.
        unsafe = result.get("response_safety") == "unsafe" and result.get("response_refusal") != "refusal"
        reason = f"response_safety={result.get('response_safety')}, response_refusal={result.get('response_refusal')}"
    else:
        tasks = {"prompt_safety": SAFETY_LABELS, "jailbreak_detection": JAILBREAK_TASK}
        result = model.classify_text(text, tasks)
        jailbreak = result.get("jailbreak_detection", ["benign"])
        unsafe = result.get("prompt_safety") == "unsafe" or any(label != "benign" for label in jailbreak)
        reason = f"prompt_safety={result.get('prompt_safety')}, jailbreak_detection={jailbreak}"

    if unsafe:
        return {"action": "BLOCKED", "blocked_reason": reason}
    return {"action": "NONE"}


@app.get("/health")
async def health():
    return {"status": "ok", "device": device}
