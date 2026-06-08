import httpx

GUARD_URL = "http://localhost:8765"


class GuardClient:
    def __init__(self, base_url: str = GUARD_URL, timeout: float = 5.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def check(
        self,
        prompt: str | None = None,
        response: str | None = None,
        tasks: list[str] | None = None,
    ) -> dict:
        payload = {}
        if prompt is not None:
            payload["prompt"] = prompt
        if response is not None:
            payload["response"] = response
        if tasks is not None:
            payload["tasks"] = tasks
        r = await self._client.post("/guard", json=payload)
        r.raise_for_status()
        return r.json()

    async def is_prompt_safe(self, prompt: str) -> bool:
        # Block on an explicit "unsafe" verdict OR any non-benign jailbreak label —
        # jailbreak attempts often read as benign on the safety axis alone.
        result = await self.check(
            prompt=prompt,
            tasks=["prompt_safety", "jailbreak_detection"],
        )
        if result.get("prompt_safety") == "unsafe":
            return False
        jailbreak = result.get("jailbreak_detection", ["benign"])
        return all(label == "benign" for label in jailbreak)

    async def is_response_safe(self, prompt: str | None, response: str) -> bool:
        # Only block "unsafe" responses that aren't refusals — the model tags
        # refusals as unsafe based on the topic they decline, not the reply itself.
        result = await self.check(
            prompt=prompt,
            response=response,
            tasks=["response_safety", "response_refusal"],
        )
        return not (
            result.get("response_safety") == "unsafe"
            and result.get("response_refusal") != "refusal"
        )

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()
