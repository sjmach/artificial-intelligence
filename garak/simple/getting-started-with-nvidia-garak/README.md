# Getting Started with Nvidia Garak

Installs and runs [Garak](https://github.com/NVIDIA/garak), NVIDIA's open-source LLM vulnerability scanner, against a local Ollama model.

**Article:** [Getting started with Nvidia Garak](https://www.sundeepmachado.com/2026/06/getting-started-with-nvidia-garak.html)

## Files

| File | Purpose |
|---|---|
| `setup.sh` | Creates a Python venv and installs `garak` |
| `scan_ollama.sh` | Runs a Garak probe against a local Ollama model |

## Prerequisites

- Python 3 with `venv`
- Ollama running locally with the target model already pulled (run `ollama list` to confirm the exact name)

## Quick start

### 1. Install Garak

```bash
bash setup.sh
```

### 2. Run a probe against an Ollama model

```bash
bash scan_ollama.sh llama3.2:3b-clean dan.DanInTheWild
```

Both arguments are optional and default to `llama3.2:3b-clean` and `dan.DanInTheWild`. The model name must exactly match an entry from `ollama list`.

## What Garak checks for

Garak probes target LLMs for seven vulnerability categories:

- Prompt injection
- Jailbreaks / safety guardrail bypasses
- Training data exposure
- Toxicity generation
- Hallucinations and misinformation
- Encoding-based filter evasion
- Malware generation

Beyond Ollama, Garak also targets OpenAI, HuggingFace, Bedrock, Groq, NVIDIA NIMs, and generic REST API endpoints.

## Architecture

| Component | Role |
|---|---|
| Generator | Connects to the target LLM |
| Probe | Crafts attack payloads for a specific vulnerability |
| Detector | Analyses responses for a successful attack |
| Harness | Orchestrates the probe → generator → detector workflow |

## Results

Run output is written to:

```
~/.local/share/garak/garak_runs
```

Results are scored on NVIDIA's Defense Capability (DC) scale — a result below DC-3 indicates an unguarded model. The example in the article showed a 90% DAN jailbreak success rate against an unprotected Llama 3.2 model.
