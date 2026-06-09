# How to Reduce CPU Spikes for AI Summarisation with Ollama

Docker Compose setup and model configuration for running Llama 3.2 3B with Ollama on CPU-only hardware, tuned to reduce the sustained 100% CPU spikes caused by the prefill phase during long-form document summarisation.

**Article:** [How to Reduce CPU Spikes for AI Summarisation with Ollama](https://www.sundeepmachado.com/2026/05/how-to-reduce-cpu-spikes-for-ai.html)

## Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | Ollama service with memory and attention optimisations enabled |
| `Modelfile` | Custom model variant with resource-stability parameters |
| `init_optimized_model.sh` | One-time initialisation script — pulls the base model and compiles the optimised variant |

## Quick start

### 1. Start Ollama

```bash
docker compose up -d
```

### 2. Build the optimised model variant (first run only)

Copy `Modelfile` into the container and run the init script:

```bash
docker compose exec ollama mkdir -p /config
docker compose cp Modelfile ollama:/config/Modelfile
docker compose exec ollama sh /config/init_optimized_model.sh
```

This pulls `llama3.2:3b` and compiles it into a new variant called `llama3.2:3b-clean` with the parameters in `Modelfile`.

### 3. Run a summarisation

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b-clean",
  "prompt": "Summarise the following document:\n\n<your text here>",
  "stream": false
}'
```

## What each setting does

### `docker-compose.yml`

| Variable | Value | Effect |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | `true` | Reduces memory bandwidth during attention, lowering CPU pressure |
| `OLLAMA_KV_CACHE_TYPE` | `q4_0` | Quantises the KV cache to 4-bit, shrinking working memory and reducing cache-read CPU cost |
| `OLLAMA_CONTEXT_LENGTH` | `4096` | Caps the context window to prevent unbounded prefill work |
| `OLLAMA_KEEP_ALIVE` | `0s` | Unloads the model immediately after each request, freeing RAM between calls |

### `Modelfile`

| Parameter | Value | Effect |
|---|---|---|
| `num_thread` | `3` | Limits CPU thread usage to avoid saturating all cores during prefill |
| `num_batch` | `128` | Smaller batch size processes tokens in chunks, spreading the prefill spike |
| `num_ctx` | `4096` | Matches the context cap set in the compose file |
| `num_predict` | `80` | Limits output length — summaries don't need long responses |
