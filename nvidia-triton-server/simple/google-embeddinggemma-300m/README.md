# Serving an Embedding Model on NVIDIA Triton Server

Deploys Google's [EmbeddingGemma-300M](https://huggingface.co/google/gemma-3-300m-pt) model as a Triton Inference Server Python backend, serving float32 embeddings over HTTP.

**Article:** [How to Deploy Google's Latest Embedding Model on NVIDIA Triton Server](https://www.sundeepmachado.com/2025/10/how-to-deploy-googles-latest-embedding.html)

## Files

| File | Purpose |
|---|---|
| `dockerfile` | Builds the Triton image with PyTorch (CUDA 12.4) and HuggingFace dependencies |
| `config.pbtxt` | Triton model config — name, backend, I/O shapes, dynamic batching |
| `model.py` | `TritonPythonModel` implementation — tokenisation, prompt templates, mean pooling |
| `request.sh` | Example `curl` inference request |
| `response.json` | Sample response showing the `[2, 768]` embedding output |

## Prerequisites

- Docker with NVIDIA Container Toolkit
- Model weights downloaded separately from HuggingFace and placed at:
  ```
  model_repository/embeddinggemma-300m/1/embeddinggemma-300m/
  ```

## Build

```bash
docker build -t embeddinggemma-triton -f dockerfile .
```

## Run

```bash
docker run --gpus all --rm \
  -v $(pwd)/model_repository:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  embeddinggemma-triton \
  tritonserver --model-repository=/models
```

## Inference

```bash
bash request.sh
```

Or send the request directly:

```bash
curl -X POST localhost:8000/v2/models/embeddinggemma-300m/infer -d '{
  "inputs": [
    {
      "name": "TEXT",
      "shape": [1, 2],
      "datatype": "BYTES",
      "data": ["This is the first document", "This is the second document"]
    }
  ]
}'
```

Response shape is `[N, 768]` — one 768-dimensional float32 vector per input string.

## Configuration

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDINGGEMMA_LOCAL_PATH` | `<model.py dir>/embeddinggemma-300m` | Path to downloaded model weights |
| `EMBEDDING_DEVICE` | auto-detect | Force device, e.g. `"cuda:0"` or `"cpu"` |
| `EMBEDDING_MAX_TOKENS` | `2048` | Tokenizer truncation length |

### Modes

Pass an optional `MODE` input alongside `TEXT` to switch prompt templates:

| Mode | Prompt template applied |
|---|---|
| `query` | `task: search result \| query: {text}` |
| anything else (default) | `title: none \| text: {text}` |

### Batching

`config.pbtxt` enables dynamic batching with `max_batch_size: 64` and a preferred batch size of 2. GPU inference uses `torch.autocast` (float16) for speed.
