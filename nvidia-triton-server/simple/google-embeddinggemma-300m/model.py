import os
import numpy as np
import triton_python_backend_utils as pb_utils

# Let Triton see GPUs; allow override via env (e.g., "cuda:0", "cpu")
DEVICE_ENV = os.environ.get("EMBEDDING_DEVICE", "").strip()

from transformers import AutoTokenizer, AutoModel
import torch

MODEL_SUBDIR = os.path.join(os.path.dirname(__file__), "embeddinggemma-300m")
MODEL_LOCAL_PATH = os.environ.get("EMBEDDINGGEMMA_LOCAL_PATH", MODEL_SUBDIR)
MAX_LENGTH = int(os.environ.get("EMBEDDING_MAX_TOKENS", "2048"))

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

class TritonPythonModel:
    def initialize(self, args):
        if not os.path.isdir(MODEL_LOCAL_PATH):
            raise RuntimeError("Local model files not found at '{}'.".format(MODEL_LOCAL_PATH))

        # pick device
        if DEVICE_ENV:
            self.device = torch.device(DEVICE_ENV)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH,local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL_LOCAL_PATH,local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        # Warmup
        _ = self._encode(["warmup"], "document")

    @staticmethod
    def _bytes_to_str_list(arr):
        flat = arr.reshape(-1)
        out = []
        for x in flat:
            out.append(x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray)) else str(x))
        return out

    @staticmethod
    def _normalize_mode(m):
        m = (m or "").strip().lower()
        return "query" if m == "query" else "document"

    @staticmethod
    def _apply_prompt(texts, mode):
        if mode == "query":
            return [f"task: search result | query: {t}" for t in texts]
        return [f"title: none | text: {t}" for t in texts]

    @torch.no_grad()
    def _encode(self, texts, mode):
        prepared = self._apply_prompt(texts, mode)
        tok = self.tokenizer(
            prepared, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self.device)

        # autocast for speed on GPU
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(**tok)
        else:
            out = self.model(**tok)

        emb = mean_pooling(out.last_hidden_state, tok["attention_mask"])
        return emb.detach().cpu().numpy().astype(np.float32)

    def execute(self, requests):
        responses = []
        for req in requests:
            try:
                t_in = pb_utils.get_input_tensor_by_name(req, "TEXT")
                if t_in is None:
                    raise pb_utils.TritonModelException("Required input 'TEXT' missing")
                texts = self._bytes_to_str_list(t_in.as_numpy())

                mode = "document"
                m_in = pb_utils.get_input_tensor_by_name(req, "MODE")
                if m_in is not None:
                    mv = self._bytes_to_str_list(m_in.as_numpy())
                    if mv:
                        mode = self._normalize_mode(mv[0])

                vecs = self._encode(texts, mode)
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("EMBEDDINGS", vecs)]
                ))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
        return responses
