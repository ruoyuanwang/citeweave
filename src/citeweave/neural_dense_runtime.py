from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from citeweave.io import sha256_file


class OnnxSentenceEncoder:
    """Sentence-Transformers-compatible encoder for official pooled ONNX exports."""

    def __init__(
        self,
        model: str,
        *,
        revision: str,
        onnx_file: str = "onnx/model.onnx",
        provider: str = "CPUExecutionProvider",
        intra_op_threads: int | None = None,
        local_files_only: bool = False,
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        candidate = Path(model)
        if candidate.is_dir():
            snapshot = candidate.resolve()
        else:
            snapshot = Path(
                snapshot_download(
                    repo_id=model,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            ).resolve()
        model_path = snapshot / onnx_file
        data_path = model_path.with_name(f"{model_path.name}_data")
        if not model_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(
                f"Official ONNX graph or external data is missing: {model_path}"
            )
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"ONNX provider {provider!r} is unavailable; found {available}"
            )
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads
            options.inter_op_num_threads = 1
        if provider == "DmlExecutionProvider":
            options.enable_mem_pattern = False
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[provider],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=local_files_only
        )
        config_path = snapshot / "sentence_bert_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.max_seq_length = int(config["max_seq_length"])
        self.input_names = {item.name for item in self._session.get_inputs()}
        output_names = {item.name for item in self._session.get_outputs()}
        if "sentence_embedding" not in output_names:
            raise RuntimeError(
                "Official ONNX export must expose the pooled sentence_embedding output"
            )
        self.runtime_metadata = {
            "runtime_backend": "onnxruntime",
            "onnxruntime_version": ort.__version__,
            "onnx_provider": provider,
            "onnx_provider_options": self._session.get_provider_options(),
            "onnx_file": onnx_file,
            "onnx_model_sha256": sha256_file(model_path),
            "onnx_data_sha256": sha256_file(data_path),
            "onnx_intra_op_threads": intra_op_threads,
        }

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        **_: Any,
    ) -> np.ndarray:
        if not convert_to_numpy:
            raise ValueError("OnnxSentenceEncoder only returns NumPy arrays")
        if show_progress_bar:
            raise ValueError("Progress bars are intentionally disabled")
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            tokens = self._tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            feed = {
                name: np.asarray(tokens[name], dtype=np.int64)
                for name in self.input_names
            }
            embeddings = self._session.run(["sentence_embedding"], feed)[0]
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(
                    norms, np.finfo(np.float32).tiny
                )
            outputs.append(embeddings)
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
