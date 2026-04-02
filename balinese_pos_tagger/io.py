from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import torch

from .model import BiLSTMTagger

@dataclass(frozen=True)
class ArtifactConfig:
    embedding_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    max_len: int

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _pick_cfg(params: Dict[str, Any]) -> ArtifactConfig:
    allowed = {"embedding_dim", "hidden_size", "num_layers", "dropout", "max_len"}
    filtered = {k: params[k] for k in allowed if k in params}

    required = {"embedding_dim", "hidden_size", "num_layers", "dropout", "max_len"}
    missing = sorted(required - set(filtered))
    if missing:
        raise ValueError(f"params.json missing required keys: {missing}")

    return ArtifactConfig(**filtered)

def load_artifacts(
    artifact_dir: Path, 
    device: str = "cpu"
) -> Tuple[torch.nn.Module, ArtifactConfig, Dict[str, int], Dict[int, str]]:
    
    params_path = artifact_dir / "params.json"
    model_path = artifact_dir / "model.pt"
    word2idx_path = artifact_dir / "word2idx.json"
    idx2tag_path = artifact_dir / "idx2tag.json"

    for p in (params_path, model_path, word2idx_path, idx2tag_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    params = _read_json(params_path)
    cfg = _pick_cfg(params)
    word2idx = _read_json(word2idx_path)
    idx2tag = {int(k): v for k, v in _read_json(idx2tag_path).items()}

    model = BiLSTMTagger(
        vocab_size=len(word2idx),
        tag_size=len(idx2tag),
        embedding_dim=cfg.embedding_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_idx=int(word2idx.get("<PAD>", 0)),
    )

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, cfg, word2idx, idx2tag