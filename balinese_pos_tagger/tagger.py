from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import torch

from .utils.text_utils import tokenize
from .io import load_artifacts

@dataclass(frozen=True)
class LoadedArtifacts:
    max_len: int
    word2idx: Dict[str, int]
    idx2tag: Dict[int, str]
    embedding_dim: int
    hidden_size: int
    num_layers: int

class PosTagger:
    """
    Balinese Part-of-Speech Tagger menggunakan arsitektur Bi-LSTM.
    """
    
    __version__ = "0.1.0"

    def __init__(
        self, 
        model: Optional[torch.nn.Module] = None, 
        art: Optional[LoadedArtifacts] = None, 
        device: str = "cpu"
    ) -> None:
        
        if model is None or art is None:
            current_dir = Path(__file__).resolve().parent
            default_path = current_dir / "resources"
            
            if not default_path.exists():
                raise FileNotFoundError(
                    f"Default resources not found at {default_path}. "
                    "Pastikan folder 'resources' tersedia di dalam paket library."
                )
            
            model, cfg, word2idx, idx2tag = load_artifacts(default_path, device=device)
            art = LoadedArtifacts(
                max_len=int(cfg.max_len),
                word2idx=word2idx,
                idx2tag=idx2tag,
                embedding_dim=int(cfg.embedding_dim),
                hidden_size=int(cfg.hidden_size),
                num_layers=int(cfg.num_layers)
            )

        self.model = model
        self.art = art
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.max_len = self.art.max_len
        self.embedding_dim = self.art.embedding_dim
        self.hidden_size = self.art.hidden_size
        self.num_layers = self.art.num_layers
        
        self.pad_word_id = self.art.word2idx.get("<PAD>", 0)
        self.unk_word_id = self.art.word2idx.get("<UNK>", 1)

    @classmethod
    def from_pretrained(cls, artifact_dir: str | Path, device: str = "cpu") -> "PosTagger":
        """Load model dari direktori spesifik."""
        model, cfg, word2idx, idx2tag = load_artifacts(Path(artifact_dir), device=device)
        art = LoadedArtifacts(
            max_len=int(cfg.max_len),
            word2idx=word2idx,
            idx2tag=idx2tag,
            embedding_dim=int(cfg.embedding_dim),
            hidden_size=int(cfg.hidden_size),
            num_layers=int(cfg.num_layers)
        )
        return cls(model=model, art=art, device=device)

    def tag(self, text_or_tokens: Union[str, List[str]]) -> List[Tuple[str, str]]:
        # 1. Preprocessing (Tokenisasi)
        if isinstance(text_or_tokens, str):
            tokens = tokenize(text_or_tokens)
        elif isinstance(text_or_tokens, list):
            tokens = text_or_tokens
        else:
            raise ValueError("Input harus berupa string atau list token.")

        if not tokens:
            return []

        # 2. Convert tokens ke numerical IDs
        ids = [self.art.word2idx.get(t, self.unk_word_id) for t in tokens]
        
        original_len = len(ids)

        # 3. Truncation & Padding
        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids += [self.pad_word_id] * (self.max_len - len(ids))

        # 4. Inference
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x) 
            
            preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

        # 5. Mapping kembali ID ke Label (Tag)
        valid_len = min(original_len, self.max_len)
        preds = preds[:valid_len]
        
        tags = [self.art.idx2tag.get(int(i), "UNK") for i in preds]

        return list(zip(tokens[:valid_len], tags))

    def __repr__(self) -> str:
        return (f"BalinesePosTagger(version='{self.__version__}', "
                f"device='{self.device}', "
                f"max_len={self.max_len}, "
                f"embedding_dim={self.embedding_dim}, "
                f"hidden_size={self.hidden_size})")