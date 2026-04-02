from __future__ import annotations

import torch
from torch import nn

class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tag_size: int,      
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 2. LSTM Layer
        lstm_dropout = dropout if num_layers > 1 else 0
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )

        # 3. Linear Layer
        self.fc = nn.Linear(hidden_size * 2, tag_size)

        # 4. Dropout Layer 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        
        lstm_out, _ = self.lstm(embeds)        
        return self.fc(lstm_out)