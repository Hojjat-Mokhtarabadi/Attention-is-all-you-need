import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        i = torch.arange(0, emb_size)

        angle = 1 / (10000 ** (2*(i/2) / emb_size))
        encoding =  pos * angle

        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', encoding) # [max_len, d_model]

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :])
        
# convert a list of token indices to their corresponding embbeding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        return self.token_emb(x)


class Translate(nn.Module):
    def __init__(self, model, data, size, max_len):
        super(Translate, self).__init__()

        self.model = model.to('cpu')
        self.data = data
        self.val_loader = DataLoader(
            self.data('valid'),
            batch_size= 1,
            collate_fn=self.data.collate_fn
        )
        self.size = size
        self.max_len = max_len

    def __call__(self):
        return self._greedy_decode()

    def _greedy_decode(self):
        src, trg_real = next(iter(self.val_loader))
        src, trg_real = src.to('cpu'), trg_real.to('cpu')
        mem = self.model.encode(src)

        trg_pred = torch.tensor([self.data.BOS]).view(1, 1) # predicted sequence
        for _ in range(self.max_len):
            preds = self.model.decode(mem, src, trg_pred)
            next_word = preds.argmax(dim=-1)[:, -1:]

            trg_pred = torch.cat([trg_pred, next_word], dim=-1)

            if next_word == self.data.EOS:
                break

        return trg_pred








