import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, drop_rate):
        super().__init__()
        if d_model is None:
            raise ValueError("model dimension should be specified")

        self.heads = heads
        self.d_model = d_model
        assert self.d_model % self.heads == 0
        # according to the original paper -> d_key = d_value = d_model // heads
        self.d_key = self.d_model // self.heads
        self.all_heads_size = self.d_key * self.heads
        self.dropout_rate = drop_rate

        self._query = nn.Linear(self.d_model, self.all_heads_size)
        self._key = nn.Linear(self.d_model, self.all_heads_size)
        self._value = nn.Linear(self.d_model, self.all_heads_size)

        self._out = nn.Linear(self.all_heads_size, self.d_model)

        self._drop_out = nn.Dropout(self.dropout_rate)

    def _split_heads(self, x):
        """
        shapes:
            x_shape -> [bs, n, heads*d_key]
            x_new_shape -> [bs, n, heads, d_key]
        """
        x_new_shape = x.shape[:-1] + (self.heads, self.d_key)
        return x.view(*x_new_shape).permute(0, 2, 1, 3)

    def _scaled_dot_product(self, q, k, mask):
        scaled_dot_logits = torch.matmul(
            q, k.transpose(-1, -2)) / self.d_key ** 0.5

        if mask is not None:
            scaled_dot_logits.masked_fill_(mask, -1e9)

        return scaled_dot_logits # [bs, head, n, n]

    def _out_projection(self, att):
        bs = att.shape[0]
        seq_len = att.shape[2]
        return att.view(bs, seq_len, -1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None):
        # q = k = v = [bs, n, d_model]
        q = self._split_heads(self._query(query))
        k = self._split_heads(self._key(key))
        v = self._split_heads(self._value(value))

        alignment_scores = self._scaled_dot_product(q, k, mask)
        attention_weights = F.softmax(alignment_scores, dim=-1) # [bs, heads, n, n]
        attention = torch.matmul(attention_weights, v) # [bs, heads, n, d_value]
        attention = self._drop_out(attention)

        out = self._out(self._out_projection(attention)) # [bs, n, d_model]
        return out
