import torch
from torch import nn, Tensor
from model import MultiHeadAttention


class EncoderStack(nn.Module):
    def __init__(self, heads, d_model, layers, drop_rate):
        super().__init__()
        self.num_enc_layer = layers

        self._encoder_stack = nn.ModuleList(
            [EncoderBlock(heads, d_model, drop_rate) for _ in range(self.num_enc_layer)])

    def forward(self, input: Tensor, padding_mask):
        enc_out = input
        for enc in self._encoder_stack:
            enc_out = enc(enc_out, padding_mask)

        return enc_out


class EncoderBlock(nn.Module):
    def __init__(self, heads, d_model, drop_rate):
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = drop_rate
        # self-attention
        self._sa = MultiHeadAttention(heads, d_model, drop_rate)

        self._layer_norm = nn.LayerNorm(self.d_model)
        self._layer_norm2 = nn.LayerNorm(self.d_model)

        self._mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def forward(self, x: Tensor, mask=None):
        sa_out = self._sa(query=x, key=x, value=x, mask=mask)

        normalized_att = self._layer_norm(sa_out + x) # residual

        mlp_out = self._mlp(normalized_att)

        normalized_mlp = self._layer_norm2(mlp_out + normalized_att) # residual

        return normalized_mlp
