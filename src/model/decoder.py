import torch
from torch import nn, Tensor
from model import MultiHeadAttention


class DecoderStack(nn.Module):
    def __init__(self, heads, d_model, layers, drop_rate):
        super().__init__()
        self.num_dec_layers = layers

        self.decoder_stack = nn.ModuleList(
            [DecoderBlock(heads, d_model, drop_rate) for _ in range(self.num_dec_layers)])

    def forward(self, target: Tensor, enc_out, padding_mask, shift_mask):
        dec_out = target
        for dec in self.decoder_stack:
            dec_out = dec(dec_out, enc_out, padding_mask, shift_mask)

        return dec_out


class DecoderBlock(nn.Module):
    def __init__(self, heads, d_model, drop_rate):
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = drop_rate

        # self-attention
        self._sa = MultiHeadAttention(heads, d_model, drop_rate)
        # cross-attention
        self._cross_att = MultiHeadAttention(heads, d_model, drop_rate)

        self._layer_norm = nn.LayerNorm(self.d_model)
        self._layer_norm2 = nn.LayerNorm(self.d_model)
        self._layer_norm3 = nn.LayerNorm(self.d_model)

        self._mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def forward(self, x: Tensor, enc_out, padding_mask, shift_mask):
        sa = self._sa(query=x, key=x, value=x, mask=shift_mask)
        normalized_sa = self._layer_norm(sa + x) # residual

        cross_att = self._cross_att(
            query=normalized_sa, key=enc_out, value=enc_out, mask=padding_mask)
        normalized_cross = self._layer_norm2(normalized_sa + cross_att) # residual

        mlp = self._mlp(normalized_cross)
        normalized_mlp = self._layer_norm3(mlp + normalized_cross)

        return normalized_mlp
