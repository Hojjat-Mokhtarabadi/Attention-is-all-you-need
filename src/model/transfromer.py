import torch
from torch import nn, LongTensor
from torch.functional import Tensor
from model import EncoderStack, DecoderStack, PositionalEncoding, TokenEmbedding


class Seq2SeqTransformer(nn.Module):
    def __init__(self, heads,
                 d_model, layers, 
                 drop_rate,
                 input_vocab_size,
                 target_vocab_size,
                 max_seq_len,
                 device):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.layers = layers
        self.dropout_rate = drop_rate
        self.target_vocab_size = target_vocab_size
        self.max_seq_len = max_seq_len
        self.device=device

        # input word embeddings
        self._inp_word_emb = TokenEmbedding(input_vocab_size, self.d_model) # emb_size == d_model
        self._trg_word_emb = TokenEmbedding(target_vocab_size, self.d_model)
        self._position_enc = PositionalEncoding(self.d_model, self.dropout_rate, max_seq_len)

        self._encoder = EncoderStack(self.heads, self.d_model, self.layers, self.dropout_rate)
        self._decoder = DecoderStack(self.heads, self.d_model, self.layers, self.dropout_rate)

        # final layer to project decoder output to the desired dimension
        self._final_linear = nn.Linear(self.d_model, self.target_vocab_size)

    def encode(self, src: Tensor):
        pad_mask = self._padding_mask(src, 1)
        src = self._position_enc(self._inp_word_emb(src))
        memory = self._encoder(src, pad_mask)
        return memory

    def decode(self, enc_out, src, trg):
        _, dec_pad_mask, dec_shift_mask = self._create_mask(src, trg)
        trg = self._position_enc(self._trg_word_emb(trg))
        dec_out = self._decoder(trg, enc_out, dec_pad_mask, dec_shift_mask)
        proj2target_vocab = self._final_linear(dec_out)

        return proj2target_vocab

    def forward(self, src: LongTensor, trg: LongTensor):
        src, trg = src, trg # [bs, n]

        positioned_inp_seq = self._position_enc(self._inp_word_emb(src)) # [bs, n, emb_size]
        positioned_trg_seq = self._position_enc(self._trg_word_emb(trg))

        enc_padding_mask, dec_padding_mask, dec_shift_mask = self._create_mask(src, trg)

        enc_out = self._encoder(positioned_inp_seq, enc_padding_mask)
        dec_out = self._decoder(positioned_trg_seq, enc_out, dec_padding_mask, dec_shift_mask)
        final_out = self._final_linear(dec_out)

        return final_out

    # ---- Methods --------------------------------------------------------------

    def _padding_mask(self, inp, pad_idx):
        bs, seq_len = inp.size(0), inp.size(1)
        return (inp == pad_idx).view(bs, 1, 1, seq_len)

    def _shift_mask(self, inp):
        bs, seq_len = inp.size(0), inp.size(1)
        shift_mask = (torch.ones(seq_len, seq_len, device=self.device).triu(diagonal=1) == 1)
        return shift_mask.view(1, 1, seq_len, seq_len)

    def _create_mask(self, input, target, pad_idx=1):
        # encoder input mask
        enc_padding_mask = self._padding_mask(input, pad_idx)

        # cross-attention mask in decoder
        dec_gd_padding_mask = self._padding_mask(input, pad_idx)

        # look ahead mask for decoder input
        dec_shift_mask = self._shift_mask(target)
        dec_input_padding_mask = self._padding_mask(target, pad_idx)
        # dec_shift_mask = torch.maximum(dec_shift_mask, dec_input_padding_mask)
        dec_shift_mask = dec_shift_mask | dec_input_padding_mask

        return enc_padding_mask, dec_gd_padding_mask, dec_shift_mask
