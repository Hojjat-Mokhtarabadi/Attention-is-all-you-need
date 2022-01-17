from typing import List
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
import torch


class PrepareData:
    """
    download data set, tokenize it, build a vocabulary of tokens (their corresponding index in dictionary)
    """
    def __init__(self):
        self._SRC_LANGUAGE = 'de'
        self._TRG_LANGUAGE = 'en'
        self.split = 'train'

        # special symbols
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.UNK, self.PAD, self.BOS, self.EOS, = 0, 1, 2, 3

        # transforms
        self.token_transform = {
            self._SRC_LANGUAGE: get_tokenizer('spacy', 'de_core_news_sm'),
            self._TRG_LANGUAGE: get_tokenizer('spacy', 'en_core_web_sm')
        }
        self.vocab_transform = self._build_vocab()
        self.transform_ops = self._apply_transform()

    def __call__(self, split):
        return Multi30k(root=f"../{split}",split=split, 
                        language_pair=(self._SRC_LANGUAGE,
                                       self._TRG_LANGUAGE))

    @property
    def target_size(self):
        return len(self.vocab_transform[self._TRG_LANGUAGE])

    @property
    def input_size(self):
        return len(self.vocab_transform[self._SRC_LANGUAGE])

    def _yield_tokens(self, data_iter, lang):
        lang2idx = {self._SRC_LANGUAGE: 0, self._TRG_LANGUAGE: 1}
        tokenizer = self.token_transform[lang]
        for data_sample in data_iter:
            yield tokenizer(data_sample[lang2idx[lang]])

    def _build_vocab(self):
        vocab_transform = {}
        for ln in [self._SRC_LANGUAGE, self._TRG_LANGUAGE]:
            train_iter = Multi30k(split=self.split,
                                  language_pair=(self._SRC_LANGUAGE,
                                                 self._TRG_LANGUAGE))
            vocab_transform[ln] = build_vocab_from_iterator(
                self._yield_tokens(train_iter, ln),
                min_freq=1,
                specials=self.special_symbols,
                special_first=True)
            vocab_transform[ln].set_default_index(self.UNK)

        return vocab_transform

    def _sequential_transform(self, *transforms):
        def apply(text):
            for transform in transforms:
                text = transform(text)
            return text

        return apply

    def _tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS]), 
                          torch.tensor(token_ids),
                          torch.tensor([self.EOS])))

    def _apply_transform(self):
        transform_ops = {}
        for ln in [self._SRC_LANGUAGE, self._TRG_LANGUAGE]:
            transform_ops[ln] = self._sequential_transform(
                self.token_transform[ln],  # tokenize
                self.vocab_transform[ln],  # numerical
                self._tensor_transform  # tensor
            )
        return transform_ops

    # a callable/function to further process batches or dataset items
    def collate_fn(self, batch):
        src_seq, trg_seq = [], []
        for src_sample, trg_sample in batch:
            src_seq.append(self.transform_ops[self._SRC_LANGUAGE](src_sample.rstrip("\n")))
            trg_seq.append(self.transform_ops[self._TRG_LANGUAGE](trg_sample.rstrip("\n")))
        src_batch = pad_sequence(src_seq, padding_value=self.PAD)
        trg_batch = pad_sequence(trg_seq, padding_value=self.PAD)

        return src_batch.transpose(0, 1), trg_batch.transpose(0, 1)
