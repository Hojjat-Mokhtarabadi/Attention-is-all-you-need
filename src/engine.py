import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import AverageMeter
from model import Translate


def train_step(model, optim, criterian, multi30kdata, bs, device):
    model.train()
    loss_train = AverageMeter()
    train_data = multi30kdata('train')
    train_loader = DataLoader(train_data,
                          batch_size=bs,
                          collate_fn=multi30kdata.collate_fn)

    for idx , (src, trg) in enumerate(train_loader):
        """ Because of 'teacher force' type of training we need to shift the target one step to the 
        right [trg_real] so the model must predict the next token in the given sequence.

        The criterian expects (bs*seq_len, vocab_size) for prediction and (bs*seq_len) for real target.
        """

        src, trg = src.to(device), trg.to(device)
        trg_input = trg[:, :-1]
        trg_pred = model(src, trg_input)
        trg_real = trg[:, 1:]

        loss = criterian(trg_pred.reshape(-1, trg_pred.shape[-1]), trg_real.reshape(-1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_train.update(loss.item(), 1)

    return loss_train.avg()


def val_step(model, criterian, multi30kdata, bs, device):
    model.eval()
    val_loss = AverageMeter()
    val_data = multi30kdata('valid')
    val_loader = DataLoader(val_data,
                            batch_size=bs,
                            collate_fn=multi30kdata.collate_fn)
    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:, :-1]
            trg_pred = model(src, trg_input).permute(1, 0, 2)
            trg_real = trg[:, 1:]

            loss = criterian(trg_pred.reshape(-1, trg_pred.shape[-1]), trg_real.reshape(-1))
            val_loss.update(loss.item(), 1)

    return val_loss.avg()

def on_validation_end(model: nn.Module, dataset: Dataset, sample_size: int):
    model.eval()
    translation_test = Translate(model, dataset, sample_size, max_len=10000)
    return translation_test()