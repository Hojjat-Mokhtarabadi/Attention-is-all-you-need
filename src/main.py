from argparse import ArgumentParser
import time

import torch
from torch.optim import Adam
from torch import nn
from tqdm import trange, tqdm

from dataset import PrepareData
from model import Seq2SeqTransformer
from engine import train_step, val_step, on_validation_end


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Preparing Data...")
    multi30kdata = PrepareData()
    print("Preparing Model...")
    model = Seq2SeqTransformer(args.heads, args.d_model, 
                               args.layers, args.drop_rate, 
                               input_vocab_size=multi30kdata.input_size,
                               target_vocab_size=multi30kdata.target_size,
                               max_seq_len=10000, 
                               device=device).to(device)

    # configurations
    optim = Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=multi30kdata.PAD) # ignore pad index

    if args.eval:
        model = model.load_state_dict(torch.load('../checkpoint.pth')['model_state_dict'])
        on_validation_end(model, multi30kdata, 10)

    epoch = 0
    print("Starting training...")
    if args.resume:
        checkpoint = torch.load('../checkpoint.pth')
        model.state_dict = model.load_state_dict(checkpoint['model_state_dict'])
        optim.state_dict = optim.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        
    for e in trange(epoch, args.max_epoch, desc='Epoch'):
        start = time.time()
        train_loss = train_step(model, optim, criterion, multi30kdata, args.batch_size, device)
        end = time.time()
        val_loss = val_step(model, criterion, multi30kdata, args.batch_size, device)
        tqdm.write(f"Epoch {e}: [Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Epoch time: {end-start:.4f}]")

        if args.save:
            check_point = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': train_loss 
            }
            torch.save(check_point, '../checkpoint.pth')

        if e == args.max_epoch-1:
            translated_seq = on_validation_end(model, multi30kdata, 10) 
            print(translated_seq)



parser = ArgumentParser("Implementation of Attention is all you need")
# model
parser.add_argument('--heads', default=4, type=int)
parser.add_argument('--d_model', default=256, type=int)
parser.add_argument('--layers', default=4, type=int)
parser.add_argument('--drop_rate', default=0.2, type=float)
# training
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=128, type=int)
# fit
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
    