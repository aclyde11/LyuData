import argparse
from model.vocab import get_vocab_from_file, START_CHAR, END_CHAR
from model.model import DockRegressor
import torch.utils.data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn
import torch.nn.functional as F
from tqdm import tqdm
import os

def getconfig(args):
    config_ = {
        'epochs': 10,
        'batch_size': 512,
        'vocab_size': 28,
        'emb_size': 32,
        'sample_freq': 1,
        'max_len': 150
    }

    return config_

def count_valid_samples(smiles):
    from rdkit import Chem
    count = 0
    for smi in smiles:
        try:
            mol     = Chem.MolFromSmiles(smi[1:-1])
        except:
            continue
        if mol is not None:
            count += 1
    return count

def get_input_data(fname, fname2, c2i):
    lines = open(fname, 'r').readlines()
    lines = (map(lambda x: x.split(','), (filter(lambda x: len(x) != 0, map(lambda x: x.strip(), lines)))))

    lines1 = [torch.from_numpy(np.array([c2i(START_CHAR)] + list(map(lambda x: int(x), y)), dtype=np.int64)) for y in
              lines]

    lines = open(fname2, 'r')
    ys = list(map(lambda x : torch.tensor(float(x.strip())), lines))
    assert(len(ys) == len(lines1))
    print("Read", len(lines1), "SMILES.")

    return lines1, ys

def mycollate(x):
    x_batches = []
    y_batchese = []
    for i in x:
        x_batches.append(i[0])
        y_batchese.append(i[1])
    y_batchese = torch.stack(y_batchese, dim=0).float()
    return x_batches, y_batchese


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, s, e):
        self.s = s
        self.e = e
        assert (len(self.s) == len(self.e))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, item):
        return self.s[item], self.e[item]


def train_epoch(model, optimizer, dataloader, config):
    model.train()
    lossf = nn.CrossEntropyLoss().cuda()
    for i, (y, y_hat) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        y_hat = y_hat.float().cuda()
        y = [x.cuda() for x in y]

        pred = model(y)
        loss = lossf(pred, y_hat).mean()
        loss.backward()
        optimizer.step()
        print(loss.item())


def main(args):
    config = getconfig(args)
    print("loading data.")
    vocab, c2i, i2c = get_vocab_from_file(args.i + "/vocab.txt")
    print("Vocab size is", len(vocab))
    s, e = get_input_data(args.i + "/out.txt", args.i + "/out_y.txt", c2i)
    input_data = ToyDataset(s, e)
    print("Done.")

    ## make data generator
    dataloader = torch.utils.data.DataLoader(input_data, pin_memory=True, batch_size=config['batch_size'],
                                             collate_fn=mycollate)

    model = DockRegressor(config['vocab_size'], config['emb_size'], max_len=config['max_len']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_start = 0
    if args.ct:
        print("Continuing from save.")
        pt = torch.load(args.logdir + "/autosave.model.pt")
        model.load_state_dict(pt['state_dict'])
        optimizer.load_state_dict(pt['optim_state_dict'])
        epoch_start = pt['epoch'] + 1


    for epoch in range(epoch_start, config['epochs']):
        train_epoch(model, optimizer, dataloader, config)

        torch.save(
            {
                'state_dict' : model.state_dict(),
                'optim_state_dict' : optimizer.state_dict(),
                'epoch' : epoch
            }, args.logdir + "/autosave.model.pt"
        )


if __name__ == '__main__':
    print("Note: This script is very picky. This will only run on a GPU. ")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Data from vocab folder', type=str, required=True)
    parser.add_argument('--logdir', help='place to store things.', type=str, required=True)
    parser.add_argument('--ct', help='continue training for longer', type=bool, default=False)
    args = parser.parse_args()

    path = args.logdir
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed. Maybe it already exists? I will overwrite :)" % path)
    else:
        print("Successfully created the directory %s " % path)

    main(args)
