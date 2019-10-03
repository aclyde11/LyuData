import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn
import torch.utils.data
from tqdm import tqdm

from model.model import DockRegressor
from model.vocab import get_vocab_from_file, START_CHAR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train import getconfig,

def get_input_data(fname, fname2, c2i):
    lines = open(fname, 'r').readlines()
    lines = list(map(lambda x: x.split(','), (filter(lambda x: len(x) != 0, map(lambda x: x.strip(), lines)))))
    lines1 = [torch.from_numpy(np.array([c2i(START_CHAR)] + list(map(lambda x: int(x), y)), dtype=np.int64)) for y in
              lines]

    lines = open(fname2, 'r')
    ys = list(map(lambda x: torch.tensor(float(x.strip())), lines))
    assert (len(ys) == len(lines1))
    print("Read", len(lines1), "SMILES.")

    return lines1, ys, list(range(len(ys)))


def mycollate(x):
    x_batches = []
    y_batchese = []
    n_batches = []
    for i in x:
        x_batches.append(i[0])
        y_batchese.append(i[1])
        n_batches.append(i[2])
    y_batchese = torch.stack(y_batchese, dim=0).float()
    return x_batches, y_batchese, n_batches


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, s, e,n):
        self.s = s
        self.e = e
        self.n = n
        assert (len(self.s) == len(self.e))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, item):
        return self.s[item], self.e[item], self.n[item]


def get_metrics(y_pred, y_int_pred, y, bin=0.25):
    from sklearn import metrics

    y_int_pred = (y_int_pred >= 0.5).astype(np.int32)
    y_int = (y <= bin).astype(np.int32)

    cm = metrics.confusion_matrix(y_int, y_int_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    met = {
        'r2_score': metrics.r2_score(y, y_pred),
        'mse': metrics.mean_squared_error(y, y_pred),
        'mae': metrics.mean_absolute_error(y, y_pred),
        'median error': metrics.median_absolute_error(y, y_pred),

        'balacc' : metrics.balanced_accuracy_score(y_int, y_int_pred),
        'mcc' : metrics.matthews_corrcoef(y_int, y_int_pred),
        'recall' : metrics.recall_score(y_int, y_int_pred),
        'tpr' : cm[0,0],
        'tnr' : cm[1,1]
    }

    return met


def test_model(model, optimizer, dataloader, config):
    model.eval()
    with torch.no_grad():
        ys, ys_hat = [], []
        ys_int = []
        for i, (y, y_hat,name) in tqdm(enumerate(dataloader)):
            y_hat = y_hat.float().to(device)
            y = [x.to(device) for x in y]

            pred1, pred2, _ = model(y)

            ys.append(pred1.cpu())
            ys_int.append((torch.sigmoid(pred2.cpu())))
            ys_hat.append(y_hat.cpu())
        ys = torch.cat(ys).flatten().numpy()
        ys_int = torch.cat(ys_int).flatten().numpy()
        ys_hat = torch.cat(ys_hat).flatten().numpy()
        print("Test Numpy Suite")
        print(get_metrics(ys, ys_int, ys_hat))
        np.savez_compressed(config['testdir']+"/preds.npz", y=ys, y_int=ys_int, y_true=ys_hat)


def main(args):
    config = getconfig(args)
    config['testdir'] = args.testdir
    print("loading data.")
    vocab, c2i, i2c = get_vocab_from_file(args.i + "/vocab.txt")
    print("Vocab size is", len(vocab))
    s_t, e_t, n_t = get_input_data(args.testdir + "/out.txt", args.testdir + "/out_y.txt", c2i)
    test_data = ToyDataset(s_t, e_t, n_t)
    print("Done.")

    ## make data generator
    test_dataloader = torch.utils.data.DataLoader(test_data, pin_memory=True, batch_size=config['batch_size'],
                                                  collate_fn=mycollate, num_workers=0, shuffle=False)

    model = DockRegressor(config['vocab_size'], config['emb_size'], max_len=config['max_len']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epoch_start = 0
    if args.ct:
        print("Continuing from save.")
        pt = torch.load(args.logdir + "/autosave.model.pt")
        model.load_state_dict(pt['state_dict'])
        optimizer.load_state_dict(pt['optim_state_dict'])

    test_model(model, optimizer, test_dataloader, config)



if __name__ == '__main__':
    print("Note: This script is very picky. This will only run on a GPU. ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='place to store things.', type=str, required=True)
    parser.add_argument('--testdir', type=str, required=True)
    parser.add_argument('--ct', help='continue training for longer', action='store_true')

    args = parser.parse_args()
    path = args.logdir
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed. Maybe it already exists? I will overwrite :)" % path)
    else:
        print("Successfully created the directory %s " % path)

    main(args)
