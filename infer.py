import argparse
import os

import torch
import torch.nn.utils.rnn
import torch.utils.data

from model.model import DockRegressor
from model.vocab import get_vocab_from_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train import getconfig, get_input_data,mycollate, ToyDataset, test_model


def main(args):
    config = getconfig(args)
    config['testdir'] = args.testdir
    print("loading data.")
    vocab, c2i, i2c = get_vocab_from_file(args.testdir + "/vocab.txt")
    print("Vocab size is", len(vocab))
    s_t, e_t, n_t = get_input_data(args.testdir + "/out.txt", args.testdir + "/out_y.txt", args.testdir + "/out_names.txt", c2i)
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
