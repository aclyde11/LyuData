import argparse
import os
from tqdm import tqdm

START_CHAR = '%'
END_CHAR = '^'

def get_vocab_from_file(fname):
    vocab = open(fname, 'r').readlines()
    vocab_ = list(filter(lambda x : len(x) != 0, map(lambda x : x.strip(), vocab)))
    vocab_c2i = {k : v for v, k in enumerate(vocab_)}
    vocab_i2c = {v : k for v, k in enumerate(vocab_)}

    def i2c(i):
        return vocab_i2c[i]
    def c2i(c):
        return vocab_c2i[c]

    return vocab, c2i, i2c

#
# Generates a random permutations of smile strings.
#
def randomSmiles(smi, max_len=150, attempts=100):
    from rdkit import Chem
    import random
    def randomSmiles_(m1):
        m1.SetProp("_canonicalRankingNumbers", "True")
        idxs = list(range(0, m1.GetNumAtoms()))
        random.shuffle(idxs)
        for i, v in enumerate(idxs):
            m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
        return Chem.MolToSmiles(m1)

    m1 = Chem.MolFromSmiles(smi)
    if m1 is None:
        return None
    if m1 is not None and attempts == 1:
        return [smi]

    s = set()
    for i in range(attempts):
      smiles = randomSmiles_(m1)
      s.add(smiles)
    s = list(filter(lambda x : len(x) < max_len, list(s)))

    if len(s) > 1:
        return s
    else:
        return [smi]


def main(args):
    if args.permute_smiles != 0:
        try:
            randomSmiles('CNOPc1ccccc1', 10)
        except:
            print("Must set --permute_smiles to 0, cannot import RdKit. Smiles validity not being checked either.")

    #first step generate vocab.
    if not args.r:
        count = 0
        vocab = set()
        vocab.update([START_CHAR, END_CHAR])
        print(vocab)
        with open(args.i, 'r') as f:
            next(f) #skip header
            for line in f:
                smis = line.split(",")
                _, smi = float(smis[2].strip()), smis[1].strip()
                count += 1
                if len(smi) > args.maxlen - 2:
                    continue
                vocab.update(smi)

        vocab = list(vocab)
        with open(args.o + '/vocab.txt', 'w') as f:
            for v in vocab:
                f.write(v + '\n')


        print("Read ", count,"smiles.")
        print("Vocab length: ", len(vocab), "Max len: ", args.maxlen)

        #seconnd step is to make data:
        count_=count
    else:
        count_ = 0
    count = 0
    _, c2i, _ = get_vocab_from_file(args.o + '/vocab.txt')
    if args.permute_smiles != -1:
        with open(args.i, 'r') as f:
            with open(args.o + '/out.txt', 'w') as o:
                with open(args.o + '/out_y.txt', 'w') as oy:
                    with open(args.o + '/out_names.txt', 'w') as on:
                        next(f) # skip header
                        for line in tqdm(f, total=count_):
                            smis = line.split(",")
                            name, y, smis = str(smis[0].strip()), float(smis[2].strip()), smis[1].strip()
                            if args.permute_smiles != 0:
                                smis = randomSmiles(smis, max_len=args.maxlen, attempts=args.permute_smiles)
                                if smis is None:
                                    continue
                            else:
                                smis = [smis]
                            for smi in smis:
                                if len(smi) > args.maxlen - 2:
                                    continue

                                # convert to index number
                                try:
                                    i = list(map(lambda x : str(c2i(x)), smi))
                                    on.write(name + "\n")
                                    oy.write(str(y) + '\n')
                                    o.write(','.join(i) + '\n')
                                    count += 1
                                except:
                                    print("key error did not print.")
                                    continue
        print("Output",count,"smiles.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='input file with single smile per line')
    parser.add_argument('-o', type=str, required=True, help='output directory where preprocressed data and vocab will go')
    parser.add_argument('--maxlen', type=int, required=True, help='max length for smile strings to be')
    parser.add_argument('--permute_smiles', type=int, help='generates permutations of smiles', default=0)
    parser.add_argument('-r', action='store_true')
    args = parser.parse_args()

    print(args)
    path = args.o
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed. Maybe it already exists? I will overwrite :)" % path)
    else:
        print("Successfully created the directory %s " % path)
    main(args)