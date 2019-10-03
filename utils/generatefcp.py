from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from multiprocessing import Pool
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tqdm import tqdm
import time

def compute_finerprint(mol, raidus=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol,raidus,useFeatures=True) #equivlanet to ECFP4

def getScaffold(mol):
    try:
        return Chem.MolToSmiles(GetScaffoldForMol(mol))
    except:
        return None

def get_mol(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol
    except:
        return None

def procress1(x):
    smi,name =x
    mol = get_mol(smi)
    if mol is None:
        return (None, name)
    fp = compute_finerprint(mol)
    scaff = getScaffold(mol)
    return (name, smi, scaff, fp.ToBitString())

def main(args):
    print("Computing ECFP Fingerprints and Scaffolds for", args.i, "using", args.n, "jobs")

    infile = open(args.i, 'r').readlines()
    infile = (map(lambda x : x.strip().split(' '), infile))

    p = Pool(args.n)
    res = p.imap_unordered(procress1, infile, chunksize=100)

    with open(args.o, 'w') as f:
        for (name, smi, scaff, fp) in res:
            f.write(name)
            f.write(',')
            f.write(smi)
            f.write(',')
            f.write(scaff)
            f.write(',')
            f.write(fp)
            f.write('\n')


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input smi file with smi name layout', required=True, type=str)
    parser.add_argument('-o', help='out', type=str, default='out.npy')
    parser.add_argument('-n', help='n_jobs', type=int, default=1, required=False)
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print(end - start)