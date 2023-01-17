from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matrixprofile as mp
import pickle
import numpy as np
import os
from matrixprofile.exceptions import NoSolutionPossible

PATH = "/mnt/f/Datasets/new-maestro-augmented-pickles-pmp"
SAMPLE_PCT = 0.25
MIN_TOKENS, MAX_TOKENS = 50, 1000000


def generate_pmp(tokens, savepath):
    profile = mp.compute(tokens, sample_pct=SAMPLE_PCT)
    transposed = profile['pmp'].T
    indices = np.argmax((1 < transposed) & (transposed < np.inf), axis=-1)
    indices = normalize(indices, {'actual': {'lower': 0, 'upper': transposed.shape[1]}, 'desired': {'lower': 0, 'upper': 1}})
    assert len(indices) == len(tokens)
    pickle.dump(indices, open(savepath, "wb"))


# In the nontraining folders we don't have to deal with data augmentation
def pmp_nontrain(pick):
    dir_split = pick.rfind("/") + 1
    filepath, filename = pick[:dir_split], pick[dir_split:-7]
    savepath = f"{filepath}/Pmp-{filename}.pickle"

    if os.path.exists(savepath):
        return

    tokens = pickle.load(open(pick, "rb"))
    if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
        print(f"Will delete {filename} because it has {len(tokens)} tokens.")
        os.remove(pick)
        return

    generate_pmp(tokens, savepath)


def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]


if __name__ == "__main__":
    PARALLELIZE, MAX_WORKERS = True, 6
    
    pickles = glob(f"{PATH}/**/*.pickle", recursive=True)
    pickles = [p for p in pickles if "/Pmp-" not in p and "/Aug-" not in p]
    
    print("Sorting...")
    pickles = sorted(train_pickles, key=lambda x: os.stat(x).st_size)
    
    if PARALLELIZE:
        process_map(pmp_nontrain, pickles, max_workers=MAX_WORKERS, chunksize=1)
    else:
        for pick in tqdm(test_valid_pickles):
            pmp_nontrain(pick)

    print("Done.")
