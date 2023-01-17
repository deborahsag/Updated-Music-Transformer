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
TIME_WARP = [0.95, 0.975, 1.025, 1.05] # We handle 1 separately because it's the original file
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
    savepath = f"{filepath}/Pmp-{filename}-1.pickle"

    if os.path.exists(savepath):
        return

    tokens = pickle.load(open(pick, "rb"))
    if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
        print(f"Will delete {filename} because it has {len(tokens)} tokens.")
        os.remove(pick)
        return

    generate_pmp(tokens, savepath)


# We save some calculations with the fact that files with the same amount of time warp have the same pmp
def pmp_train(pick):
    dir_split = pick.rfind("/") + 1
    filepath, filename = pick[:dir_split], pick[dir_split:-7]
    
    tokens = pickle.load(open(pick, "rb"))
    if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
        print(f"Will delete {filename} and its augmentations because it has {len(tokens)} tokens.")
        os.remove(pick)
        for invalid_file in glob(f"{filepath}/Aug-{filename}*"):
            os.remove(invalid_file)
        return

    savepath = f"{filepath}/Pmp-{filename}-1.pickle"
    if not os.path.exists(savepath):
        generate_pmp(tokens, savepath)

    for time in TIME_WARP:
        savepath = f"{filepath}/Pmp-{filename}-{time}.pickle"
        if not os.path.exists(savepath):
            tokens = pickle.load(open(f"{filepath}/Aug-{filename}-0-{time}.pickle", "rb"))
            generate_pmp(tokens, savepath)


def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]


if __name__ == "__main__":
    PARALLELIZE, MAX_WORKERS = True, 6

    train_pickles = glob(f"{PATH}/train/*.pickle")
    train_pickles = [p for p in train_pickles if "/Pmp-" not in p and "/Aug-" not in p]

    test_valid_pickles = glob(f"{PATH}/test/*.pickle") + glob(f"{PATH}/validation/*.pickle")
    test_valid_pickles = [p for p in test_valid_pickles if "/Pmp-" not in p]
    
    print("Sorting...")
    train_pickles, test_valid_pickles = sorted(train_pickles, key=lambda x: os.stat(x).st_size), sorted(test_valid_pickles, key=lambda x: os.stat(x).st_size)

    print("Starting with test and valid")
    if PARALLELIZE:
        process_map(pmp_nontrain, test_valid_pickles, max_workers=MAX_WORKERS, chunksize=1)
    else:
        for pick in tqdm(test_valid_pickles):
            pmp_nontrain(pick)

    print("Moving on to train")
    if PARALLELIZE:
        process_map(pmp_train, train_pickles, max_workers=MAX_WORKERS, chunksize=1)
    else:
        for pick in tqdm(train_pickles):
            pmp_train(pick)

    print("Done.")
