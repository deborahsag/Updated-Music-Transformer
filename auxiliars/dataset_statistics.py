from glob import glob
import pickle
from tqdm import tqdm
from statistics import mean

PATH = "/mnt/f/Datasets/new-maestro-augmented-pickles-pmp"

training_examples = [p for p in glob(f"{PATH}/train/*.pickle") if "/Aug-" not in p and "/Pmp-" not in p]
test_examples = [p for p in glob(f"{PATH}/test/*.pickle") if "/Aug-" not in p and "/Pmp-" not in p]
validation_examples = [p for p in glob(f"{PATH}/validation/*.pickle") if "/Aug-" not in p and "/Pmp-" not in p]

lens, uniques = [], []
for pick in tqdm(training_examples + test_examples + validation_examples):
    tokens = pickle.load(open(pick, "rb"))
    lens.append(len(tokens))
    uniques.append(len(set(tokens)))

print(f"Total examples:      {len(training_examples) + len(test_examples) + len(validation_examples)}")
print(f"Training examples:   {len(training_examples)}")
print(f"Test examples:       {len(test_examples)}")
print(f"Validation examples: {len(validation_examples)}")  
print(f"Average length:      {mean(lens)}")
print(f"Average unique:      {mean(uniques)}")
