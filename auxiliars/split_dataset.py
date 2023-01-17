from glob import glob
import shutil
import os
import random

PATH = "/mnt/f/Datasets/lmd-full-pianified-pickles"
midis = glob(f"{PATH}/**/*.[mid|MID]*", recursive=True)
print(f"Splitting dataset with {len(midis)} midis")
random.shuffle(midis)

train, test, validation = midis[:-len(midis)//5], midis[-len(midis)//5:-len(midis)//10], midis[-len(midis)//10:]

if not os.path.exists(f"{PATH}/train/"):
    os.makedirs(f"{PATH}/train/")
if not os.path.exists(f"{PATH}/test/"):
    os.makedirs(f"{PATH}/test/")
if not os.path.exists(f"{PATH}/validation/"):
    os.makedirs(f"{PATH}/validation/")

for mid in train:
    shutil.move(mid, f"{PATH}/train/{mid[len(PATH)+3:]}")

for mid in test:
    shutil.move(mid, f"{PATH}/test/{mid[len(PATH)+3:]}")

for mid in validation:
    shutil.move(mid, f"{PATH}/validation/{mid[len(PATH)+3:]}")
