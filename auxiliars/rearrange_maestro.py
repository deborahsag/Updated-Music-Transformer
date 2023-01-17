import shutil
import pandas as pd

csv = pd.read_csv("/mnt/e/Datasets/new-maestro/maestro-v3.0.0.csv")
csv = csv[["midi_filename", "split"]]

PATH = "/mnt/e/Datasets/new-maestro/"

for _, row in csv.iterrows():
    shutil.copyfile(f"{PATH}{row['midi_filename']}", f"{PATH}{row['split']}/{row['midi_filename'][5:]}")
