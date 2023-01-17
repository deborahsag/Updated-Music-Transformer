import json
from shutil import copy

JSON_FILE = "/home/gabrielsouza/Masters/maestro/maestro-v3.0.0.json"
maestro_json = json.load(open(JSON_FILE, "r"))
filenames = list(maestro_json["midi_filename"].values())
split = list(maestro_json["split"].values())

pitch_shift = [-3, -2, -1, 1, 2, 3]
time_shift = [0.95, 0.975, 1.025, 1.05]
augmentations = [f"{pitch}-{time}" for pitch in pitch_shift for time in time_shift]

for f, s in zip(filenames, split):
    destination = f"./maestro-pickles/{s}"
    copy(f"/home/gabrielsouza/Masters/maestro-pickles/{f[5:-5]}.midi.pickle", destination)
    for aug in augmentations:
        copy(f"/home/gabrielsouza/Masters/maestro-pickles/maestro_{f[:4]}_{f[5:-5]}-{aug}.mid.pickle", destination)
print('yo')
