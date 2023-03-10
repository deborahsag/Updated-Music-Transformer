import argparse
import os
import pickle

import numpy as np
from scipy.stats import entropy

import pretty_midi


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_dir", type=str, help="Folder of midi files to be evaluated.")

    args = parser.parse_args()

    directory = args.midi_dir

    entropies = []

    print("Pitch class histograms entropies:")
    print()

    for root, dirs, files in os.walk(directory):
        for midi_name in files:
            midi_path = os.path.join(root, midi_name)

            if os.path.isfile(midi_path):
                ext = os.path.splitext(midi_path)[-1].lower()

                if ext == ".mid":
                    midi_data = pretty_midi.PrettyMIDI(midi_path)
                    hist = midi_data.get_pitch_class_histogram()

                    e = entropy(hist)
                    entropies.append(e)

                    print(f"{midi_name} \t entropy: {e}")

    ent = np.array(entropies)
    ent_mean = np.mean(ent)

    print()
    print(f"Mean of entropies: {ent_mean}")


if __name__ == "__main__":
    main()
