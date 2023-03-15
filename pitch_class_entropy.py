import argparse
import os

import numpy as np
from scipy.stats import entropy

import pretty_midi


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_dir", type=str, help="Folder of midi files to be evaluated.")

    args = parser.parse_args()

    directory = args.midi_dir

    original = {}
    primer = {}
    generated = {}

    print("Pitch Class Histogram Entropy")
    print(f"MIDI directory: {directory}")
    print()

    for root, dirs, files in os.walk(directory):
        for midi_name in files:
            midi_path = os.path.join(root, midi_name)
            ext = os.path.splitext(midi_path)[-1].lower()

            if ext == ".mid":
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                hist = midi_data.get_pitch_class_histogram()
                e = entropy(hist, base=2)

                if midi_name.startswith("original"):
                    original[midi_name] = e

                if midi_name.startswith("primer"):
                    primer[midi_name] = e

                if midi_name.startswith("rand"):
                    generated[midi_name] = e

    print("-----Original files-----")
    for name, ent in sorted(original.items()):
        print(f"{name} \t entropy: {ent}")

    ori_ent = np.array(list(original.values()))
    ori_mean = np.mean(ori_ent)
    print(f"Mean entropy: {ori_mean}")
    print()

    print("-----Primer files-----")
    for name, ent in sorted(primer.items()):
        print(f"{name} \t entropy: {ent}")

    primer_ent = np.array(list(primer.values()))
    primer_mean = np.mean(primer_ent)
    print(f"Mean entropy: {primer_mean}")
    print()

    print("-----Generated files-----")
    for name, ent in sorted(generated.items()):
        print(f"{name} \t entropy: {ent}")

    gen_ent = np.array(list(generated.values()))
    gen_mean = np.mean(gen_ent)
    print(f"Mean entropy: {gen_mean}")
    print()

    print("-----Mean entropy comparison-----")
    print(f"Original: \t {ori_mean}")
    print(f"Primer: \t {primer_mean}")
    print(f"Generated: \t {gen_mean}")


if __name__ == "__main__":
    main()
