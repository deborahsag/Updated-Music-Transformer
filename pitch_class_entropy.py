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

    for root, dirs, files in os.walk(directory):
        for midi_name in files:
            midi_path = os.path.join(root, midi_name)
            ext = os.path.splitext(midi_path)[-1].lower()

            if ext == ".mid":
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                hist = midi_data.get_pitch_class_histogram()
                e = entropy(hist)

                if midi_name.startswith("original"):
                    original[midi_name] = e

                if midi_name.startswith("primer"):
                    primer[midi_name] = e

                if midi_name.startswith("rand"):
                    generated[midi_name] = e

    for name, ent in original.items():
        print(f"{name} \t entropy: {ent}")

    ori_ent = np.array(list(original.values()))
    ori_mean = np.mean(ori_ent)
    print(f"Mean of the entropies of the original samples: {ori_mean}")
    print()

    for name, ent in primer.items():
        print(f"{name} \t entropy: {ent}")

    primer_ent = np.array(list(primer.values()))
    primer_mean = np.mean(primer_ent)
    print(f"Mean of the entropies of the primer files: {primer_mean}")
    print()

    for name, ent in generated.items():
        print(f"{name} \t entropy: {ent}")

    gen_ent = np.array(list(generated.values()))
    gen_mean = np.mean(gen_ent)
    print(f"Mean of the entropies of the generated samples: {gen_mean}")
    print()

    print("Entropy comparison:")
    print(f"Original: \t {ori_mean}")
    print(f"Primer: \t {primer_mean}")
    print(f"Generated: \t {gen_mean}")


if __name__ == "__main__":
    main()
