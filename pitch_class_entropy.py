import argparse
import os

import math
import numpy as np
from scipy.stats import entropy, norm

import pretty_midi

from midi_processor.processor import encode_midi


def compute_entropy_confidence_interval(hist, num_tokens):
    # https://math.stackexchange.com/questions/1259843/confidence-interval-of-information-entropy
    N = num_tokens
    var = np.sum(((np.log2(hist) + 1) ** 2) * (hist * (1 - hist) / N))
    scale = var
    return norm.interval(confidence=0.95, loc=0, scale=scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-midi_root", type=str, help="Folder of midi files to be evaluated.")
    args = parser.parse_args()

    directory = args.midi_root

    original = {}
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
                ent = entropy(hist, base=2)
                num_tokens = len(encode_midi(midi_path))
                interval = compute_entropy_confidence_interval(hist, num_tokens)

                if midi_name.startswith("original"):
                    original[midi_name] = []
                    original[midi_name].append(hist)
                    original[midi_name].append(ent)
                    original[midi_name].append(interval)

                if midi_name.startswith("rand"):
                    generated[midi_name] = []
                    generated[midi_name].append(hist)
                    generated[midi_name].append(ent)
                    generated[midi_name].append(interval)

    print("-----Original pieces-----")

    ori_hist = []
    ori_ent = []
    for name, info in sorted(original.items()):
        hist = info[0]
        ent = info[1]
        interval = info[2]

        ori_hist.append(hist)
        ori_ent.append(ent)

        print(f"{name}:")
        print(f"\tentropy:\t{ent}")
        print(f"\tconfidence interval\t: {interval}")

    ori_ent_mean = np.mean(np.array(ori_ent))
    print(f"Mean entropy: {ori_ent_mean}")

    # ori_hist_mean = np.mean(ori_hist)
    # ori_interval = compute_entropy_confidence_interval(ori_hist_mean, ?)

    print()

    print("-----Generated pieces-----")
    gen_hist = []
    gen_ent = []
    for name, info in sorted(generated.items()):
        hist = info[0]
        ent = info[1]
        interval = info[2]

        gen_hist.append(hist)
        gen_ent.append(ent)

        print(f"{name}:")
        print(f"\tentropy:\t{ent}")
        print(f"\tconfidence interval\t: {interval}")

    gen_ent_mean = np.mean(np.array(gen_ent))
    print(f"Mean entropy: {gen_ent_mean}")

    # gen_hist_mean = np.mean(ori_hist)
    # gen_interval = compute_entropy_confidence_interval(ori_hist_mean, ?)

    print()

    print("-----Mean entropy comparison-----")
    print(f"Original: \t {ori_ent_mean}")
    print(f"Generated: \t {gen_ent_mean}")


if __name__ == "__main__":
    main()
