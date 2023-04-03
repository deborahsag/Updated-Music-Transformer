import argparse
import os

import numpy as np
from scipy.stats import bootstrap, entropy

import pretty_midi

from utilities.metrics_argument_funcs import parse_pitch_class_entropy_args, print_pitch_class_entropy_args


def print_piece_entropy(pieces):
    # pieces: dictionary type
    for name, ent in sorted(pieces.items()):
        print(f"{name}\tentropy: {ent}")
    return


def compute_mean_entropy(pieces):
    # pieces: dictionary type
    entropies = np.array(list(pieces.values()))
    return np.mean(entropies)


def compute_confidence_interval(pieces, statistic=np.mean, n_resamples=9999, confidence_level=0.95, method='BCa'):
    # pieces: dictionary type
    # bootstrap confidence interval
    entropies = list(pieces.values())
    data = (entropies,)
    res = bootstrap(data, statistic=statistic, n_resamples=n_resamples, confidence_level=confidence_level, method=method)
    return res.confidence_interval


def main():
    args = parse_pitch_class_entropy_args()
    print_pitch_class_entropy_args(args)

    # get entropy value for each piece in directory
    original = {}
    generated = {}
    for root, dirs, files in os.walk(args.midi_root):
        for midi_name in files:
            midi_path = os.path.join(root, midi_name)
            ext = os.path.splitext(midi_path)[-1].lower()

            if ext == ".mid":
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                hist = midi_data.get_pitch_class_histogram()
                ent = entropy(hist, base=2)

                if midi_name.startswith("original"):
                    original[midi_name] = ent

                if midi_name.startswith("rand") or midi_name.startswith("beam"):
                    generated[midi_name] = ent

    # compute mean entropy
    ori_mean = compute_mean_entropy(original)

    # bootstrap confidence interval
    if len(original) > 1:
        ori_interval = compute_confidence_interval(original, n_resamples=args.n_resamples,
                                                   confidence_level=args.confidence_level)

    if args.print_each:
        print("Original pieces:")
        print_piece_entropy(original)
        print(f"Mean entropy:\t{ori_mean}")
        print(f"Mean entropy confidence interval:\t{ori_interval}")
        print("---------------")
        print("")

    # compute mean entropy
    gen_mean = compute_mean_entropy(generated)

    # bootstrap confidence interval
    if len(generated) > 1:
        gen_interval = compute_confidence_interval(generated, n_resamples=args.n_resamples,
                                                   confidence_level=args.confidence_level)

    if args.print_each:
        print("Generated pieces:")
        print_piece_entropy(generated)
        print(f"Mean entropy:\t{gen_mean}")
        print(f"Mean entropy confidence interval:\t{gen_interval}")
        print("---------------")
        print("")

    print("Mean entropy comparison:")
    if len(original) > 1:
        print(f"Original: \t {ori_mean} \t {ori_interval}")
    else:
        print(f"Original: \t {ori_mean}")
    if len(generated) > 1:
        print(f"Generated: \t {gen_mean} \t {gen_interval}")
    else:
        print(f"Generated: \t {gen_mean}")
    print("---------------")
    print("")


if __name__ == "__main__":
    main()
