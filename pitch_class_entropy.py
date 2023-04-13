import os

import numpy as np
from scipy.stats import bootstrap, entropy

import pretty_midi

from utilities.metrics_argument_funcs import parse_pitch_class_entropy_args, print_pitch_class_entropy_args


def compute_mean_entropy(pieces):
    """
    Computes mean entropy value among given pieces.
    Receives dictionary of pieces and its entropy values.
    Returns float value.
    """
    entropies = np.array(list(pieces.values()))
    return np.mean(entropies)


def compute_confidence_interval(pieces, statistic=np.mean, n_resamples=9999, confidence_level=0.95, method='BCa'):
    """
    Computes confidence interval of the mean value using the bootstrap method.
    Receives a dictionary of pieces and its entropy values.
    Returns a confidence interval.
    """
    entropies = list(pieces.values())
    data = (entropies,)
    res = bootstrap(data, statistic=statistic, n_resamples=n_resamples, confidence_level=confidence_level, method=method)
    return res.confidence_interval


def print_piece_entropy(pieces):
    """
    Receives dictionary of pieces and its entropy values.
    Prints entropy value of every piece.
    """
    for name, ent in sorted(pieces.items()):
        print(f"{name}\tentropy: {ent}")
    return


def main():
    args = parse_pitch_class_entropy_args()
    print_pitch_class_entropy_args(args)

    original = {}
    generated = {}
    for root, dirs, files in os.walk(args.midi_root):   # get entropy value for each piece in directory
        for midi_name in files:
            midi_path = os.path.join(root, midi_name)
            ext = os.path.splitext(midi_path)[-1].lower()

            if ext == ".mid":
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                hist = midi_data.get_pitch_class_histogram()
                p = np.array(hist) + 1e-6
                p = p / p.sum()
                ent = entropy(p, base=2)

                if midi_name.startswith("original"):
                    original[midi_name] = ent

                if midi_name.startswith("rand") or midi_name.startswith("beam"):
                    generated[midi_name] = ent

    if args.print_each:
        print("Original pieces:")
        print_piece_entropy(original)
        print("---------------")
        print("")
        print("Generated pieces:")
        print_piece_entropy(generated)
        print("---------------")
        print("")

    ori_mean = compute_mean_entropy(original)
    print(f"Mean Entropy of Original Pieces: {ori_mean}")
    if len(original) > 1:
        ori_interval = compute_confidence_interval(original, n_resamples=args.n_resamples,
                                                   confidence_level=args.confidence_level)
        print(ori_interval)
    print("")

    gen_mean = compute_mean_entropy(generated)
    print(f"Mean Entropy of Generated Pieces: {gen_mean}")
    if len(generated) > 1:
        gen_interval = compute_confidence_interval(generated, n_resamples=args.n_resamples,
                                                   confidence_level=args.confidence_level)
        print(gen_interval)
    print("")


if __name__ == "__main__":
    main()
