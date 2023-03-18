import argparse
import os

import numpy as np
from scipy.stats import bootstrap, entropy

import pretty_midi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-midi_root", type=str, help="Folder of midi files to be evaluated.")
    parser.add_argument("--print_each", action="store_true", default=False, help="Print entropy value of each piece.")
    parser.add_argument("-n_resamples", type=int, default=9999, help="Number of resamples for the bootstrap method.")
    parser.add_argument("-confidence_level", type=float, default=0.95, help="Confidence level for the confidence interval. Default: 0.95.")
    args = parser.parse_args()

    print("Pitch Class Histogram Entropy")
    print(f"MIDI directory: {args.midi_root}")
    print()

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

                if midi_name.startswith("rand"):
                    generated[midi_name] = ent

    print("-----Original pieces-----")
    if args.print_each:
        for name, ent in sorted(original.items()):
            print(f"{name}\tentropy: {ent}")

    # compute mean entropy
    ori_ent = np.array(list(original.values()))
    ori_ent_mean = np.mean(ori_ent)
    print(f"Mean entropy:\t{ori_ent_mean}")

    # bootstrap confidence interval
    if len(ori_ent) > 1:
        data = (ori_ent,)
        ori_res = bootstrap(data, np.mean, n_resamples=args.n_resamples, confidence_level=args.confidence_level, method='BCa')
        print(f"Mean entropy confidence interval:\t{ori_res.confidence_interval}")

    print()

    print("-----Generated pieces-----")
    if args.print_each:
        for name, ent in sorted(generated.items()):
            print(f"{name}\tentropy: {ent}")

    # compute mean entropy
    gen_ent = np.array(list(generated.values()))
    gen_ent_mean = np.mean(gen_ent)
    print(f"Mean entropy:\t{gen_ent_mean}")

    # bootstrap confidence interval
    data = (gen_ent,)
    gen_res = bootstrap(data, np.mean, n_resamples=99, confidence_level=0.95, method='BCa')
    print(f"Mean entropy confidence interval:\t{gen_res.confidence_interval}")

    print()

    print("-----Mean entropy comparison-----")
    if len(ori_ent) > 1:
        print(f"Original: \t {ori_ent_mean} \t {ori_res.confidence_interval}")
    else:
        print(f"Original: \t {ori_ent_mean}")
    print(f"Generated: \t {gen_ent_mean} \t {gen_res.confidence_interval}")


if __name__ == "__main__":
    main()
