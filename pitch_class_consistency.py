import os

import numpy as np
from scipy.stats import bootstrap, entropy

import pretty_midi

from utilities.metrics_argument_funcs import parse_pitch_class_consistency_args, print_pitch_class_consistency_args


def get_midi_partitions(midi, num_partitions):
    """
    Partitions a PrettyMIDI object given the number of partitions. Separation is based in seconds.
    Returns a list of PrettyMIDI objects.
    """
    midi_partitions = []
    for i in range(num_partitions):
        midi_partitions.append(pretty_midi.PrettyMIDI())

    for instrument in midi.instruments:
        instrument_partitions = get_instrument_partitions(instrument, num_partitions)

        for i in range(num_partitions):
            midi_partitions[i].instruments.append(instrument_partitions[i])

    return midi_partitions


def get_instrument_partitions(instrument, num_partitions):
    """
    Partitions a PrettyMIDI object given the number of partitions. Separation is based in seconds.
    Returns a list of Instrument objects.
    """
    end_time = instrument.get_end_time()
    partition_duration = end_time // num_partitions
    current_limit = partition_duration  # defines end of current partition in seconds

    program = instrument.program
    notes = instrument.notes
    n = 0   # run through entire instrument track note for note

    instrument_partitions = []     # list of separate partitions of the same track. Length = num_partitions
    for p in range(num_partitions):
        part = pretty_midi.Instrument(program=program)    # create a new object for new partition
        partition_notes = []
        while n < len(notes):
            if notes[n].start > current_limit:
                current_limit += partition_duration
                break   # stop populating current partition and jump to the next
            else:
                partition_notes.append(notes[n])
                n += 1
        part.notes = partition_notes    # set current Instrument object with the list of Notes
        instrument_partitions.append(part)

    return instrument_partitions


def get_partitions_histograms(partitions):
    """
    Get the pitch class histogram of each PrettyMIDI object in a list.
    Returns a list of histograms.
    """
    histograms = []
    for part in partitions:
        histograms.append(part.get_pitch_class_histogram())
    return histograms


def compute_total_variation_distance(histogram_1, histogram_2):
    """
    Compute the Total Variation Distance between two histograms.
    Returns a float value.
    """
    hist_1 = np.array(histogram_1)
    hist_2 = np.array(histogram_2)
    hist_res = np.absolute(hist_1 - hist_2)
    return 0.5 * np.sum(hist_res)


def compute_total_dkl(histogram_1, histogram_2):
    """
    Compute the Kullback–Leibler Distance between two histograms.
    Returns a float value.
    """
    hist_1 = np.array(histogram_1)
    hist_2 = np.array(histogram_2)
    return entropy(hist_1, hist_2, base=2)


def get_mean_consistency(pieces):
    """
    Gets the mean value of consistency among a list of pieces.
    Receives a dictionary of pieces and its consistency values.
    Returns a float value.
    """
    consistencies = []
    for name, cons in pieces.items():
        consistencies.append(cons)
    return np.mean(np.array(consistencies))


def compute_confidence_interval(pieces, statistic=np.mean, n_resamples=9999, confidence_level=0.95, method='BCa'):
    """
    Computes confidence interval of the mean value using the bootstrap method.
    Receives a dictionary of pieces and its consistency values.
    Returns a confidence interval.
    """
    values = list(pieces.values())
    data = (values,)
    res = bootstrap(data, statistic=statistic, n_resamples=n_resamples, confidence_level=confidence_level, method=method)
    return res.confidence_interval


def print_piece_consistency(pieces):
    """
    Receives a dictionary of pieces and its values of consistency.
    Prints consistency values of every piece.
    """
    for name, cons in sorted(pieces.items()):
        print(f"{name}\tconsistency: {cons}")
    return


def main():
    args = parse_pitch_class_consistency_args()
    print_pitch_class_consistency_args(args)

    original = {}
    generated = {}
    for root, dirs, files in os.walk(args.midi_root):     # walk through midi root
        for midi_name in files:
            midi_path = os.path.join(root, midi_name)
            ext = os.path.splitext(midi_path)[-1].lower()

            if ext == ".mid":
                midi_data = pretty_midi.PrettyMIDI(midi_path)   # get midi data and divide into partitions
                partitions = get_midi_partitions(midi_data, args.num_partitions)
                histograms = get_partitions_histograms(partitions)  # get pitch class histograms for each partition

                distances = []
                for i in range(len(histograms) - 1):    # calculate the distance between histograms pair by pair
                    p = np.array(histograms[i]) + 1e-6
                    q = np.array(histograms[i+1]) + 1e-6
                    p = p/p.sum()
                    q = q/q.sum()
                    # dist = compute_total_variation_distance(p, q)
                    dist = compute_total_dkl(p, q)
                    distances.append(dist)
                consistency = np.mean(np.array(distances))  # consistency is given by the mean value of distances within a piece

                if midi_name.startswith("original"):
                    original[midi_name] = consistency

                if midi_name.startswith("rand") or midi_name.startswith("beam"):
                    generated[midi_name] = consistency

    if args.print_each:
        print("Original pieces:")
        print_piece_consistency(original)
        print("---------------")
        print("")
        print("Generated pieces:")
        print_piece_consistency(generated)
        print("---------------")
        print("")

    ori_mean = get_mean_consistency(original)
    print(f"Mean Consistency of Original Pieces: {ori_mean}")
    if len(original) > 1:
        ori_interval = compute_confidence_interval(original, n_resamples=args.n_resamples,
                                                   confidence_level=args.confidence_level)
        print(ori_interval)
    print("")

    gen_mean = get_mean_consistency(generated)
    print(f"Mean Consistency of Generated Pieces: {gen_mean}")
    if len(generated) > 1:
        gen_interval = compute_confidence_interval(generated, n_resamples=args.n_resamples,
                                                   confidence_level=args.confidence_level)
        print(gen_interval)
    print("")

    return


if __name__ == "__main__":
    main()
