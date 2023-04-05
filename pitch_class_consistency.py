import os
import numpy as np

import pretty_midi

from utilities.metrics_argument_funcs import parse_pitch_class_consistency_args, print_pitch_class_consistency_args


def get_midi_partitions(midi, num_partitions):
    midi_partitions = []
    for i in range(num_partitions):
        midi_partitions.append(pretty_midi.PrettyMIDI())

    for instrument in midi.instruments:
        instrument_partitions = get_instrument_partitions(instrument, num_partitions)

        for i in range(num_partitions):
            midi_partitions[i].instruments.append(instrument_partitions[i])

    return midi_partitions


# to be fixed: some notes are left behind.
def get_instrument_partitions(instrument, num_partitions):
    end_time = instrument.get_end_time()
    partition_duration = end_time // num_partitions
    current_limit = partition_duration  # defines end of current partition in seconds

    program = instrument.program
    notes = instrument.notes
    n = 0   # run through entire instrument track note for note

    instrument_partitions = []     # list of separate parts of the same track. Length = num_partitions
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
    histograms = []
    for part in partitions:
        histograms.append(part.get_pitch_class_histogram())
    return histograms


def compute_total_variation_distance(histogram_1, histogram_2):
    hist_1 = np.array(histogram_1)
    hist_2 = np.array(histogram_2)
    hist_res = np.absolute(hist_1 - hist_2)
    return 0.5 * np.sum(hist_res)


def print_piece_consistency(pieces):
    # pieces: dictionary type
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
                    dist = compute_total_variation_distance(histograms[i], histograms[i+1])
                    distances.append(dist)

                if midi_name.startswith("original"):
                    original[midi_name] = distances

                if midi_name.startswith("rand") or midi_name.startswith("beam"):
                    generated[midi_name] = distances

    if args.print_each:
        print("Original pieces:")
        print_piece_consistency(original)
        print("---------------")
        print("")
        print("Generated pieces:")
        print_piece_consistency(generated)
        print("---------------")
        print("")

    return


if __name__ == "__main__":
    main()
