import os
import numpy as np

import pretty_midi

from utilities.metrics_argument_funcs import parse_pitch_class_consistency_args, print_pitch_class_consistency_args


def get_midi_partitions(midi, num_partitions):
    """
    Only valid for MIDI files with a single instrument. Returns a list of Intrument object type.
    """
    end_time = midi.get_end_time()
    partition_duration = end_time // num_partitions

    current_limit = partition_duration
    notes = midi.instruments[0].notes
    n = 0

    partitions = []
    for p in range(num_partitions):
        part = pretty_midi.Instrument(program=1)
        partition_notes = []
        while n < len(notes):
            if notes[n].start > current_limit:
                current_limit += partition_duration
                break
            else:
                partition_notes.append(notes[n])
                n += 1
        part.notes = partition_notes
        partitions.append(part)

    return partitions


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

                distances = []
                for i in range(args.num_partitions - 1):    # calculate the distance between partitions in pairs
                    hist_1 = partitions[i].get_pitch_class_histogram()
                    hist_2 = partitions[i + 1].get_pitch_class_histogram()

                    distances.append(compute_total_variation_distance(hist_1, hist_2))

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
