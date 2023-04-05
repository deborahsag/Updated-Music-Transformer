import argparse

from utilities.constants import *


def parse_pitch_class_entropy_args():
    """
    ----------
    Author: Deborah
    ----------
    Argparse arguments for Overall Pitch Class Histogram Entropy
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_root", type=str, help="Folder of midi files to be evaluated.")
    parser.add_argument("-n_resamples", type=int, default=9999, help="Number of resamples for the bootstrap method.")
    parser.add_argument("-confidence_level", type=float, default=0.95, help="Confidence level for the confidence interval. Default: 0.95.")

    parser.add_argument("--print_each", action="store_true", default=False, help="Print entropy value of each piece.")

    return parser.parse_args()


def print_pitch_class_entropy_args(args):
    """
    ----------
    Author: Deborah
    ----------
    Prints arguments for Overall Pitch Class Histogram Entropy
    ----------
    """

    print(SEPARATOR)
    print("Overall Pitch Class Histogram Entropy")
    print(SEPARATOR)
    print(f"-midi_root: {args.midi_root}")
    print(f"n_resamples: {args.n_resamples}")
    print(f"confidence_level: {args.confidence_level}")
    print(SEPARATOR)
    print("")


def parse_pitch_class_consistency_args():
    """
        ----------
        Author: Deborah
        ----------
        Argparse arguments for Pitch Class Consistency Entropy
        ----------
        """

    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_root", type=str, help="Folder of midi files to be evaluated.")
    parser.add_argument("-num_partitions", type=int, default=4, help="Number of partitions for pair to pair comparison.")
    parser.add_argument("-n_resamples", type=int, default=9999, help="Number of resamples for the bootstrap method.")
    parser.add_argument("-confidence_level", type=float, default=0.95,
                        help="Confidence level for the confidence interval. Default: 0.95.")

    parser.add_argument("--print_each", action="store_true", default=False, help="Print entropy value of each piece.")

    return parser.parse_args()


def print_pitch_class_consistency_args(args):
    """
    ----------
    Author: Deborah
    ----------
    Prints arguments for Pitch Class Consistency Entropy
    ----------
    """

    print(SEPARATOR)
    print("Overall Pitch Class Histogram Entropy")
    print(SEPARATOR)
    print(f"-midi_root: {args.midi_root}")
    print(f"-num_partitions: {args.num_partitions}")
    print(f"n_resamples: {args.n_resamples}")
    print(f"confidence_level: {args.confidence_level}")
    print(SEPARATOR)
    print("")
