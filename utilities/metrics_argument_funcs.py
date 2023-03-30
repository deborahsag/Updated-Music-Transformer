import argparse

from utilities.constants import *

def parse_mirex_args():
    """
    ----------
    Author: Deborah
    ----------
    Argparse arguments for MIREX-like Continuation Prediction Challenge
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_root", type=str, help="Directory containing test MIDI files.")
    parser.add_argument("-model_weights", type=str, help="Pickled model weights file saved with torch.save and model.state_dict()")
    parser.add_argument("-prompt_length", type=int, default=128, help="Length of prompt sequence in tokens.")
    parser.add_argument("-continuation_length", type=int, default=128, help="Length of continuation sequence in tokens.")                                                            # Change default
    parser.add_argument("-num_continuations", type=int, default=4, help="Number of continuations to test the model's continuation prediction (including real continuation).")
    parser.add_argument("-num_tests", type=int, default=1, help="Number of prediction tests to be run.")
    parser.add_argument("-seed", type=int, default=None, help="Seed for the random prompt and continuations selector")
    parser.add_argument("--new_notation", action="store_true", help="Uses new notation based on note duration")

    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-beam", type=int, default=0, help="Beam search k. 0 for random probability sample and 1 for greedy")
    parser.add_argument("--rpr", action="store_true", default=True, help="Use a modified Transformer for Relative Position Representations")

    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")     # Change default for training on GPU

    return parser.parse_args()


def print_mirex_args(args):
    """
    ----------
    Author: Deborah
    ----------
    Prints arguments for MIREX-like Continuation Prediction Challenge
    ----------
    """

    print(SEPARATOR)
    print("MIREX-like Continuation Prediction Challenge")
    print(SEPARATOR)
    print("midi_root:", args.midi_root)
    print("model_weights:", args.model_weights)
    print("prompt_length:", args.prompt_length)
    print("continuation_length:", args.continuation_length)
    print("num_continuations:", args.num_continuations)
    print("num_tests:", args.num_tests)
    print("seed:", args.seed)
    print("")
    print("new_notation:", args.new_notation)
    print("n_layers:", args.n_layers)
    print("num_heads:", args.num_heads)
    print("d_model:", args.d_model)
    print("dim_feedforward:", args.dim_feedforward)
    print("rpr:", args.rpr)
    print("max_sequence:", args.max_sequence)
    print("beam:", args.beam)
    print("")
    print("force_cpu:", args.force_cpu)
    print(SEPARATOR)
    print("")


def parse_pitch_class_entropy_args():
    """
    ----------
    Author: Deborah
    ----------
    Argparse arguments for Pitch Class Histogram Entropy
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
    Prints arguments for MIREX-like Continuation Prediction Challenge
    ----------
    """

    print(SEPARATOR)
    print("Pitch Class Histogram Entropy")
    print(SEPARATOR)
    print(f"-midi_root: {args.midi_root}")
    print(f"n_resamples: {args.n_resamples}")
    print(f"confidence_level: {args.confidence_level}")
    print(SEPARATOR)
    print("")
