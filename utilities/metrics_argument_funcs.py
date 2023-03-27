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

    parser.add_argument("-midi_root", type=str, default="dataset/data_maestro", help="Directory containing test MIDI files.")
    parser.add_argument("-model_weights", type=str, default="maestro_models/new_notation.pickle", help="Pickled model weights file saved with torch.save and model.state_dict()")  # Change default
    parser.add_argument("-prompt_length", type=int, default=128, help="Length of prompt sequence in tokens.")
    parser.add_argument("-continuation_length", type=int, default=128, help="Length of continuation sequence in tokens.")                                                            # Change default
    parser.add_argument("-num_continuations", type=int, default=4, help="Number of continuations to test the model's continuation prediction (including real continuation).")
    parser.add_argument("-num_tests", type=int, default=1, help="Number of prediction tests to be run.")
    parser.add_argument("-beam", type=int, default=0, help="Beam search k. 0 for random probability sample and 1 for greedy")

    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("--rpr", action="store_true", default=True, help="Use a modified Transformer for Relative Position Representations")

    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")     # Change default for training on GPU
    parser.add_argument("--new_notation", action="store_true", default=True, help="Uses new notation based on note duration")

    return parser.parse_args()


def print_mirex_args(args):
    """
    ----------
    Author: Deborah
    ----------
    Prints arguments for MIREX-like Continuation Prediction Challenge
    ----------
    """

    print(SEPERATOR)
    print("Generate arguments")
    print("midi_root:", args.midi_root)
    print("model_weights:", args.model_weights)
    print("prompt_length:", args.prompt_length)
    print("continuation_length:", args.continuation_length)
    print("num_continuations:", args.num_continuations)
    print("num_tests:", args.num_tests)
    print("beam:", args.beam)
    print("")
    print("Model arguments")
    print("n_layers:", args.n_layers)
    print("num_heads:", args.num_heads)
    print("d_model:", args.d_model)
    print("dim_feedforward:", args.dim_feedforward)
    print("max_sequence:", args.max_sequence)
    print("rpr:", args.rpr)
    print("")
    print("force_cpu:", args.force_cpu)
    print("new_notation:", args.new_notation)
    print(SEPERATOR)
    print("")
