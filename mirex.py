import numpy as np
import random
import sys

from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets

from utilities.metrics_argument_funcs import parse_mirex_args, print_mirex_args

from utilities.constants import *
from utilities.device import get_device


def get_prompt(pieces, prompt_len):
    return pieces[0][:prompt_len]


def get_continuations(pieces, prompt_len):
    return [piece[prompt_len:] for piece in pieces]


def compute_continuation_prob(gen_probs, continuation):
    # Get average probability of token sequences
    probs = []
    for i in range(len(continuation)):
        probs.append(gen_probs[i + 1][continuation[i] - 1])
    return np.mean(np.array(probs))


def main():
    args = parse_mirex_args()
    print_mirex_args(args)

    # Set seed for the random prompt and continuations selector
    SEED = args.seed if args.seed is not None else random.randrange(sys.maxsize)
    print(f"Setting seed to {SEED}")
    random.seed(SEED)

    # Load model
    model = MusicTransformer(new_notation=args.new_notation, n_layers=args.n_layers, num_heads=args.num_heads,
                             d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                             max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')))

    model.training = False

    # Grab test dataset
    target_seq_length = args.prompt_length + args.continuation_length
    _, _, dataset = create_epiano_datasets(args.midi_root, target_seq_length, args.new_notation, random_seq=True)

    score = []
    for i in range(args.num_tests):
        print(f"Computing test number {i + 1} of {args.num_tests}:")

        # Grab random pieces from dataset
        indexes = random.sample(range(len(dataset)), args.num_continuations)
        pieces = [dataset[idx][0] for idx in indexes]

        # Get prompts and continuations (tensor type)
        prompt = get_prompt(pieces, args.prompt_length)
        continuations = get_continuations(pieces, args.prompt_length)

        # Generate token probabilities from prompt
        _, gen_probs = model.generate(prompt, target_seq_length, beam=args.beam)

        # For each continuation, compute the probability of it being chosen
        probs = []
        for cont in continuations:
            probs.append(compute_continuation_prob(gen_probs, cont))

        # See if it matches the real continuation (index 0) and keep score
        choice = np.argmax(np.array(probs))
        if choice == 0:
            print("Correct prediction.")
            score.append(1)
        else:
            print("Wrong prediction.")
            score.append(0)

        print("---------------")

    # Compute final score
    final_score = np.mean(np.array(score))

    print(f"Final MIREX-like Continuation Prediction Challenge Score: {final_score}")
    print("")

    return


if __name__ == "__main__":
    main()
