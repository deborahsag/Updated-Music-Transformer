import numpy as np
import random

from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets

from utilities.metrics_argument_funcs import parse_mirex_args, print_mirex_args
# Reminder: correct default arguments in metrics_argument_funcs.py after tests are done

from utilities.constants import *
from utilities.device import get_device


def get_prompt(pieces, prompt_len):
    return pieces[0][:prompt_len]


def get_continuations(pieces, prompt_len):
    return [piece[prompt_len:] for piece in pieces]


def compute_continuation_prob(gen_probs, continuation, cont_length):
    # Decide if tensors or np.arrays are better. For np.array, use continuation.detach().numpy()
    # P = 1/L sum(p(x_ij|)
    return 1


def main():
    args = parse_mirex_args()
    print_mirex_args(args)

    # Load model
    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                             d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                             max_sequence=args.max_sequence, rpr=args.rpr, pmp=False).to(get_device())
    model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')))     # Change map_location for training on GPU

    model.training = False  # Find out why it presumes the model is being trained

    # Grab test dataset
    _, _, dataset = create_epiano_datasets(args.midi_root, args.target_seq_length, random_seq=False)

    for i in range(args.num_tests):
        print(f"Computing test number {i + 1} of {args.num_tests}:")

        # Grab random pieces from dataset
        indexes = random.sample(range(len(dataset)), args.num_continuations)    # Include random state
        pieces = [dataset[idx][0] for idx in indexes]

        # Get prompts and continuations (they are tensors)
        prompt = get_prompt(pieces, args.prompt_length)
        continuations = get_continuations(pieces, args.prompt_length)

        # Generate probability distributions from prompt
        _, gen_probs = model.generate(prompt, args.target_seq_length, beam=args.beam)
        # For illustration purposes, save gen_probs in a csv file
        np.savetxt('gen_probs.csv', gen_probs, delimiter=',')  # Save to root directory

        # For each continuation, compute the probability of it being chosen
        cont_length = args.target_seq_length - args.prompt_length
        probs = []
        for cont in continuations:
            probs.append(compute_continuation_prob(gen_probs, cont, cont_length))

        # See if it matches the real continuation (index 0) and keep score
        choice = np.argmax(np.array(probs))
        score = []
        if choice == 0:
            print("Correct prediction.")
            score.append(1)
        else:
            print("Wrong prediction.")
            score.append(0)

        print("------")

    # Compute final score
    final_score = np.mean(np.array(score))

    print(f"Final MIREX-like Continuation Prediction Challenge Score: {final_score}")

    return


if __name__ == "__main__":
    main()
