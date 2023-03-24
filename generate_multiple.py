import os
import random

from third_party.midi_processor.processor import decode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets

from utilities.constants import *
from utilities.device import get_device, use_cuda

from structureness_indicators import structureness_indicators


# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args(generate_multiple=True)
    print_generate_args(args, generate_multiple=True)

    if (args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.target_seq_length, args.new_notation, random_seq=False)

    # Can be None, an integer index to dataset, or a file path
    if (args.primer_file is None):
        f = random.sample(range(len(dataset)), args.num_primer_files)
    else:
        f = [args.primer_file]

    for j in range(args.num_primer_files):
        idx = int(f[j])
        primer, _ = dataset[idx]
        primer = primer.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")
        decode_midi(primer.tolist(), f"{args.output_dir}/original-{idx}.mid")

        model = MusicTransformer(new_notation=args.new_notation, n_layers=args.n_layers, num_heads=args.num_heads,
                                 d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                                 max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

        model.load_state_dict(torch.load(args.model_weights))

        # Saving primer first
        f_path = os.path.join(args.output_dir, f"primer-{idx}.mid")
        decode_midi(primer[:args.num_prime].tolist(), file_path=f_path)

        # GENERATION
        for i in range(args.num_samples):
            print(f"Generating piece {idx}-{i}")
            model.eval()
            with torch.set_grad_enabled(False):
                if (args.beam > 0):
                    print("BEAM:", args.beam)
                    beam_seq, _ = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam, pmp=pmp if args.pmp else None)

                    f_path = os.path.join(args.output_dir, f"beam-{idx}-{i}.mid")
                    decode_midi(beam_seq[0].tolist(), f_path)
                else:
                    print("RAND DIST")
                    rand_seq, _ = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0, pmp=pmp if args.pmp else None)

                    f_path = os.path.join(args.output_dir, f"rand-{idx}-{i}.mid")
                    decode_midi(rand_seq[0].tolist(), f_path)

            print()

        # if args.struct:
        #     structureness_indicators(args.output_dir)

if __name__ == "__main__":
    main()
