import torch
import torch.nn as nn
import os
import random
import pretty_midi
import pickle

from third_party.midi_processor.processor import decode_midi, encode_midi

from statistics import mean
from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

NSAMPLES = 10
INDICES = []


# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False, pmp=args.pmp)
    
    if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file
    
    idx = int(f)
    primer, _, _  = dataset[idx]
    primer = primer.int().to(get_device())

    print("Using primer index:", idx, "(", dataset.data_files[idx], ")")
    decode_midi(primer.tolist(), f"{args.output_dir}/original-{idx}.mid")
    
    dir_split = dataset.data_files[idx].rfind("/") + 1
    filepath, filename = dataset.data_files[idx][:dir_split], dataset.data_files[idx][dir_split:-7]
    pmp = torch.FloatTensor(pickle.load(open(f"{filepath}Pmp-{filename}-1.pickle", "rb"))).to(get_device()) if args.pmp else None
    if pmp is not None and pmp.shape[0] < args.target_seq_length:
        pmp = torch.cat((pmp, torch.zeros(args.target_seq_length - pmp.shape[0]).to(get_device())))


    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr, pmp=args.pmp).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))
    
    f_path = os.path.join(args.output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].tolist(), f_path)
    
    for i in range(NSAMPLES):
        print(f"Generating song {i}")
        # GENERATION
        model.eval()
        with torch.set_grad_enabled(False):
            if(args.beam > 0):
                print("BEAM:", args.beam)
                beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam, pmp=pmp if args.pmp else None)

                f_path = os.path.join(args.output_dir, f"beam-{i}.mid")
                decode_midi(beam_seq[0].tolist(), f_path)
            else:
                print("RAND DIST")
                rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0, pmp=pmp if args.pmp else None)

                f_path = os.path.join(args.output_dir, f"rand-{i}.mid")
                decode_midi(rand_seq[0].tolist(), f_path)
                
        print()


if __name__ == "__main__":
    main()
