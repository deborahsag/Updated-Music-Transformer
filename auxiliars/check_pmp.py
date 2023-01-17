from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pickle
from os.path import exists
import pretty_midi


if __name__ == "__main__":
    PATH = "/mnt/f/Datasets/new-maestro-augmented-pickles-pmp/train"
    pickles = glob(f"{PATH}/**/*.pickle*", recursive=True)
    pickles = [p for p in pickles if "/Pmp-" not in p and "/Aug-" not in p]
    pitch_shift, time_shift = [-3, -2, -1, 0, 1, 2, 3], [0.95, 0.975, 1, 1.025, 1.05]
    
    for pick in tqdm(pickles):
        dir_split = pick.rfind("/") + 1
        filepath, filename = pick[:dir_split], pick[dir_split:-7]

        for time in time_shift:
            n_tokens = [] if time != 0 else [len(pickle.load(pick))]
            for pitch in pitch_shift:
                if time == 1 and pitch == 0:
                    continue
                n_tokens.append(len(pickle.load(open(f"{filepath}Aug-{filename}-{pitch}-{time}.pickle" , "rb"))))
            if n_tokens.count(n_tokens[0]) != len(n_tokens):
                print(f"Discrepancy on {filename} pitch {pitch}: {n_tokens}")
                exit(0)
    exit(0)
    filename, pitch, time = "MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--5", 2, 0.95
    aug, orig = f"/mnt/f/Datasets/new-maestro-augmented-pickles/train/Aug-{filename}-{pitch}-{time}.pickle", f"/mnt/f/Datasets/new-maestro-augmented-pickles/train/{filename}.pickle"
    aug, orig = pickle.load(open(aug, "rb")), pickle.load(open(orig, "rb"))
    print(f"Check {filename}. Aug has {len(aug)} but orig has {len(orig)} tokens")

    aug_mid, orig_mid = f"/mnt/f/Datasets/new-maestro-augmented/train/Aug-{filename}-{pitch}-{time}.midi", f"/mnt/f/Datasets/new-maestro-augmented/train/{filename}.midi"
    aug_mid, orig_mid = pretty_midi.PrettyMIDI(aug_mid), pretty_midi.PrettyMIDI(orig_mid)
    print(f"Aug mid instruments: {aug_mid.instruments}")
    print(f"Orig mid instruments: {orig_mid.instruments}")
    print(f"Aug instr0 notes: {len(aug_mid.instruments[0].notes)}, orig instr0 notes: {len(orig_mid.instruments[0].notes)}")

    print("Done.")
