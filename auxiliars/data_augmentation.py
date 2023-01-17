from glob import glob
import pretty_midi
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from copy import deepcopy
from os.path import exists

pitch_shift = [-3, -2, -1, 0, 1, 2, 3]
time_shift = [0.95, 0.975, 1, 1.025, 1.05]
parallelize = False
PATH = "/mnt/f/Datasets/vgmusic-snes-pianified-augmented/train"


def extract_midi(mid):
    print(f"Doing for {mid}")
    dir_split, extension_split = mid.rfind("/") + 1, mid.rfind(".mid") if ".mid" in mid else mid.rfind(".MID")
    filepath, filename, extension = mid[:dir_split], mid[dir_split:extension_split], mid[extension_split:]

    mid = pretty_midi.PrettyMIDI(mid)

        
    mid.remove_invalid_notes()
    mid.instruments = [instr for instr in mid.instruments if not instr.is_drum]

    for pitch in pitch_shift:
        for time in time_shift:
            if (pitch == 0 and time == 1) or exists(f"{filepath}Aug-{filename}-{pitch}-{time}{extension}"):
                continue

            new_mid = deepcopy(mid)
            for instr in new_mid.instruments:
                for note in instr.notes:
                    note.pitch += pitch
                    if note.pitch > 127:
                        note.pitch -= 12
                    elif note.pitch < 0:
                        note.pitch += 12
                    note.start, note.end = note.start * time, note.end * time
            print(f"Writing {filepath}Aug-{filename}-{pitch}-{time}{extension}")
            new_mid.write(f"{filepath}Aug-{filename}-{pitch}-{time}{extension}")


if __name__ == "__main__":
    if "train" not in PATH:
        print(f"Path should be of training folder, but is {PATH}")
        exit(0)
    midis = glob(f"{PATH}/**/*.mid*", recursive=True) + glob(f"{PATH}/**/*.MID*", recursive=True)
    midis = [m for m in midis if "/Aug-" not in m]
    if not parallelize:
        [extract_midi(mid) for mid in tqdm(midis)]
    else:
        process_map(extract_midi, midis, max_workers=12, chunksize=1)
    print("Done.")
