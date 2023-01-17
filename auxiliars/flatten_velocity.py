from glob import glob
import pretty_midi
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

parallelize = True  # Uses a lot of RAM
PATH = "/mnt/e/Datasets/lmd-full-pianified-augmented-flat"


def extract_midi(mid):
    filename = mid
    try:
        mid = pretty_midi.PrettyMIDI(mid)
    except:  # (IOError, ValueError, KeySignatureError) as e:
        return None
    mid.remove_invalid_notes()

    for instr in mid.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                note.velocity = 70
    mid.write(filename)


if __name__ == "__main__":
    midis = glob(f"{PATH}/**/*.mid*", recursive=True)
    if not parallelize:
        [extract_midi(mid) for mid in tqdm(midis)]
    else:
        process_map(extract_midi, midis, chunksize=1)
    print("Done.")
