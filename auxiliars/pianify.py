from glob import glob
import pretty_midi
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os

parallelize = True
PATH = "/mnt/e/Curated MIDIs/pianified/"
FAILED_FILES = []


def extract_midi(mid):
    filename = mid

    try:
        mid = pretty_midi.PrettyMIDI(mid)
    except Exception as e:  # (IOError, ValueError, KeySignatureError) as e:
        FAILED_FILES.append(mid)
        return

    mid.remove_invalid_notes()
    notes = []
    for instr in mid.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                notes.append(note)

    # Different instruments may play the same notes at the same time or have overlapping notes
    # This code cleans those up since they're now redundant
    notes = sorted(notes, key=lambda x: (x.pitch, x.start))
    for i in range(len(notes)-1, -1, -1):
        if notes[i].pitch == notes[i-1].pitch:
            if notes[i].start == notes[i-1].start:
                notes[i-1].end = max(notes[i].end, notes[i-1].end)
                del notes[i]
            else:
                notes[i-1].end = min(notes[i-1].end, notes[i].start)

    new_mid = pretty_midi.PrettyMIDI()
    new_mid_instr = pretty_midi.Instrument(0)
    new_mid_instr.notes = notes
    new_mid.instruments.append(new_mid_instr)
    new_mid.write(filename)


if __name__ == "__main__":
    midis = glob(f"{PATH}/**/*.[mid|MID]*", recursive=True)
    print(f"Detected {len(midis)} files")
    if not parallelize:
        [extract_midi(mid) for mid in tqdm(midis)]
    else:
        process_map(extract_midi, midis, chunksize=1)

    print(f"Failed to pianify {len(FAILED_FILES)} files. Removing them...")
    for file in FAILED_FILES:
        os.remove(file)
    print("Done.")
