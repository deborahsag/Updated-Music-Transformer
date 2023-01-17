import pretty_midi
import os
from glob import glob
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from itertools import cycle
from statistics import mode
from copy import deepcopy
from pathlib import Path


# Groups notes with the same start timestamp. The result is a mixed list of single notes and lists of notes
def group_chords(sorted_notes, tolerance):
    notes = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        notes[-1].append(note) if abs(note.start - notes[-1][-1].start) <= tolerance else notes.append([note])
    return [note[0] if len(note) == 1 else sorted(note, key=lambda x: x.pitch, reverse=True) for note in notes]


# Extract all voices from all tracks of given midi
def extract_midi(mid):
    dir_split, extension_split = mid.rfind("/") + 1, mid.rfind(".mid")
    filepath, filename, extension = mid[:dir_split].replace(PATH, ""), mid[dir_split:extension_split], mid[extension_split:]

    try:
        mid = pretty_midi.PrettyMIDI(mid)
    except:
        return
    mid.remove_invalid_notes()

    for i, instr in enumerate(mid.instruments):
        if not instr.is_drum:
            formatted_notes = group_chords(sorted(instr.notes, key=lambda x: (x.start, x.pitch)), tolerance)
            n_voices = mode([len(x) if type(x) is list else 1 for x in formatted_notes])
            melodies = extract_melodies(formatted_notes, n_voices, tolerance)

            for j, melody in enumerate(melodies):
                intervals = to_intervals([(note.pitch, note.duration) for note in melody])
                is_melody = classify(intervals)
                if is_melody:
                    export(melody, Path(f"{EXPORT_PATH}{filepath}{filename}-{i}-{j}{extension}").as_posix())


# Save midi of the given melody at given filepath
def export(melody, filepath):
    # Remove silence at start
    initial_timestamp = melody[0].start
    for note in melody:
        note.start -= initial_timestamp
        note.end -= initial_timestamp

    new_mid = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    piano.notes = melody
    for note in piano.notes:
        note.velocity = 70
    new_mid.instruments.append(piano)
    new_mid.write(Path(filepath).as_posix())


# Decide if a sequence of notes is a melody or accompaniment. Returns true if melody
def classify(formatted_melody):
    intervals = [note[0] for note in formatted_melody]
    durations = [note[1] for note in formatted_melody]
    too_repetitive = intervals.count(0) >= len(intervals) * 0.66
    too_slow = sum(durations) >= len(durations) * 2
    return not too_slow and not too_repetitive


def to_intervals(notes):
    seq_in_intervals = [(0, notes[0][1])]  # First interval is always 0
    for i in range(1, len(notes)):
        seq_in_intervals.append((notes[i][0] - notes[i - 1][0], notes[i][1]))
    return seq_in_intervals


# Extract n_voices monophonic melodies from the formatted stream built by group_chords
def extract_melodies(notes, n_voices=1, tolerance=0.05):
    melodies = [[] for _ in range(n_voices)]

    # Since we select the notes based on previous ones, we put first note/chord in separately...
    if type(notes[0]) != list:
        for i in range(n_voices):
            melodies[i].append(notes[0])
    else:
        if n_voices > len(notes[0]):
            circular = cycle(notes[0])
            temp = []
            for i in range(n_voices):
                temp.append(circular.__next__())
            temp = sorted(temp, key=(lambda x: (x.start, x.pitch)))
            for i in range(n_voices):
                melodies[i].append(temp[i])
        else:
            for i in range(n_voices//2):
                melodies[i].append(notes[0][i])
                melodies[-(i+1)].append(notes[0][-(i+1)])
            if n_voices % 2 == 1:
                melodies[n_voices//2].append(notes[0][n_voices//2])

    # And then the rest of the notes/chords
    for item in notes[1:]:
        if type(item) != list:
            for melody in melodies:
                melody.append(item)
        else:
            if len(item) == n_voices:
                for i in range(n_voices):
                    melodies[i].append(item[i])
            elif len(item) < n_voices:
                available_voices = [i for i in range(n_voices) if item[0].start - melodies[i][-1].end >= -tolerance]
                circular = cycle(reversed(item))
                temp = []
                for i in range(len(available_voices)):
                    temp.append(circular.__next__())
                temp = sorted(temp, key=(lambda x: (x.start, x.pitch)))
                for note, voice in zip(temp, available_voices):
                    melodies[voice].append(note)
            else:
                for i in range(n_voices//2):
                    melodies[i].append(item[i])
                    melodies[-(i+1)].append(item[-(i+1)])
                if n_voices % 2 == 1:
                    melodies[n_voices//2].append(item[n_voices//2])

    # Fill rests and truncate overlapping notes
    melodies = [deepcopy(melody) for melody in melodies]  # This is needed because notes in lists are just references
    for melody in melodies:
        for i in range(len(melody)-1):
            melody[i].end = melody[i+1].start

    return melodies



if __name__ == '__main__':
    tolerance = 0.05
    parallelize = True  # Uses a lot of RAM
    PATH = "/mnt/e/Datasets/maestro-flat"
    EXPORT_PATH = "/mnt/e/Datasets/maestro-flat-monophonic/"
    for dir in glob(f"{PATH}/*/"):
        os.makedirs(EXPORT_PATH + dir.split("/")[-2])
    midis = glob(f"{PATH}/**/*.mid*", recursive=True)

    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)

    if parallelize:
        process_map(extract_midi, midis, chunksize=1)
    else:
        for mid in tqdm(midis):
            extract_midi(mid)

    print("Done.")
