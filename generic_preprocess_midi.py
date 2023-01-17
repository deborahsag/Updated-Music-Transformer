import argparse
import os
import pickle
from glob import glob
import third_party.midi_processor.processor as midi_processor
from os.path import exists

# prep_midi
def prep_midi(input_dir, output_dir):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pre-processes the maestro dataset, putting processed midi data (train, eval, test) into the
    given output folder
    ----------
    """

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    midis = glob(f"{input_dir}/**/*.mid*", recursive=True)
    print("Found", len(midis), "pieces")
    print("Preprocessing...")

    total_count = 0
    train_count = 0
    val_count   = 0
    test_count  = 0

    for mid in midis:
        dir_split, extension_split = mid.rfind("/") + 1, mid.rfind(".mid")
        filepath, filename, extension = mid[:dir_split], mid[dir_split:extension_split], mid[extension_split:]

        split_type  = filepath.split("/")[-2]
        f_name      = filename + ".pickle"

        if(split_type == "train"):
            o_file = os.path.join(train_dir, f_name)
            train_count += 1
        elif(split_type == "validation"):
            o_file = os.path.join(val_dir, f_name)
            val_count += 1
        elif(split_type == "test"):
            o_file = os.path.join(test_dir, f_name)
            test_count += 1
        else:
            print("ERROR: Unrecognized split type:", split_type)
            return False

        if not exists(o_file):
            try:
                prepped = midi_processor.encode_midi(mid)
                o_stream = open(o_file, "wb")
                pickle.dump(prepped, o_stream)
                o_stream.close()
            except: #(EOFError, IndexError, KeySignatureError) as e:
                print(f"Error encoding {mid}")
                #return False

        total_count += 1
        if(total_count % 100 == 0):
            print(total_count, "/", len(midis))

    print("Num Train:", train_count)
    print("Num Val:", val_count)
    print("Num Test:", test_count)
    return True



# parse_args
def parse_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Parses arguments for preprocess_midi using argparse
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", type=str, help="Root folder for the Maestro dataset")
    parser.add_argument("output_dir", type=str, help="Output folder to put the preprocessed midi into")

    return parser.parse_args()

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Preprocesses maestro and saved midi to specified output folder.
    ----------
    """

    args            = parse_args()
    input_dir       = args.input_dir
    output_dir      = args.output_dir

    print("Preprocessing midi files and saving to", output_dir)
    prep_midi(input_dir, output_dir)
    print("Done!\n")

if __name__ == "__main__":
    main()
