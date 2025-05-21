# combine_data.py
# combine data according to ratios

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="", help="Comma-separated paths to data files.")
parser.add_argument("--ratios", type=str, default="", help="Comma-separated ratios for each data file.")
parser.add_argument("--total_words_m", type=float, default=10, help="Total words in combined data, in millions.")
parser.add_argument("--output_path", type=str, default="", help="Output path for combined data.")


def main():
    args = parser.parse_args()
    data_paths = args.data_path.split(",")
    ratios = [float(r) for r in args.ratios.split(",")]
    # normalize ratios
    total_ratio = sum(ratios)
    ratios = [r / total_ratio for r in ratios]

    total_words = args.total_words_m * 1e6
    total_words_per_file = [total_words * r for r in ratios]

    # read data
    data = []
    word_count = 0
    for path, total in zip(data_paths, total_words_per_file):
        with open(path, "r") as file:
            for line in file:
                if word_count + len(line.split(" ")) > total:
                    word_count = 0
                    break
                word_count += len(line.split(" "))
                data.append(line)

    with open(args.output_path, "w") as f:
        f.writelines(data)


if __name__ == "__main__":
    main()