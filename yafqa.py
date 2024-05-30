import argparse, sys

parser=argparse.ArgumentParser()
parser.add_argument("--tr_dataset")
parser.add_argument("--hari")

def main():
    args = parser.parse_args()
    print(f"args = {args.tr_dataset} {args.hari}")

if __name__ == "__main__":
    main()