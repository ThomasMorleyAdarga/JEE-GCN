import argparse
import json



















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='egnn for ed')
    parser.add_argument('input_file', default="", type=str)
    parser.add_argument('output_file', default="", type=str)
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    # Open ACE corpus from function params
    with open(input_file, 'r') as f:
        ace_dataset = json.load(f)
        f.close()

    train(ace_dataset)