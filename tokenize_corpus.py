import argparse
import json


def main(input_file, output_file, dep_file):

    entities = set()
    pos_tags = set()
    dep_labels = set()

    # First open Json file
    with open(input_file, 'r') as f:
        ace_original = json.load(f)
        f.close()

    with open(output_file, 'w') as f:
        for example in ace_original:
            for entity in example['golden-entity-mentions']:
                entities.add(entity['entity-type'])
            for pos in example['pos-tags']:
                pos_tags.add(pos)
            for dep in example['stanford-colcc']:
                dep_labels.add(dep)
            for word in example['words']:
                f.write(word + " ")
            f.write("\n")
        f.close()

    j = 0
    pos_string = ""
    for pos_tag in pos_tags:
        pos_string += "'" + pos_tag + "': " + str(j) + ", "
        j += 1
    print(pos_string)


    dep_string = ""
    for dep_label in dep_labels:
        dep_string += "'" + dep_label.split('/')[0] + "', "
    with open(dep_file, 'w') as f:
        f.write(dep_string)
        f.close()



    '''
    i = 0
    string = ""
    for entity in entities:
        string += "'" + entity + "': "+ str(i) + ", "
        i += 1
    print(string)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='egnn for ed')

    parser.add_argument('input_file', default="", type=str)
    parser.add_argument('output_file', default="", type=str)
    parser.add_argument('dep_file', default="", type=str)

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    dep_file = args.dep_file

    main(input_file, output_file, dep_file)