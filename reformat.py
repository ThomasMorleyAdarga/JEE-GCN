import argparse
import json

def main(input, output):

    #First open Json file
    with open(input, 'r') as f:
        ace_original = json.load(f)

        ace_formatted = list()


        max_seq_len = 0



        #Iterate data points
        for sample in ace_original:

            #Create array matrix for current sentence
            entire_block = [dict.fromkeys(('word', 'entity-types', 'syntax-label', 'pos-tag', 'event-bio')) for i in range(len(sample['words']))]

            #Iterate words in this current sample
            for i, current in enumerate(entire_block):

                #Add token to the output matrix
                if sample['words'][i] == ' ' or sample['words'][i].startswith("\\"):
                    word = 'q'
                else:
                    word = sample['words'][i].lower()

                current.update({'word': word})

                #Set default values for entity type
                current.update({'entity-types': []})

                #Set default "None" for event type
                current.update({'event-bio': "None"})

            #Set pos_tag
            for i, pos_tag in enumerate(sample['pos-tags']):
                entire_block[i]['pos-tag'] = pos_tag

            # Set syntactic label
            for syntactic_label in sample['stanford-colcc']:
                if len(syntactic_label.split('/')) < 4:
                    if int(syntactic_label.split('/')[1].split('=')[1]) < len(sample['words']):
                        entire_block[int(syntactic_label.split('/')[1].split('=')[1])].update({'syntax-label': syntactic_label.split('/')[0]})
                else:
                    if int(syntactic_label.split('/')[2].split('=')[1]) < len(sample['words']):
                        entire_block[int(syntactic_label.split('/')[2].split('=')[1])].update({'syntax-label': syntactic_label.split('/')[0]})


            #Add entity types for all words

            for entity in sample['golden-entity-mentions']:
                wordIndex = entity['start']
                while wordIndex < entity['end']:
                    entire_block[wordIndex]['entity-types'] += [entity['entity-type']]
                    wordIndex += 1


            # Add golden event labels for training

            for event in sample['golden-event-mentions']:

                if len(event['trigger']['text'].split('-')) > 1:
                    event_text = event['trigger']['text'].split('-')
                else:
                    event_text = event['trigger']['text'].split()

                for i in range(len(event_text)):
                    wordIndex = i + event['trigger']['start']
                    if i == 0:
                        entire_block[wordIndex]['event-bio'] = "B-" + event['event_type']
                    else:
                        entire_block[wordIndex]['event-bio'] = "I-" + event['event_type']


            #write data to output file
            ace_formatted.append(entire_block)

            if len(entire_block) > max_seq_len: max_seq_len = len(entire_block)

    print(max_seq_len)

    f.close()
    with open(out_file, 'w') as o:
        json.dump({'train': ace_formatted}, o, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='egnn for ed')

    parser.add_argument('inp', default="", type=str)
    parser.add_argument('out', default="", type=str)

    args = parser.parse_args()

    in_file = args.inp
    out_file = args.out

    main(in_file, out_file)