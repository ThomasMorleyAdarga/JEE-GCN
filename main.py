import argparse
import json
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel


def produce_embeddings(ace_dataset, split_category, output_file):

    #Init device
    device = torch.device('CUDA')

    #Init JSON output
    data = dict.fromkeys({'train', 'dev', 'test'})
    data['train'] = dict.fromkeys({'X', 'Y', 'N'})
    data['train']['X'], data['train']['Y'] = list(), list()
    with open(output_file, "w") as f:
        json.dump(data, f)
        f.close()

    #Init N (number of samples total)
    N = 0

    #   ----    dictionary for event types  ---     #
    event_dict = {'B-Life:Marry': 0, 'I-Personnel:Start-Position': 1, 'I-Justice:Release-Parole': 2,
                  'B-Conflict:Attack': 3, 'B-Business:Start-Org': 4, 'B-Justice:Trial-Hearing': 5, 'B-Justice:Fine': 6,
                  'B-Justice:Pardon': 7, 'B-Contact:Phone-Write': 8, 'I-Business:Merge-Org': 9, 'B-Life:Injure': 10,
                  'I-Justice:Sentence': 11, 'I-Transaction:Transfer-Ownership': 12, 'B-Justice:Appeal': 13,
                  'I-Movement:Transport': 14, 'I-Life:Be-Born': 15, 'B-Personnel:Start-Position': 16,
                  'I-Conflict:Attack': 17, 'B-Justice:Sue': 18, 'B-Business:Declare-Bankruptcy': 19,
                  'B-Justice:Convict': 20, 'B-Justice:Arrest-Jail': 21, 'B-Life:Divorce': 22,
                  'B-Justice:Charge-Indict': 23, 'I-Conflict:Demonstrate': 24, 'I-Personnel:Elect': 25,
                  'B-Business:End-Org': 26, 'I-Transaction:Transfer-Money': 27, 'B-Justice:Extradite': 28,
                  'B-Justice:Acquit': 29, 'I-Justice:Acquit': 30, 'B-Justice:Release-Parole': 31, 'B-Contact:Meet': 32,
                  'I-Business:Start-Org': 33, 'B-Life:Die': 34, 'I-Contact:Meet': 35, 'I-Personnel:End-Position': 36,
                  'B-Transaction:Transfer-Ownership': 37, 'B-Personnel:Elect': 38, 'I-Justice:Charge-Indict': 39,
                  'I-Justice:Convict': 40, 'I-Life:Injure': 41, 'I-Business:End-Org': 42, 'B-Conflict:Demonstrate': 43,
                  'B-Justice:Execute': 44, 'B-Life:Be-Born': 45, 'I-Justice:Execute': 46, 'I-Life:Die': 47,
                  'I-Life:Marry': 48, 'I-Contact:Phone-Write': 49, 'B-Personnel:Nominate': 50,
                  'B-Transaction:Transfer-Money': 51, 'I-Justice:Arrest-Jail': 52, 'B-Justice:Sentence': 53,
                  'B-Business:Merge-Org': 54, 'B-Personnel:End-Position': 55, 'B-Movement:Transport': 56, 'None': 57}

    # Instantiate pretrained bert model and tokenizer !
    berty = BertModel.from_pretrained("bert-base-uncased").float().to(device)
    berty_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').float().to(device)

    # Load glove embeddings
    glove_embeddings_dict = {}
    with open('glove.6B/glove.6B.300d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_embeddings_dict[word] = vector

    # Define entity types and create dictionary of indeces for one hot embeddings
    entities_dict = {'GPE:Continent': 0, 'VEH:Subarea-Vehicle': 1, 'PER:Individual': 2, 'LOC:Address': 3, 'ORG:Educational': 4, 'VEH:Water': 5, 'GPE:Special': 6, 'WEA:Nuclear': 7, 'VEH:Underspecified': 8, 'Contact-Info:E-Mail': 9, 'WEA:Projectile': 10, 'FAC:Path': 11, 'LOC:Region-International': 12, 'Job-Title': 13, 'GPE:GPE-Cluster': 14, 'Crime': 15, 'ORG:Sports': 16, 'GPE:Population-Center': 17, 'WEA:Biological': 18, 'LOC:Boundary': 19, 'TIM:time': 20, 'LOC:Land-Region-Natural': 21, 'WEA:Shooting': 22, 'ORG:Religious': 23, 'LOC:Water-Body': 24, 'ORG:Media': 25, 'LOC:Region-General': 26, 'PER:Indeterminate': 27, 'ORG:Government': 28, 'WEA:Exploding': 29, 'ORG:Entertainment': 30, 'LOC:Celestial': 31, 'ORG:Commercial': 32, 'WEA:Sharp': 33, 'GPE:Nation': 34, 'GPE:County-or-District': 35, 'WEA:Underspecified': 36, 'VEH:Land': 37, 'FAC:Building-Grounds': 38, 'FAC:Subarea-Facility': 39, 'WEA:Blunt': 40, 'Contact-Info:URL': 41, 'WEA:Chemical': 42, 'VEH:Air': 43, 'Numeric:Money': 44, 'GPE:State-or-Province': 45, 'Contact-Info:Phone-Number': 46, 'ORG:Non-Governmental': 47, 'Numeric:Percent': 48, 'Sentence': 49, 'ORG:Medical-Science': 50, 'FAC:Plant': 51, 'FAC:Airport': 52, 'PER:Group': 53}

    # Define pos tags and define random vectors to each
    pos_dict = {'ADD': 0, 'MD': 1, ':': 2, 'NFP': 3, '$': 4, 'NNPS': 5, 'JJ': 6, 'WP': 7, 'TO': 8, 'RP': 9, 'HYPH': 10, 'SYM': 11, 'CD': 12, '-RRB-': 13, 'LS': 14, 'PRP': 15, 'IN': 16, '_SP': 17, 'VB': 18, 'WRB': 19, 'AFX': 20, 'VBN': 21, 'VBD': 22, 'EX': 23, 'RB': 24, 'VBZ': 25, 'JJR': 26, 'POS': 27, 'VBG': 28, 'UH': 29, 'WP$': 30, "''": 31, 'NNP': 32, 'XX': 33, 'DT': 34, 'WDT': 35, 'NNS': 36, 'NN': 37, 'PDT': 38, ',': 39, '.': 40, 'JJS': 41, 'PRP$': 42, 'RBR': 43, 'CC': 44, '-LRB-': 45, '``': 46, 'FW': 47, 'RBS': 48, 'VBP': 49, None: 50}
    for pos in pos_dict:
        pos_dict.update({pos: np.random.rand(15)})

    # Define dep labels and assign random vectors again
    dep_dict = {'O': 0, 'punct': 1, 'iobj': 2, 'parataxis': 3, 'auxpass': 4, 'aux': 5, 'conj': 6,
                'advcl': 7, 'acl:relcl': 8, 'nsubjpass': 9, 'csubj': 10, 'compound': 11, 'compound:prt': 12,
                'mwe': 13, 'cop': 14, 'neg': 15, 'nmod:poss': 16, 'appos': 17, 'cc:preconj': 18, 'nmod': 19,
                'nsubj': 20, 'xcomp': 21, 'det:predet': 22, 'nmod:npmod': 23, 'acl': 24, 'amod': 25,
                'expl': 26, 'csubjpass': 27, 'case': 28, 'ccomp': 29, 'dobj': 30, 'ROOT': 31,
                'discourse': 32, 'nmod:tmod': 33, 'dep': 34, 'nummod': 35, 'mark': 36, 'advmod': 37,
                'cc': 38, 'det': 39, 'nmod:in': 40, 'det:qmod': 41, 'nmod:until': 41, 'nmod:of': 43, 'nmod:on': 44,
                'advcl:as': 45, 'nmod:with': 46, 'nmod:by': 47,
                'nmod:instead_of': 48, 'nmod:for': 49, 'nmod:to': 50, 'ref': 52, 'nmod:agent': 53, 'conj:and': 54,
                'nmod:like': 55, 'nsubj:xsubj': 56,
                'nmod:at': 57, 'advcl:if': 58, 'nmod:than': 59, 'conj:or': 60, 'acl:to': 61, 'nmod:under': 62,
                'advcl:to': 63, 'acl:of': 64, 'nmod:from': 65,
                'nmod:between': 66, 'advcl:for': 67, 'conj:but': 68, 'advcl:in': 69, 'nmod:as': 70, 'advcl:because': 71,
                'advcl:than': 72, 'nmod:after': 73,
                'nmod:about': 74, 'nmod:over': 75, 'nmod:within': 76, 'advcl:about': 77, 'nsubjpass:xsubj': 78,
                'advcl:while': 79, 'nmod:through': 80,
                'advcl:into': 81, 'advcl:like': 82, 'nmod:against': 83, 'nmod:out_of': 84, 'nmod:across': 85,
                'acl:for': 86, 'nmod:because_of': 87,
                'conj:instead': 88, 'acl:in': 89, 'advcl:without': 90, 'nmod:because': 91, 'advcl:before': 92,
                'nmod:except_for': 93, 'nmod:around': 94,
                'advcl:at': 95, 'nmod:during': 96, 'nmod:according_to': 97, 'advcl:since': 98, 'conj:&': 99,
                'acl:whether': 100, 'advcl:so': 101,
                'advcl:unless': 102, 'nmod:toward': 103, 'advcl:until': 104, 'advcl:in_order': 105, 'acl:on': 106,
                'nmod:behind': 107, 'advcl:by': 108,
                'nmod:among': 109, "nmod:'s": 110, 'advcl:on': 111, 'nmod:without': 112, 'advcl:of': 113,
                'nmod:into': 114, 'nmod:such_as': 115,
                'nmod:except': 116, 'nmod:but': 117, 'nmod:far_from': 118, 'nmod:that': 119, 'nmod:near': 120,
                'nmod:out': 121, 'advcl:whether': 122,
                'advcl:with': 123, 'nmod:down': 124, 'advcl:so_that': 125, 'advcl:that': 126, 'acl:as': 127,
                'nmod:including': 128, 'nmod:before': 129,
                'nmod:beyond': 130, 'conj:in': 131, 'nmod:up': 132, 'nmod:atop': 133, 'advcl:once': 134,
                'nmod:beginning': 135, 'advcl:after': 136, 'nmod:past': 137, 'advcl:such': 138, 'nmod:vs.': 139,
                'advcl:behind': 140, 'nmod:inside_of': 141,
                'nmod:given': 142, 'acl:based_on': 143, 'nmod:plus': 144, 'nmod:onto': 145, 'nmod:\'': 146,
                'acl:before': 147, 'conj:just': 148, 'advcl:till': 149,
                'advcl:whilst': 150, 'nmod:in_front_of': 151, 'advcl:below': 152, 'nmod:oconer': 153,
                'nmod:away_from': 154, 'nmod:pending': 155, 'acl:compared_to': 156,
                'acl:until': 157, 'advcl:f.': 158, 'conj:+': 159, 'advcl:among': 160, 'advcl:through': 161,
                'acl:including': 162, 'nmod:together_with': 163, 'acl:because': 164,
                'conj:plus': 165, 'nmod:versus': 166, 'acl:next_to': 167, 'nmod:contrary_to': 168,
                'advcl:close_to': 169, 'advcl:around': 170, 'conj:andor': 171, 'conj:as': 172,
                'nmod:above': 173, 'nmod:while': 174, 'acl:between': 175, 'nmod:involving': 176, 'nmod:towards': 177,
                'acl:instead_of': 178, 'advcl:over': 179, 'nmod:regarding': 180,
                'nmod:despite': 181, 'nmod:next_to': 182, 'nmod:as_for': 183, 'nmod:besides': 184, 'advcl:toward': 185,
                'acl:about': 186, 'nmod:concerning': 187, 'advcl:instead_of': 188,
                'advcl:not': 189, 'conj:x': 190, 'conj:v.': 191, 'nmod:compared_with': 192, 'nmod:both': 193,
                'conj:even': 194, 'nmod:alongside': 195, 'nmod:beside': 196,
                'nmod:along': 197, 'advcl:though': 198, 'nmod:across_from': 199, 'nmod:aboard': 200,
                'nmod:on_behalf_of': 201, 'nmod:upon': 202, 'nmod:worth': 203, 'advcl:either': 204,
                'nmod:with_regard_to': 205, 'advcl:inside': 206, 'acl:besides': 207, 'nmod:throughout': 208,
                'nmod:as_of': 209, 'conj:vs': 210, 'nmod:amongst': 211,
                'nmod:considering': 212, 'nmod:next': 213, 'acl:at': 214, 'conj:only': 215, 'acl:from': 216,
                'nmod:regardless_of': 217, 'nmod:and': 218, 'nmod:excluding': 219,
                'advcl:compared_to': 220, 'acl:over': 221, 'advcl:_': 222, 'acl:after': 223, 'nmod:_': 224,
                'advcl:as_if': 225, 'conj:so': 226, 'advcl:ago': 227, 'advcl:depending': 228,
                'nmod:inside': 229, 'nmod:underneath': 230, 'nmod:via': 231, 'nmod:other': 232, 'advcl:although': 233,
                'nmod:off': 234, 'nmod:since': 235, 'advcl:abc': 236, 'nmod:beneath': 237,
                'advcl:rather_than': 238, 'nmod:outside': 239, 'nmod:either': 240, 'nmod:close_to': 241,
                'nmod:in_spite_of': 242, 'advcl:along': 243, 'acl:against': 244, 'nmod:along_with': 245,
                'advcl:under': 246, 'nmod:once': 247, 'nmod:per': 248, 'nmod:on_top_of': 249, 'nmod:amid': 250,
                'acl:without': 251, 'advcl:in_case': 252, 'acl:that': 253, 'nmod:as_to': 254, 'advcl:including': 255,
                'nmod:till': 256, 'acl:as_to': 257, 'advcl:\'s': 258, 'nmod:apart_from': 259, 'advcl:near': 260,
                'nmod:whether': 261, 'conj:not': 262, 'nmod:following': 263, 'acl:except': 264,
                'nmod:based_on': 265, 'nmod:in_addition_to': 266, 'acl:since': 267, 'advcl:ta': 268, 'acl:like': 269,
                'nmod:de': 270, 'nmod:outside_of': 271, 'advcl:based_on': 272, 'advcl:a.': 273,
                'nmod:or': 274, 'conj:negcc': 275, 'nmod:due_to': 276, 'advcl:except': 277, 'nmod:unlike': 278,
                'nmod:if': 279, 'acl:with': 280, 'nmod:below': 281, 'advcl:within': 282, 'advcl:outside': 283,
                'advcl:despite': 284,
                'conj:nor': 285, 'nmod:thru': 286, 'acl:by': 287, 'advcl:out_of': 288, 'advcl:out': 289,
                'nmod:in_case_of': 290, 'advcl:during': 291, 'advcl:from': 292, 'advcl:between': 293,
                'conj:versus': 294, None: 295}
    for dep in dep_dict:
        dep_dict.update({dep: np.random.rand(15)})



    #   ----    Iterate through dataset     ----    #

    current_batch, current_gold_batch = list(), list()
    encoding_batch_size = 128

    #Handle dataset in batches
    for batch in range(len(ace_dataset[split_category])/encoding_batch_size):

        #Iterate examples in batch
        for i, example in enumerate(ace_dataset[split_category][i*encoding_batch_size:(i+1)*encoding_batch_size]):

            N += 1      #Count num samples

            # Matrix and bert token holder for current example
            matrix = list()
            gold_matrix = list()    #Holder for golden labels for loss calculation
            bert_tokens = []



            # Iterate through tokens in current example
            for i, word in enumerate(example):

                #Produce one hot vector for target
                golden_event_label = torch.zeros(len(event_dict))
                golden_event_label[event_dict.get(word['event-bio'])] = 1
                golden_event_label = golden_event_label.float()

                # Update bert token holder - this is necessary because the bert tokenizer wants multiple words at once otherwise individual letters are tokenized
                bert_tokens.append(word['word'])

                # Lookup glove embedding, use random embedding if word is not in glove dictionary
                if word['word'] not in glove_embeddings_dict:
                    glove_embedding = [torch.from_numpy(np.zeros(300))]
                else:
                    glove_embedding = torch.tensor([glove_embeddings_dict[word['word']]])

                # Produce sum of one-hot vectors for entity embedding
                entity_embedding = torch.zeros(len(entities_dict))
                for entity in word['entity-types']:
                    entity_embedding[entities_dict.get(entity)] = 1

                # Look up random vectors from pos and dep dictionaries
                pos_embedding = torch.from_numpy(pos_dict.get(word['pos-tag']))
                dep_embedding = torch.from_numpy(dep_dict.get(word['syntax-label']))

                # Create embedding for current word and add to current matrix for this example - bert section will be concatenated to the front later
                matrix.append(torch.cat((
                    glove_embedding[0],
                    entity_embedding,
                    pos_embedding,
                    dep_embedding
                )))
                gold_matrix.append(golden_event_label)

            # Produce bert embedding for each word in current example at once
            bert_embedding = berty(torch.tensor([berty_tokenizer.encode(bert_tokens, add_special_tokens=True)]).float().to(device))[0]

            # Concatenate bert embeddings
            for i, elem in enumerate(matrix):
                if len(bert_embedding[0][i]) < 1:
                    bert_embedding[0][i] = torch.from_numpy(np.random.rand(768))
                matrix[i] = torch.cat((bert_embedding[0][i], elem))
                assert len(matrix[i]) == (768 + 300 + 54 + 15 + 15)

            # Compile matrices for current example
            matrix = torch.stack(matrix)
            gold_matrix = torch.stack(gold_matrix)

            # Pad sequence dim to 150
            max_len = 150
            for i in range(max_len - matrix.shape[0]):
                matrix = torch.cat((matrix, torch.zeros(1, 1152)))
                gold_matrix = torch.cat((gold_matrix, torch.zeros(1, 58)))

            # Append to batch
            current_batch.append(matrix)
            current_gold_batch.append(gold_matrix)


        # Save batch to JSON output
        with open(output_file, 'r') as f:
            data = json.load(f)
            f.close()
        with open(output_file, 'w') as f:
            for i in range(len(current_batch)):
                data['train']['X'] += current_batch[i].tolist()
                data['train']['Y'] += current_gold_batch[i].tolist()
            json.dump(data, f, indent=2)
            f.close()

    # Set N
    with open(output_file, 'r') as f:
        data = json.load(f)
        f.close()
    data['train']['N'] = N
    with open(output_file, 'w') as f:
        json.dump(data, f)
        f.close()



def train(input_file):

    device = torch.device('CUDA')

    #   --- Data prep   --- #
    X, Y, N = input_file['train']['X'], input_file['train']['Y'], input_file['train']['N']
    X = nn.utils.rnn.pad_sequence(torch.tensor(X)).float().to(device)
    Y = nn.utils.rnn.pad_sequence(torch.tensor(Y)).float().to(device)



    #    ---- NN Setup  -----   #
    input_size = (768 + 300 + 54 + 15 + 15)
    hidden_size = 64
    num_layers = 1
    learning_rate = 0.0001
    batch_size = 32
    dropout = 0.5
    regularisation = 0.0001
    sequence_length = 150
    dense_input_size = 128
    dense_hidden_size = 256
    dense_output_size = 58
    epochs = 100



    class BiLSTM_Layer(nn.Module):

        def __init__(self, input_size, hidden_size, num_layers, dropout, dense_input_size, dense_hidden_size, dense_output_size):
            super(BiLSTM_Layer, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.hidden_linear = nn.Linear(dense_input_size, dense_hidden_size)
            self.hidden = nn.ReLU()
            self.linear = nn.Linear(dense_hidden_size, dense_output_size)


        def forward(self, sample):
            h0 = torch.zeros(self.num_layers * 2, sequence_length, self.hidden_size)
            c0 = torch.zeros(self.num_layers * 2, sequence_length, self.hidden_size)

            out, (hn, cn) = self.lstm(sample, (h0, c0))
            out = self.dropout(out)
            out = self.hidden_linear(out)
            out = self.hidden(out)
            out = self.linear(out)

            return F.softmax(out, dim=1)




    #           ----        train           ----            #
    model = BiLSTM_Layer(input_size, hidden_size, num_layers, dropout, dense_input_size, dense_hidden_size,
                         dense_output_size).to(device)
    model = model.float().to(device)
    optimiser = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=regularisation)
    loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        print('epoch: ', epoch+1)

        for i in range(int(N/32)):
            model.zero_grad()
            input, targets = X[:, i*32:(i+1)*32, :], Y[:, i*32:(i+1)*32, :]
            scores = model(input.float())
            J = loss(scores, targets)
            J.backward(retain_graph=True)
            optimiser.step()
            print("batch-loss: ", J)

        model.zero_grad()
        input, targets = X[:, N-(N%32):-1, :], Y[:, N-(N%32):-1, :]
        scores = model(input.float())
        J = loss(scores, targets)
        J.backward(retain_graph=True)
        optimiser.step()
        print("batch-loss: ", J)

    with torch.no_grad():
        scores = model(X.float())

    for i in range(batch_size):
        for j in range(sequence_length):
            print(torch.argmax(targets[j][i]), torch.argmax(scores[j][i]))

    return


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