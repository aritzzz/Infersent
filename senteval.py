# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#The code is adapted from https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import, division, unicode_literals
from data_utils import *
from models.models import *
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import sys
import os
import numpy as np
import torch
from torch.autograd import Variable


PATH_TO_DATA_DAFAULT = './SentEval/data'
PATH_TO_VEC_DEFAULT = './glove.840B.300d.txt'
PATH_TO_SENTEVAL_DEFAULT = './SentEval'

sys.path.insert(0, PATH_TO_SENTEVAL_DEFAULT)
import senteval


def load_vocabulary():
    vocab = Vocab()  #load vocabulary
    with open('./vocab.json', 'r') as fp:
        vc = json.load(fp)
    vocab.vocab = vc

    vocab.build_vectors()

    return vocab



# SentEval prepare and batcher
def prepare(params, samples):
    params['vocab'] = load_vocabulary()

    model = Encoder(args.encoder, params.vocab.vectors, hidden_dim=args.hidden_dim, device=torch.device(args.device)).to(torch.device(args.device))
    checkpoint_path = os.path.join(args.save_dir, args.exp_name, 'encoder_'  + str(args.epoch_ckp) + '.pt')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_state_dict'])  
        
    params['model'] = model
    print(params['model'])    
    return

def embed_batch(batch, params):
    
    sen_lens = np.array([len(sent) for sent in batch])
    max_len = np.max(sen_lens)
    
    sentences = []
    for sent in batch:
        embed = [params['vocab'].vocab[token] if token in params['vocab'].vocab.keys() else params['vocab'].vocab['<unk>'] for token in sent]
        while len(embed) < max_len:
            embed.append(0)
        sentences.append(embed)

    return torch.tensor(sentences), torch.tensor(sen_lens)






#the format of a sentence is a list of words (tokenized and lowercased)
def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    
    #pass to batch to embed_batch() to map words to the Vocabulary
    embs, sen_lens = embed_batch(batch, params)
    batch = Variable(torch.tensor(embs).to(torch.device(args.device))) 
    with torch.no_grad():
        embeddings = params['model'].forward(batch, sen_lens)

    return embeddings.to('cpu')



# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA_DAFAULT, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

def main(args):
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    device = torch.device(args.device)
    transfer_tasks = ['STS14','MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness']
    results = se.eval(transfer_tasks)
    print(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('encoder', type=str, choices=["AWE", "LSTM", "BiLSTM", "BiLSTMpooling"],
                        help="Type of Sentence Encoder")
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help="Hidden dimensionality of the encoder you are using")
    parser.add_argument('--save_dir', type=str, default='./Models',
                        help='path to save the checkpoints')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='Name of the experiment. Checkpoints will be saved with this name')
    parser.add_argument('--epoch_ckp', type=int, default=0,
                        help='Epoch number of the checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use cpu, or cuda')

    args = parser.parse_args()

    main(args)