from __future__ import division
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import json
import pandas as pd
from tqdm import tqdm
import pickle
from models.models import *
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.test.utils import datapath, get_tmpfile

class Vocabulary(object):
    def __init__(self, dim=300, pretrained=True):
        self.embed_dim = dim
        self.pretrained = pretrained
        self.vec_path = './'
        self.embedding = 'glove.840B.300d'

    def build_vectors(self):
        #initialize the randomly initiliazed word-vectors
        fname = os.path.join(self.vec_path, self.embedding + '_aligned.pt')
        if not os.path.exists(fname):
            vocab_size = len(self.vocab)
            std = 1/torch.sqrt(torch.tensor(self.embed_dim))
            self.vectors = torch.normal(0.0, std, (vocab_size, self.embed_dim)).float()
            self.align()
            torch.save(self.vectors, fname)
        else:
            self.vectors = torch.load(fname)

    def align(self):
        if not self.pretrained:
            pass
        else:
            with open(os.path.join(self.vec_path, self.embedding +'.txt'), mode='r', encoding="utf-8") as f:
                pbar = tqdm(f)
                for line in pbar:
                    pbar.set_description("Aligning the word vectors...")
                    values = line.strip().split(" ")
                    word, vec = values[0], values[1:]
                    word_id = self.vocab.get(word, None)
                    if word_id == None:
                        continue
                    else:
                        self.vectors[word_id] = torch.tensor(list(map(float, vec)), dtype=torch.float)


class Vocab(Vocabulary):
    def __init__(self):
        Vocabulary.__init__(self)
        self.vocab = {'<pad>':0, '<unk>':1}
        self.count = {'<pad':1, '<unk':1}
        self.words = 2
    
    def Sentence(self, sentence):
        numericalized = []
        for token in word_tokenize(sentence):
            numericalized.append(self.Word(token))
        return numericalized
    
    def Word(self, token):
        if token not in self.vocab.keys():
            self.vocab[token] = self.words
            self.words+=1
            self.count[token] = 1
            return self.vocab[token]
        else:
            self.count[token] += 1
            return self.vocab[token]
    
    def filter(self, threshold=0):
        return {k:v for k, v in self.vocab.items() if self.count[k] > threshold or k in ['<pad>', '<unk>']}
    
    def __len__(self):
        return len(self.vocab)
    
    def embed(self, sentence):
        tokens = word_tokenize(sentence)
        return [self.vocab[token] if token in self.vocab.keys() else self.vocab['<unk>'] for token in tokens]


class SNLI(Dataset):
    def __init__(self, data, vocab):
        super(SNLI, self).__init__()
        self.data = data
        self.vocab = vocab
        print("total pairs {}, Vocabulary size {}".format(len(self.data), len(self.vocab)))

    def __getitem__(self, index):
        ret = self.data[index]
        return ret #ret['premise'], ret['hypothesis'], ret['label']

    def __len__(self):
        return len(self.data)
     
    @classmethod   
    def read(cls, vocab = None, path = './snli_1.0/', split='train',slice_=-1):
        if vocab == None:
            vocab = Vocab()
            flag = False    #set Flag = False to indicate that the vocab was not provided
        else:
            vocab = vocab
            flag = True
        labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        split_path = os.path.join(path, 'snli_1.0_'+ split + '.jsonl')
        data = []
        with open(split_path, 'r') as f:
            lines = f.readlines()[:slice_]
        pbar = tqdm(lines)
        for line in pbar:
            pbar.set_description("Reading and Preparing dataset...")
            line = json.loads(line)
            #print(line)
            label_ = line['gold_label']
            if label_ not in labels.keys():
                continue
            premise = line['sentence1']
            hypothesis = line['sentence2']
            data.append(
                        {'label':SNLI.preprocess(label_, label=True),
                        'premise':SNLI.preprocess(premise, vocab, flag),
                        'hypothesis':SNLI.preprocess(hypothesis, vocab, flag)}
                        )
        return cls(data, vocab)
    
    @staticmethod
    def preprocess(sentence, vocab=None, flag=None, label=False):
        if not flag and not label:
            return torch.LongTensor(vocab.Sentence(sentence.lower()))
        elif flag and not label:
            return torch.LongTensor(vocab.embed(sentence))
        else:
            labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
            return labels[sentence]


def collater(batch):
    premise = pad_sequence([item['premise'] for item in batch], padding_value=0)
    hypothesis = pad_sequence([item['hypothesis'] for item in batch], padding_value=0)
    label = [item['label'] for item in batch]
    return {'premise': premise, 'hypothesis': hypothesis, 'label':label}



def get_data_loaders(path='./snli_1.0/', batch_size=32, slice_=-1):
    train = SNLI.read(path=path, split='train', slice_=slice_)
    vocab = train.vocab
    dev = SNLI.read(path=path, split='dev', vocab=vocab, slice_=slice_)
    test = SNLI.read(path=path, split='test', vocab=vocab, slice_=slice_)
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=collater, drop_last=False)
    dev_loader = DataLoader(dataset=dev, collate_fn=collater, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=dev, collate_fn=collater, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, dev_loader, test_loader, vocab

    
if __name__ == "__main__":
    trainloader, devloader, testloader, vocab = get_data_loaders(batch_size=4, slice_=1000)
    vocab.build_vectors()
    model = SNLInet("BiLSTM Pooling", vocab.vectors)
    print(model)
    for batch in testloader:
        premise, hypothesis, label = batch['premise'], batch['hypothesis'], batch['label']
        print(premise.shape)
        out = model(premise.T, hypothesis.T)
        break
