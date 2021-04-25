import torch
from models.models import *
from data_utils import *
import json
import argparse

BASE_PATH = './Models'

CONFIG = {'hidden_dim': {'LSTM':2048, 'BiLSTM':4096, 'BiLSTMpooling':4096},
		'vocab_path': './vocab.json',
		'device': 'cpu',
		'checkpoint': {'LSTM': 13, 'BiLSTM': 12, 'BiLSTMpooling': 5},
		'model_path': {'LSTM': os.path.join(BASE_PATH, 'default'),
						'BiLSTM':os.path.join(BASE_PATH, 'bilstm'),
						'BiLSTMpooling':os.path.join(BASE_PATH, 'bilstmpooling')}}


class Predictor(object):
	def __init__(self, model=None, vocab=None):
		self.model = model
		self.vocab = vocab


	@classmethod
	def Initialize(cls, encoder):
		#we will use the best checkpoint, and use only cpu
		vocab = Vocab()
		with open(CONFIG["vocab_path"], 'r') as fp:
			vc = json.load(fp)
		vocab.vocab = vc

		vocab.build_vectors()

		checkpoint_path = os.path.join(CONFIG['model_path'][encoder], 'model_' + str(CONFIG['checkpoint'][encoder]) + '.pt')
		model = SNLInet(encoder, vocab.vectors, hidden_dim=CONFIG['hidden_dim'][encoder]).to(torch.device(CONFIG['device']))
		checkpoint = torch.load(checkpoint_path, map_location=torch.device(CONFIG['device']))
		model.load_state_dict(checkpoint['model_state_dict'])
		return cls(model = model, vocab = vocab)

	def prepare(self, sentence):
		return self.vocab.embed(sentence)

	@torch.no_grad()
	def predict(self, premise, hypothesis):
		premise, hypothesis = torch.tensor(self.prepare(premise)), torch.tensor(self.prepare(hypothesis))
		device = torch.device(CONFIG['device'])
		premise = premise.T.unsqueeze(0).to(device)
		hypothesis = hypothesis.T.unsqueeze(0).to(device)
		p_len = torch.tensor([premise.shape[1]])
		h_len = torch.tensor([hypothesis.shape[1]])
		prem = self.model.encoder(premise, p_len)
		hypo = self.model.encoder(hypothesis, h_len)
		if prem.ndim == 1:
			prem = prem.unsqueeze(0)
		if hypo.ndim == 1:
			hypo = hypo.unsqueeze(0)
		out = torch.cat((prem, hypo, torch.abs(prem-hypo), prem*hypo), dim=1)
		pred = self.model.classifier(out)
		label = self.map(torch.argmax(pred))

		return label

	def map(self, argm):
		labels = { 0: 'neutral', 1: 'entailment', 2: 'contradiction'}
		return labels[argm.item()]


def main(args):
	predictor = Predictor.Initialize(args.encoder)
	pred_label = predictor.predict(args.premise, args.hypothesis)

	print("The prediction is {}".format(pred_label))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('encoder', type=str, choices=['AWE', "LSTM", "BiLSTM", "BiLSTMpooling"])
	parser.add_argument('--premise', type=str, default="I am sleeping in my bedroom.")
	parser.add_argument('--hypothesis', type=str, default="I am in my classroom.")

	args = parser.parse_args()
	main(args)