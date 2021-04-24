from __future__ import division
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
from models.models import *
from data_utils import *
from collections import defaultdict
from train import *
import  numpy as np
import json

def get_data_loaders(path='./snli_1.0/', batch_size=32, slice_=-1):
    vocab = Vocab()
    with open('./vocab.json', 'r') as fp:
    	vc = json.load(fp)
    vocab.vocab = vc
    dev = SNLI.read(path=path, split='dev', vocab=vocab, slice_=slice_)
    test = SNLI.read(path=path, split='test', vocab=vocab, slice_=slice_)
    dev_loader = DataLoader(dataset=dev, collate_fn=collater, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=dev, collate_fn=collater, batch_size=batch_size, shuffle=False, drop_last=False)

    return dev_loader, test_loader, vocab




def main(args):
	val_loader, test_loader, vocab = get_data_loaders()
	ckp_path = os.path.join(args.save_dir, args.exp_name, 'model_' + str(args.epoch_ckp) + '.pt')
	evaluator = Trainer.Initialize(encoder=args.encoder,
									val_loader=val_loader,
									test_loader=test_loader,
									vocab = vocab,
									hidden_dim=args.hidden_dim,
									device=torch.device(args.device),
									checkpoint_path=ckp_path
									)

	val_recorder = evaluator.evaluate()
	test_recorder = evaluator.evaluate_on_test_data()

	print("VALIDATION {}".format(dict(val_recorder)))
	print("TEST {}".format(dict(test_recorder)))




if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('encoder', type=str, choices=["AWE", "LSTM", "BiLSTM", "BiLSTMpooling"],
						help="Type of Sentence Encoder")
	parser.add_argument('--data_dir', type=str, default='./data/snli_1.0/',
						help="Path to the SNLI dataset")
	parser.add_argument('--pretrained_vec_path', type=str, default='./data/',
						help="Path to the pretrained embedding vectors")
	parser.add_argument('--batch_size', type=int, default=64,
						help="Batch size to train the model")
	parser.add_argument('--hidden_dim', type=int, default=2048,
						help="Hidden Layer Dimension for encoder models")
	parser.add_argument('--device', type=str, default="cpu")
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--lr', type=float, default=0.1)
	parser.add_argument('--save_dir', type=str, default='./Models',
						help='path to save the checkpoints')
	parser.add_argument('--exp_name', type=str, default='default',
						help='Name of the experiment. Checkpoints will be saved with this name')
	parser.add_argument('--epoch_ckp', type=int, default=0,
						help='Epoch number of the checkpoint')
	parser.add_argument('--slice', type=int, default=-1, help='handy arg')
	parser.add_argument('--log_after', type=int, default=1)

	args = parser.parse_args()
	main(args)