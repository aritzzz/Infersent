from __future__ import division
import argparse
from models.models import *
from data_utils import *
from collections import defaultdict


class Metrics(object):
    def __init__(self):
        pass

    def accuracy(self, pred, actual):
        return 0.0

class Recorder(object):
    def __init__(self):
        self.record_obj = defaultdict(lambda:0.0)
        self.iterations=0
    
    def record(self, val_dict):
        for key in val_dict.keys():
            self.record_obj[key]+=val_dict[key]
        self.update()

    def update(self):
        self.iterations+=1
    
    def mean(self):
        for key, val in self.record_obj.items():
            self.record_obj[key] = val/self.iterations


class Trainer(object):
    def __init__(self,
                model=None,
                criterion=None,
                optimizer=None,
                train_loader=None,
                val_loader=None,
                epochs=20,
                learning_rate=0.001,
                device=torch.device("cpu"),
                log_after=1,
                ):
        self.model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_after = log_after
        self.recorder = Recorder()
        self.metrics = Metrics()
    
    def train_batch(self, batch):
        self.model.zero_grad()
        premise, hypothesis, label = map(self._to_device, (batch['premise'], batch['hypothesis'], batch['label']))
        out = self.model(premise.T, hypothesis.T)
        loss = self.criterion(out, label)
        acc = self.metrics.accuracy(out, label)
        self.recorder.record(
                            {'loss': loss.item(), 
                            'accuracy': acc }
                            )
        loss.backward()
        self.optimizer.step()



    def train(self):
        self._model_to_device()
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
        

    def _train_epoch(self, epoch):
        for itr, batch in enumerate(self.train_loader):
            self.train_batch(batch)
            
            if itr % self.log_after == 0:
                print("itr {}:, {}".format(itr, dict(self.recorder.record_obj)))
                self.recorder = Recorder()
            

    def _to_device(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.tensor(inp)
        return inp.to(self.device)

    def _model_to_device(self):
        self.model = self.model.to(self.device)

    def evaluate(self):
        pass

    def _save_checkpoint(self):
        pass

    def _update_learning_rate(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr']*factor








def main(args):
    train_loader, val_loader, test_loader, vocab = get_data_loaders(batch_size=args.batch_size, slice_=1000)
    vocab.build_vectors()
    model = SNLInet(args.encoder, vocab.vectors, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    learning_rate=args.lr,
                    device = torch.device(args.device),
                    epochs=args.epochs
                     )
    trainer.train()





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

    args = parser.parse_args()
    main(args)
