from __future__ import division
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
from models.models import *
from data_utils import *
from collections import defaultdict
import  numpy as np
from torch.utils.tensorboard import SummaryWriter


class Metrics(object):
    def __init__(self):
        pass

    def accuracy(self, pred, actual):
        acc = torch.sum((torch.argmax(pred, dim=1) == actual).float())
        acc /= pred.shape[0]
        return acc

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
            self.record_obj[key] = np.round(val/self.iterations, decimals=4) 


class Trainer(object):
    def __init__(
                self,
                model=None,
                criterion=None,
                optimizer=None,
                train_loader=None,
                val_loader=None,
                epochs=20,
                learning_rate=0.001,
                device=torch.device("cpu"),
                log_after=1,
                checkpoint_path=None,
                tensorboard_writer=None,
                test_loader=None,
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
        self.best_val_acc = -float('inf')
        self.stop_training = False
        self.checkpoint_path = checkpoint_path
        self.tensorboard_writer = tensorboard_writer
        self.test_loader = test_loader
    
    def train_batch(self, batch):
        self.model.zero_grad()
        premise, hypothesis, label = batch['premise'], batch['hypothesis'], self._to_device(batch['label'])
        premise = premise.T
        hypothesis = hypothesis.T
        p_len = torch.sum(torch.ge(premise, torch.tensor([1])).int(), dim=1)
        h_len = torch.sum(torch.ge(hypothesis, torch.tensor([1])).int(), dim=1)
        try:
             out = self.model((self._to_device(premise), p_len), (self._to_device(hypothesis), h_len))
             loss = self.criterion(out, label)
             acc = self.metrics.accuracy(out, label)
             self.recorder.record(
                            {'loss': loss.item(), 
                            'accuracy': acc.item() }
                            )
             loss.backward()
             self.optimizer.step()
        except Exception:
             print(p_len)
             print(h_len)
             print(premise)
             print(hypothesis)



    def train(self):
        self._model_to_device()
        epoch = 0
        while epoch <= self.epochs and not self._early_stop():
            self._train_epoch(epoch)
        
            val_metrics = self.evaluate()
            print("Validation Loss {}, Accuracy {}".format(val_metrics['loss'], val_metrics['accuracy']))
            if val_metrics['accuracy'] >= self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self._save_checkpoint(epoch)
            else:
                self._update_learning_rate(0.2)
            epoch+=1

    def _train_epoch(self, epoch):
        for itr, batch in enumerate(self.train_loader):
            self.train_batch(batch)
            
            if itr % self.log_after == 0:
                self.recorder.mean()
                print("itr {} Training : {}".format(itr + epoch*len(self.train_loader), dict(self.recorder.record_obj)))
                self.tensorboard_writer.add_scalar('training loss', self.recorder.record_obj['loss'], epoch*len(self.train_loader)+itr)
                self.recorder = Recorder()
                val_metrics = self.evaluate()
                print("itr {} Validation : {} ".format(itr + epoch*len(self.train_loader), val_metrics))
                self.tensorboard_writer.add_scalar('validation loss', val_metrics['loss'], epoch*len(self.train_loader)+itr)
                
            

    def _to_device(self, inp):
        if not torch.is_tensor(inp):
            inp = torch.tensor(inp)
        return inp.to(self.device)

    def _model_to_device(self):
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self):
        val_recorder = Recorder()
        for _, batch in enumerate(self.val_loader):
            premise, hypothesis, label = batch['premise'], batch['hypothesis'], self._to_device(batch['label'])
            premise = premise.T
            hypothesis = hypothesis.T
            p_len = torch.sum(torch.ge(premise, torch.tensor([1])).int(), dim=1)
            h_len = torch.sum(torch.ge(hypothesis, torch.tensor([1])).int(), dim=1)
            out = self.model((self._to_device(premise), p_len), (self._to_device(hypothesis), h_len))
            acc = self.metrics.accuracy(out, label)
            val_recorder.record({'accuracy': acc.item()})
        val_recorder.mean()
        return val_recorder.record_obj

    def _save_checkpoint(self, epoch):
        print("Saving the model at epoch {}".format(epoch))
        whole_ckp = {'model_state_dict':self.model.state_dict(),
                'optimizer_state_dict':self.optimizer.state_dict()}
        torch.save(whole_ckp, os.path.join(self.checkpoint_path, 'model_' +str(epoch)+'.pt' ))

        encoder_ckp = {'model_state_dict':self.model.encoder.state_dict()}
        torch.save(encoder_ckp, os.path.join(self.checkpoint_path, 'encoder_' + str(epoch) + '.pt'))

    def _update_learning_rate(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr']*factor
    
    @property
    def _current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    
    def _early_stop(self):
        if self._current_learning_rate <= 1e-5:
            self.stop_training = True
        return self.stop_training

    @classmethod
    def Initialize(cls, encoder=None, checkpoint_path=None, val_loader=None, test_loader=None, vocab=None, hidden_dim=None, device=None):
        vocab.build_vectors()
        model = SNLInet(encoder, vocab.vectors, hidden_dim=hidden_dim, device=torch.device(device)).to(torch.device(device))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(
                    model=model,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=torch.device(device),
                    )

    def evaluate_on_test_data(self):
        test_recorder = Recorder()
        for _, batch in enumerate(self.test_loader):
            premise, hypothesis, label = batch['premise'], batch['hypothesis'], self._to_device(batch['label'])
            premise = premise.T
            hypothesis = hypothesis.T
            p_len = torch.sum(torch.ge(premise, torch.tensor([1])).int(), dim=1)
            h_len = torch.sum(torch.ge(hypothesis, torch.tensor([1])).int(), dim=1)
            out = self.model((self._to_device(premise), p_len), (self._to_device(hypothesis), h_len))
            acc = self.metrics.accuracy(out, label)
            test_recorder.record({'accuracy': acc.item()})
        test_recorder.mean()
        return test_recorder.record_obj









def main(args):

    ckp_path = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(ckp_path, exist_ok=True)

    tensorboard_writer = SummaryWriter('runs/' + args.exp_name)

    train_loader, val_loader, test_loader, vocab = get_data_loaders(batch_size=args.batch_size, slice_=args.slice)
    vocab.build_vectors()
    model = SNLInet(args.encoder, vocab.vectors, hidden_dim=args.hidden_dim, device=torch.device(args.device))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    learning_rate=args.lr,
                    device = torch.device(args.device),
                    epochs=args.epochs,
                    checkpoint_path=ckp_path,
                    tensorboard_writer=tensorboard_writer,
                    log_after=args.log_after
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
    parser.add_argument('--save_dir', type=str, default='./Models',
                        help='path to save the checkpoints')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='Name of the experiment. Checkpoints will be saved with this name')
    parser.add_argument('--slice', type=int, default=-1, help='handy arg')
    parser.add_argument('--log_after', type=int, default=1)

    args = parser.parse_args()
    main(args)
