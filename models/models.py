import torch
import torch.nn as nn

class AWE(nn.Module):
    def __init__(self, emb_dim):
        super(AWE, self).__init__()
        self.emb_dim = emb_dim

    def forward(self, embed, seq_lens=None):
        return torch.mean(embed, dim=1)
    
    @property
    def output_dim(self):
        return self.emb_dim

class UniLSTM(nn.Module):
    def __init__(self, emb_dim=None, hidden_dim=None):
        super(UniLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, batch_first=True)
    
    def forward(self, embed, seq_lens=None):
        packed_seq_batch = nn.utils.rnn.pack_padded_sequence(embed, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        out, (h,c) = self.lstm(packed_seq_batch.float())
        return h.squeeze()
    
    @property
    def output_dim(self):
        return self.hidden_dim
    
class BiLSTM(nn.Module):
    def __init__(self, emb_dim=None, hidden_dim=None, pooling=False):
        super(BiLSTM, self).__init__()
        self.pool = pooling
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, batch_first=True, bidirectional=True)
    
    def forward(self, embed, seq_lens=None):
        packed_seq_batch = nn.utils.rnn.pack_padded_sequence(embed, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        out, (h,c) = self.bilstm(packed_seq_batch.float())
        if not self.pool:
            hc = torch.cat((h[0,:,:], h[1,:,:]), dim=1)
            return hc
        else:
            padded_output, output_lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=torch.max(seq_lens))
            padded_output[padded_output == 0.0] = -1e9  #shape = (bsz, seq_len, 2*hidden_dim)
            out,_ = torch.max(padded_output, dim=1)
            return out
    @property
    def output_shape(self):
        return self.emb_dim*2


class Encoder(nn.Module):
    def __init__(self, init_type=None, embeddings_init=None):
        super(Encoder, self).__init__()
        self.init = init_type
        self.pretrained_embeddings = embeddings_init
        self.Embed = nn.Embedding.from_pretrained(self.pretrained_embeddings, padding_idx=0)
        if self.init == "AWE":
            self.encoder = AWE(self.pretrained_embeddings.shape[1])
        if self.init == "Unidirectional LSTM":
            self.encoder = UniLSTM(self.pretrained_embeddings.shape[1], 300)
        if self.init == "Bidirectional LSTM":
            self.encoder = BiLSTM(self.pretrained_embeddings.shape[1], 300)
        if self.init == "BiLSTM Pooling":
            self.encoder = BiLSTM(self.pretrained_embeddings.shape[1], 300, pooling=True)
            

    def forward(self, sentence): #shape = (bsz, seq_len)
        seq_lens = torch.sum(torch.ge(sentence, torch.tensor([1])).int(), dim=1)
        embedded = self.Embed(sentence)
        return self.encoder(embedded, seq_lens=seq_lens)
    
    @property
    def output_shape(self):
        return self.encoder.output_shape

class SNLInet(nn.Module):
    def __init__(self, encoder_init, embeddings_init):
        super(SNLInet, self).__init__()
        self.encoder_init = encoder_init
        self.embeddings_init = embeddings_init
        self.encoder = Encoder(init_type=self.encoder_init, embeddings_init=self.embeddings_init)
        self.inp_dim = 4*self.encoder.output_shape
        self.num_classes = 3
        self.classifier = nn.Sequential(
                                    nn.Linear(self.inp_dim, 512),
                                    nn.Linear(512, self.num_classes)
                                    )

    def forward(self, premise, hypothesis):
        prem = self.encoder(premise)
        hypo = self.encoder(hypothesis)
        out = torch.cat((prem, hypo, torch.abs(prem-hypo), prem*hypo), dim=1)
        pred = self.classifier(out)
        return pred

    



    