"""
model class
KernelPooling: the kernel pooling layer
KNRM: base class of KNRM, can choose to:
    learn distance metric
    learn entity attention
"""
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import math


class lstm(nn.Module):

    def __init__(self, opt, weights=None):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(lstm, self).__init__()

        self.word_emb = nn.Embedding(opt.vocab_size, opt.d_word_vec, padding_idx = 0)
        if weights != None:
            self.word_emb.weight.data.copy_(torch.from_numpy(np.load(weights)))

        self.bi_rnn = True
        self.n_layers = 1
        self.hidden_size = 512
        self.rnn = nn.GRU(opt.d_word_vec, self.hidden_size, self.n_layers, batch_first=True, bidirectional=self.bi_rnn)
        self.linear = nn.Linear(self.hidden_size * 2 * (2 if self.bi_rnn else 1), 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, inputs_d, inputs_q, mask_d, mask_q, is_training=False):
        q_embed = self.word_emb(inputs_q)
        d_embed = self.word_emb(inputs_d)
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        
        hidden = Variable(torch.zeros(self.n_layers * (2 if self.bi_rnn else 1), q_embed_norm.size()[0], self.hidden_size))
        hidden = hidden.cuda() if self.cuda else hidden

        inputq = q_embed_norm
        inputt = d_embed_norm
        if is_training:
            inputq = self.dropout(inputq)
            inputt = self.dropout(inputt)

        out, stateq = self.rnn(inputq, hidden)

        length_q = torch.sum(mask_q, dim=1)
        outputq = torch.sum((out * mask_q), 1) / length_q.expand((out.size()[0], out.size()[2]))

        hidden = Variable(torch.zeros(self.n_layers * (2 if self.bi_rnn else 1), d_embed_norm.size()[0], self.hidden_size))
        hidden = hidden.cuda() if self.cuda else hidden

        out, statet = self.rnn(inputt, hidden)

        length_d = torch.sum(mask_d, dim=1)
        outputt = torch.sum(out * mask_d, 1) / length_d.expand((out.size()[0], out.size()[2]))

        concated = torch.cat((outputq, outputt),1)

        output_lstm = self.linear(concated).squeeze(1)
        return output_lstm
