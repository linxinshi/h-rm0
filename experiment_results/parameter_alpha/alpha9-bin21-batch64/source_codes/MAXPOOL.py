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


class maxpool(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, opt, weights=None):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(maxpool, self).__init__()

        self.word_emb = nn.Embedding(opt.vocab_size, opt.d_word_vec, padding_idx = 0)
        if weights != None:
            self.word_emb.weight.data.copy_(torch.from_numpy(np.load(weights)))
        
    def max_score(self, inputs_d, inputs_q, mask_d, mask_q):
        q_embed = self.word_emb(inputs_q)
        d_embed = self.word_emb(inputs_d)
        
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)

        mask_d = mask_d.view(mask_d.size()[0], mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)

        q_embed_norm = q_embed_norm * mask_q
        d_embed_norm = d_embed_norm * mask_d

        q_embed_norm = q_embed_norm.permute(0, 2, 1)
        d_embed_norm = d_embed_norm.permute(0, 2, 1)
        
        maxop_q = nn.MaxPool1d(q_embed_norm.shape[2])
        maxq = maxop_q(q_embed_norm).squeeze()
        
        maxop_d = nn.MaxPool1d(d_embed_norm.shape[2])
        maxd = maxop_d(d_embed_norm).squeeze()
        
        pdist = nn.CosineSimilarity()

        output = pdist(maxq, maxd).unsqueeze(1)

        return output

    def forward(self, inputs_d, inputs_q, mask_d, mask_q, is_training=False):
        d_score = self.max_score(inputs_d, inputs_q, mask_d, mask_q)

        return d_score.squeeze()
