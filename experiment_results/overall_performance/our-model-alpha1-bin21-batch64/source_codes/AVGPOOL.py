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

def get_idf(vocab_size):
    UNKNOWN_TOKEN = '<UNK>'
    PAD_TOKEN = '<PAD>'
    vocab_dict = dict()
    vof=open('../data/vocab.tsv',mode='r')
    for line in vof:
        word = line.strip('\n')
        wd = word.split(' ')[0]
        id = int(word.split(' ')[1])
        vocab_dict[wd] = id
    vocab_dict[UNKNOWN_TOKEN] = 0
    vocab_dict[PAD_TOKEN] = 0

    idf_file = '../data/idf.norm.tsv'
    attn = [0] * vocab_size
    for line in open(idf_file):
        line = line.strip().split('\t')
        if line[0] in vocab_dict:
            attn[vocab_dict[line[0]]] = float(line[1])
    
    return np.array(attn)

class avgpool(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, opt, weights=None):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(avgpool, self).__init__()

        self.word_emb = nn.Embedding(opt.vocab_size, opt.d_word_vec, padding_idx = 0)
        if weights != None:
            self.word_emb.weight.data.copy_(torch.from_numpy(np.load(weights)))
        self.attn = nn.Embedding(opt.vocab_size, 1)
        self.attn.weight.data.copy_(torch.from_numpy(get_idf(opt.vocab_size)))
        self.attn.weight.requires_grad = False
        
    def avg_score(self, inputs_d, inputs_q, mask_d, mask_q):
        attn_q = self.attn(inputs_q)
        attn_d = self.attn(inputs_d)

        q_embed = self.word_emb(inputs_q)
        d_embed = self.word_emb(inputs_d)
        
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)

        mask_d = mask_d.view(mask_d.size()[0], mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)

        length_q = torch.sum(mask_q, dim=1)

        avgq = torch.sum(q_embed_norm * mask_q * attn_q, 1) / length_q.view(length_q.shape[0], 1)

        length_d = torch.sum(mask_d, dim=1)
        avgd = torch.sum(d_embed_norm * mask_d * attn_d, 1) / length_d.view(length_d.shape[0], 1)
        
        pdist = nn.CosineSimilarity()

        output = pdist(avgq, avgd).unsqueeze(1)

        return output

    def forward(self, inputs_d, inputs_q, mask_d, mask_q, is_training=False):
        d_score = self.avg_score(inputs_d, inputs_q, mask_d, mask_q)

        return d_score.squeeze()
