"""
model class
KernelPooling: the kernel pooling layer
HRM: base class of HRM, can choose to:
    learn distance metric
    learn entity attention
"""
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from torch import acos,clamp,Tensor

def get_idf(vocab_size):
    UNKNOWN_TOKEN = '<UNK>'
    PAD_TOKEN = '<PAD>'
    vocab_dict = dict()
    
    with open('../data/vocab.tsv',mode='r') as vof:
        for line in vof:
            word = line.strip('\n')
            wd = word.split('\t')[0]
            id = int(word.split('\t')[1])
            vocab_dict[wd] = id
    vocab_dict[UNKNOWN_TOKEN] = 0
    vocab_dict[PAD_TOKEN] = 0

    idf_file = '../data/idf.norm.tsv'
    attn = [0] * vocab_size
    with open(idf_file,'r') as idf_src:
        for line in idf_src:
            line = line.strip().split('\t')
            if line[0] in vocab_dict:
                attn[vocab_dict[line[0]]] = float(line[1])
    #print (len(attn))
    #print (attn[0:10])
    return np.array(attn)

class hrm(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, opt, weights=None):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(hrm, self).__init__()

        #self.dropout = nn.Dropout(0.2)

        self.word_emb = nn.Embedding(opt.vocab_size, opt.d_word_vec, padding_idx = 0)
        if weights != None:
            self.word_emb.weight.data.copy_(torch.from_numpy(np.load(weights)))
        self.attn = nn.Embedding(opt.vocab_size, 1)

        #temp=torch.from_numpy(get_idf(opt.vocab_size))
        #[315370]
        #print ('temp shape',temp.shape)
        #[315370]x[1]
        #print ('attn weight shape',self.attn.weight.data.shape)

        self.attn.weight.data.copy_(torch.from_numpy(get_idf(opt.vocab_size)).unsqueeze(1))
        self.attn.weight.requires_grad = False
        tensor_mu = torch.FloatTensor(opt.mu)
        tensor_sigma = torch.FloatTensor(opt.sigma)
        #tensor_alpha = torch.FloatTensor(2.0)
        if opt.cuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
            #tensor_alpha=tensor_alpha.cuda()
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, opt.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, opt.n_bins)
        self.alpha = torch.tensor(1.0,requires_grad=False).cuda()
        self.dense = nn.Linear(opt.n_bins, 1, 1)
        self.idf_dense = nn.Linear(1, 1)
        
        

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):

        #print (q_embed.shape,d_embed.shape)
        #print (q_embed.size(),d_embed.size())
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1) # n*m*d*1
        #print (sim.size())
        #pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        #trans_sim=1-acos(sim.clamp(-1.0+1e-7,1.0-1e-7))/(np.pi)
        trans_sim=acos(sim.clamp(-1.0+1e-7,1.0-1e-7))/(np.pi)
        #in_sim=sim*trans_sim
        #print (sim)
        #ts=torch.zeros(trans_sim.shape[0],trans_sim.shape[1],1,requires_grad=True).cuda()
        # two way implement to test. torch.div
        temp_sim=torch.exp(torch.clamp(self.alpha,-5.0,5.0)*trans_sim)
        temp_sim2=trans_sim*temp_sim
        #print (temp_sim2[0,0].shape)
        #print (temp_sim2[0,0].sum().shape)
        
        '''
        for i in range(trans_sim.shape[0]):
            for j in range(trans_sim.shape[1]):
                print (temp_sim2[i,j].sum().shape)
                ts[i,j,0]=torch.div(temp_sim2[i,j].sum(),temp_sim[i,j].sum())
        '''
        ts=torch.div(torch.sum(temp_sim2,2),torch.sum(temp_sim,2))
        #print (ts.shape)

        #print (self.mu.shape,self.mu.view(1,1,-1).shape,attn_d.shape)
        #print (torch.exp((- ((trans_sim - self.mu) ** 2) / (self.sigma ** 2) / 2)).shape)
        #pooling_value = torch.exp((- ((trans_sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        # no attn_d for pv2
        
        #print (pooling_value2.shape)
        #pooling_value = torch.exp((- ((torch.clamp(in_sim,min=1e-10) - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        '''
        attn_d        ([64,1,200,1])
        sim torch.Size([64, 20, 200, 1])
        ts  torch.Size([64, 20, 1])    
        # cuz 21 kernels
        pv torch.Size([64, 20, 200, 21])
        ps torch.Size([64, 20, 21])
        lps1 torch.Size([64, 20, 21])
        lps2 torch.Size([64, 21])
        path selection:  smooth maximum first or later
        '''
        #print ('sim',sim.shape)
        #print ('pv',pooling_value.shape)
        #pooling_sum = torch.sum(pooling_value, 2)
        pooling_sum = torch.exp((- ((ts - self.mu.view(1,1,-1)) ** 2) / (self.sigma.view(1,1,-1) ** 2) / 2)) 
        #print ('ps',pooling_sum.shape)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q * 0.01
        #print ('lps1',log_pooling_sum.shape)
        log_pooling_sum = torch.sum(log_pooling_sum, 1)#soft-TF
        #print ('lps2',log_pooling_sum.shape)
        return log_pooling_sum


    def forward(self, inputs_q, inputs_d, mask_q, mask_d, is_train=False):
        attn_q = self.idf_dense(self.attn(inputs_q).view(inputs_q.shape[0]*inputs_q.shape[1], 1).view(inputs_q.shape[0], inputs_q.shape[1], 1))
        q_embed = self.word_emb(inputs_q)# output embedding to q_embed
        d_embed = self.word_emb(inputs_d)
        # if is_train:
        #     q_embed = self.dropout(q_embed)
        #     d_embed = self.dropout(d_embed)
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)
        #print (mask_d.shape)
        # mask_d  [64,200]
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q * attn_q, mask_d)
        #log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q , mask_d)
        output = torch.squeeze(torch.tanh(self.dense(log_pooling_sum)), 1)
        return output
