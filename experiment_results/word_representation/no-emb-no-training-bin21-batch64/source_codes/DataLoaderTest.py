''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable

def cover_text2int(sentence):
    tokens = sentence.strip().split(",")
    return [int(token) for token in tokens]

class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, data,
            cuda=True, batch_size=64, test=False,device='cpu'):
        self.device=device
        #print ('device',device,self.device)
        self.cuda = cuda
        self.test = test
        
        f = open(data)
        count = 0
        for count, _ in enumerate(f):
            pass
        count += 1
        f.close()

        self.length = count

        self._n_batch = int(np.ceil(self.length / batch_size))

        self._batch_size = batch_size

        self._iter_count = 0

        self.data_file = data
        self.data = open(data)

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts, max_len):
            ''' Pad the instance to the max seq length in batch '''
            inst_data = np.array([
                inst[:max_len] + [0] * (max_len - len(inst[:max_len]))
                for inst in insts])
            mask = np.zeros((inst_data.shape[0], inst_data.shape[1]))
            for b in range(len(inst_data)):
                for i in range(len(inst_data[b])):
                    if inst_data[b, i] > 0:
                        mask[b, i] = 1
            
            #mask_tensor = Variable(torch.FloatTensor(mask), requires_grad = False)
            # with torch.no_grad
            #inst_data_tensor = Variable(torch.LongTensor(inst_data))
            #inst_data_tensor = Variable(torch.LongTensor(inst_data), volatile=self.test)
            # wrong inst_data_tensor = Variable(torch.LongTensor(inst_data), requires_grad=self.test)
            if self.cuda:
                #mask_tensor = mask_tensor.cuda()
                #mask_tensor.to(self.device)
                #inst_data_tensor = inst_data_tensor.cuda()
                #inst_data_tensor.to(self.device)
                mask_tensor=torch.cuda.FloatTensor(mask,device=self.device)
                inst_data_tensor=torch.cuda.LongTensor(inst_data,device=self.device)
            else:
                mask_tensor = torch.FloatTensor(mask)
                inst_data_tensor = torch.LongTensor(inst_data)
            mask_tensor.requires_grad_(False)
            #inst_data_tensor.requires_grad_(self.test)
            #print (self.device)
            #print (mask_tensor)
            return inst_data_tensor, mask_tensor

        if self._iter_count < self._n_batch:

            q_list = []
            doc_list = []
            qid_list = []
            did_list = []

            while True:
                self._iter_count += 1
                for i in range(self._batch_size):
                    line = self.data.readline().strip().split('\t')
                    if len(line) < 4:
                        continue
                        
                    query = line[2]
                    doc = line[3]
                    
                    query = cover_text2int(query)
                    doc = cover_text2int(doc)
                    # print(len(query), len(pos), len(neg))
                    if sum(query) == 0 or sum(doc) == 0:
                        continue
                        
                    q_list.append(query)
                    doc_list.append(doc)
                    qid_list.append(line[0])
                    did_list.append(line[1])
                if len(q_list) != 0:
                    break

            inst_q, mask_q = pad_to_longest(q_list, 20)
            inst_d, mask_d = pad_to_longest(doc_list, 200)
            return inst_q, inst_d, mask_q, mask_d, did_list, qid_list

        else:

            self._iter_count = 0
            self.data = open(self.data_file)
            raise StopIteration()
