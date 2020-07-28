#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
import os,sys,logging,argparse,pickle,csv
import KNRM,CKNRM,HRM,SRM
import AVGPOOL,MAXPOOL
import LSTM

import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch.autograd import Variable
from DataLoader import DataLoader
from DataLoaderTest import DataLoaderTest
from utils import *

from io_object import IO_Object

torch.manual_seed(3)

def data_forward(model, forward_data, output_dir, raw_output):
    result_dict=dict()
    writer = open(output_dir,'w')
    fout = open(raw_output, 'w')
    for idx, batch in enumerate(forward_data):
        if idx % 1000 == 0:
            print(idx)
        inputs_q, inputs_d, mask_q, mask_d, docid, qid = batch
        model.eval()
        output = model(inputs_q,inputs_d,mask_q,mask_d)
        output = output.data.tolist()
        # fout.write(str(qid) + '\t' + str(docid) + '\t' + str(output) + '\n')
        tuples = zip(qid, docid, output)
        for item in tuples:
            fout.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')
            if item[0] not in result_dict:
                result_dict[item[0]]=[]
            result_dict[item[0]].append((item[1], item[2])) #{ id: [output] }

    qrels=get_qrels('../data/qrels.dev.tsv')
    no_label = 0
    c_1_j = 0
    c_2_j = 0
    reduce_num = 0
    for qid, value in result_dict.items():
        if qid not in qrels:
            no_label += 1
            continue
        res = sorted(value, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
        count = 0.0
        score = 0.0
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:#if docid in this qrel[qid]'s docid list(which means it is relevant)
                count += 1
                score += count / (i+1) # + pos doc number/total doc num
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:
                c_2_j += 1 / float(i+1)
                break
        if count != 0:
            c_1_j += score / count
        else: # a question without pos doc
            reduce_num += 1

    print(len(result_dict), reduce_num)
    MAP = c_1_j / float(len(result_dict) - no_label)
    MRR = c_2_j / float(len(result_dict) - no_label) #
    #print ""
    #print(" evaluate on " + flag + " MAP: %f" % MAP)
    #print(" evaluate on " + flag + ' MRR: %f' % MRR)
    print(" evaluate on " + " MAP: %f" % MAP)
    print(" evaluate on " + " MRR: %f" % MRR)


    for qid, values in result_dict.items():
        res = sorted(values, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
        for rank,value in enumerate(res):
            writer.write(str(qid)+'\t'+str(value[0])+'\t'+str(rank+1)+'\n')
    # output results:
    print('len of scored dict:',len(result_dict))

def data_evaluate(model, evaluate_data, flag, qrels):
    eval_dict = dict()
    c_1_j = 0
    c_2_j = 0
    reduce_num = 0
    with torch.no_grad():
         for batch in evaluate_data:
             inputs_q, inputs_d, mask_q, mask_d, docid, qid = batch
             model.eval()
             outputs = model(inputs_q, inputs_d, mask_q, mask_d)
             output = outputs.cpu().data.tolist()
             # print(outputs)
             # output = outputs.data.tolist()
             tuples = zip(qid, docid, output)
             for item in tuples:
                 if item[0] not in eval_dict: # id not in eval dict
                    eval_dict[item[0]] = []
                 eval_dict[item[0]].append((item[1], item[2])) # {id: [(docid, output)]}
             
             #break

    no_label = 0
    eval_dict_out={}
    for qid, value in eval_dict.items():
        if qid not in qrels:
            no_label += 1
            continue
        res = sorted(value, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
        eval_dict_out[qid]=res
        count = 0.0
        score = 0.0
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:#if docid in this qrel[qid]'s docid list(which means it is relevant)
                count += 1
                score += count / (i+1) # + pos doc number/total doc num
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:
                c_2_j += 1 / float(i+1)
                break
        if count != 0:
            c_1_j += score / count
        else: # a question without pos doc
            reduce_num += 1

    #print(len(eval_dict), no_label)
    MAP = c_1_j / float(len(eval_dict) - no_label)
    MRR = c_2_j / float(len(eval_dict) - no_label) #
    #print ""
    #print(" evaluate on " + flag + " MAP: %f" % MAP)
    #print(" evaluate on " + flag + ' MRR: %f' % MRR)
    logging.info(" evaluate on " + flag + " MAP: %f" % MAP)
    logging.info(" evaluate on " + flag + ' MRR: %f' % MRR)
    return MAP, MRR, eval_dict_out

def train(model, opt, crit, optimizer, train_data, dev_data, test_data, io_obj):
    ''' Start training '''
    step = 0
    best_map_dev = 0.0
    best_mrr_dev = 0.0
    best_map_test = 0.0
    best_mrr_test = 0.0
    best_ranking_dev=None
    best_ranking_test=None
    qrels=get_qrels('../data/qrels.dev.tsv')
    alphas=[]
    for epoch_i in range(opt.epoch):
        total_loss = 0.0
        time_epstart=datetime.now()
        for batch in train_data:
            # prepare data
            inputs_q, inputs_d_pos, inputs_d_neg, mask_q, mask_d_pos, mask_d_neg = batch
            # forward
            optimizer.zero_grad()
            model.train()
            outputs_pos = model(inputs_q, inputs_d_pos, mask_q, mask_d_pos)
            outputs_neg = model(inputs_q, inputs_d_neg, mask_q, mask_d_neg)
            #label = torch.ones(outputs_pos.size())#[1,1,1,1...]
            if opt.cuda:
                #label = label.cuda()
                #label.to('cuda:%d'%(opt.device))
                label=torch.ones(outputs_pos.size(),device='cuda:%d'%(opt.device))
            else:
                label=torch.ones(outputs_pos.size())
            label.requires_grad_(False)
            #batch_loss = crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))
            batch_loss = crit(outputs_pos, outputs_neg, label)
            if opt.task=='SRM':
               alphas.append(model.alpha.item())
            # backward
            batch_loss.backward()

            # update parameters
            optimizer.step()
            step += 1
            
            #total_loss += batch_loss.data[0]
            total_loss += batch_loss.data.item()
            #if(step>=14000 and opt.eval_step!= 200):
            #    opt.eval_step=200 # make it smaller 2000--->200
            # opt.eval_step= 10
            if opt.is_ensemble:
                if step > 60000:
                    break
            # opt.eval_step
            if step % opt.eval_step == 0: 
                time_step=datetime.now()-time_epstart
                print(' Epoch %d Training step %d loss %f this epoch time %s' %(epoch_i, step, total_loss,time_step))
                
                #temp=model.alpha.detach().clone()
                #print (temp)
                
                if opt.task=='SRM':
                   print (model.alpha.data,model.alpha.grad)
                  
                with open(opt.task+".txt",'a') as logf:
                    logf.write(' Epoch %d Training step %d loss %f this epoch time %s\n' %(epoch_i, step, total_loss,time_step))
                map_dev, mrr_dev, rankings_dev = data_evaluate(model, dev_data, "dev", qrels)
                #map_test, mrr_test = data_evaluate(model, test_data, "test")
                # lets just use dev first...so modify like this:
                map_test=map_dev
                mrr_test=mrr_dev

                report_loss = total_loss
                total_loss = 0
                if map_dev >= best_map_dev:
                    best_map_dev = map_dev
                    best_map_test = map_test
                    best_mrr_dev = mrr_dev
                    best_mrr_test = mrr_test
                    best_ranking_dev = rankings_dev
                    best_ranking_test = rankings_dev
                    print ("best dev-- mrr %f map %f; test-- mrr %f map %f" % (
                    best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                    with open(opt.task+".txt",'a') as logf:
                        logf.write("best dev-- mrr %f map %f; test-- mrr %f map %f\n" % (
                    best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                else:
                    print("NOT the best dev-- mrr %f map %f; test-- mrr %f map %f" %(mrr_dev,map_dev,mrr_test,map_test))
                    with open(opt.task+".txt",'a') as logf:
                        logf.write("NOT the best dev-- mrr %f map %f; test-- mrr %f map %f\n" %(mrr_dev,map_dev,mrr_test,map_test))
                if opt.save_model:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'settings': opt,
                        'epoch': epoch_i}
                    if opt.save_mode == 'all':
                        model_name = '../chkpt/' + opt.save_model + '_step_{}.chkpt'.format(step)
                        torch.save(checkpoint, model_name)
                    elif opt.save_mode == 'best':
                        model_name = '../chkpt/' + opt.save_model + '.chkpt'
                        if map_dev == best_map_dev:
                            best_map_dev = map_dev
                            best_map_test = map_test
                            best_mrr_dev = mrr_dev
                            best_mrr_test = mrr_test
                            with open(opt.task+".txt",'a') as logf:# record log
                                logf.write(' Epoch %d Training step %d loss %f this epoch time %s' %(epoch_i, step, report_loss,time_step))
                                logf.write("best dev-- mrr %f map %f; test-- mrr %f map %f" %(best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                            torch.save(checkpoint, model_name)
                            torch.save(checkpoint, os.path.join(io_obj.path,opt.save_model + '.chkpt'))
                            print('    - [Info] The checkpoint file has been updated.')
                            with open(opt.task+".txt",'a') as logf:
                                logf.write('    - [Info] The checkpoint file has been updated.\n')
                #break
        time_epend=datetime.now()
        time_ep=time_epend-time_epstart
        print('train epoch '+str(epoch_i)+' using time: '+ str(time_ep))
        if opt.task=='SRM':
           io_obj.writeList(alphas,'alphas.txt')
        io_obj.writeList([best_map_dev,best_mrr_dev],'best_map_mrr.txt')
        io_obj.writeRun(best_ranking_dev)
        #print (best_ranking_dev)

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode',type=str,choices=['train','forward'],default='train')
    parser.add_argument('-train_data')
    parser.add_argument('-val_data')
    parser.add_argument('-test_data')
    parser.add_argument('-embed')
    parser.add_argument('-vocab_size', default=400001, type=int)
    parser.add_argument('-load_model',type=str,default=None)# saved model(chkpt) dir
    parser.add_argument('-task', choices=['KNRM', 'CKNRM', 'MAXPOOL', 'AVGPOOL', 'LSTM', 'HRM','SRM'])
    parser.add_argument('-eval_step', type=int, default=1000)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-n_bins', type=int, default=21)
    parser.add_argument('-name', type=int, default=1)
    parser.add_argument('-is_ensemble', type=bool, default=False)
    parser.add_argument('-comment',type=str,default='')
    parser.add_argument('-device',type=int,default=0)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.mu =  kernal_mus(opt.n_bins)
    opt.sigma = kernel_sigmas(opt.n_bins)
    opt.n_layers = 1
    print (opt)

    io_obj=IO_Object(opt)
    io_obj.makeFolder()
    io_obj.backup()
    #with open(opt.task+".txt",'w') as logf:#log file
    #    logf.write(str(opt)+'\n')
    if opt.mode=='train':
        # ========= Preparing DataLoader =========#
        # data_dir='/data/disk1/private/zhangjuexiao/MSMARCOReranking/'
        # train_filename = data_dir+"marco_train_pair_small.pkl"
        #test_filename = data_dir+"marco_eval.pkl"
        # dev_filename = data_dir+"marco_dev.pkl"
        # train_data = pickle.load(open(train_filename, 'rb'))
        #test_data = pickle.load(open(test_filename, 'rb'))

        training_data = DataLoader(
            data=opt.train_data,
            batch_size=opt.batch_size,
            cuda=opt.cuda,
            device='cuda:%d'%(opt.device))

        validation_data = DataLoaderTest(
            data=opt.val_data,
            batch_size=opt.batch_size,
            test=True,
            cuda=opt.cuda,
            device='cuda:%d'%(opt.device))

        test_data=None
        # dev_data = pickle.load(open(dev_filename, 'rb'))

        if opt.task == "KNRM":
            model = KNRM.knrm(opt, opt.embed)
        elif opt.task == "CKNRM":
            model = CKNRM.cknrm(opt, opt.embed)
        elif opt.task == 'AVGPOOL':
            model=AVGPOOL.avgpool(opt, opt.embed)
        elif opt.task == 'MAXPOOL':
            model=MAXPOOL.maxpool(opt, opt.embed)
        elif opt.task == 'LSTM':
            model=LSTM.lstm(opt, opt.embed)
        elif opt.task == 'HRM':
            model=HRM.hrm(opt,opt.embed)
        elif opt.task == 'SRM':
            model=SRM.srm(opt,opt.embed)
        test_data=None

        #crit = nn.MarginRankingLoss(margin=1, size_average=True)
        crit = nn.MarginRankingLoss(margin=1, reduction='mean')

        if opt.cuda:
            #model = model.cuda()
            #crit = crit.rit = crit.cuda()
            model.to('cuda:%d'%(opt.device))
            crit.to('cuda:%d'%(opt.device))
        total_time=datetime.now()
        #print (list(model.parameters()))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
        train(model, opt, crit, optimizer, training_data, validation_data, test_data,io_obj)
        total_time=datetime.now()-total_time
        print('trainning completed, using time: ' +str(total_time))
    elif opt.mode == 'forward':
        print('load pretrained model to forward')
        if opt.load_model==None:
            print('error! specify model!')
            exit()

        chkpt=torch.load(opt.load_model)# load checkpoint
        # opt=chkpt['settings']
        state_dict=chkpt['model']
        state_dict={k:v.cpu() for k,v in state_dict.items()}

        if opt.task == 'KNRM':
            model=KNRM.knrm(opt)
        elif opt.task == 'CKNRM':
            model=CKNRM.cknrm(opt)
        elif opt.task == 'AVGPOOL':
            model=AVGPOOL.avgpool(opt)
        elif opt.task == 'MAXPOOL':
            model=MAXPOOL.maxpool(opt)
        elif opt.task == 'LSTM':
            model=LSTM.lstm(opt)
        elif opt.task == 'HRM':
            model=HRM.hrm(opt,opt.embed)
        elif opt.task == 'SRM':
            model=SRM.srm(opt,opt.embed)
        model.load_state_dict(state_dict)
        #model.cuda()
        model.to('cuda:%d'%(opt.device))

        
        test_data = DataLoaderTest(
            data=opt.test_data,
            batch_size=opt.batch_size,
            test=True,
            cuda=opt.cuda,device='cuda:%d'%(opt.device))

        data_forward(model, test_data, '../output/'+opt.task+'_output_%d.txt'%opt.name, '../output/'+opt.task+'_raw_output_%d.txt'%opt.name)

if __name__ == "__main__":
    main()
