import csv
def get_qrels(QRELS_DEV):
    qrels = {}
    with open(QRELS_DEV, mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            qid = row[0]
            did = row[2]
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(did)
    return qrels

res = {}
for i in range(1, 8):
    f = open("../output/CKNRM_raw_output_%d.txt"%i)
    for line in f:
        line = line.strip().split('\t')
        qid = line[0]
        did = line[1]
        output = float(line[2])
        if qid not in res:
            res[qid] = {}
        if did not in res[qid]:
            res[qid][did] = 0
        res[qid][did] += output

result_dict = {}
for qid in res:
    result_dict[qid] = []
    for did in res[qid]:
        result_dict[qid].append((did, res[qid][did]))

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

writer = open('CKNRM_dev.res', 'w')
for qid, values in result_dict.items():
    res = sorted(values, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
    for rank,value in enumerate(res):
        writer.write(str(qid)+'\t'+str(value[0])+'\t'+str(rank+1)+'\n')
# output results:
print('len of scored dict:',len(result_dict))

MAP = c_1_j / float(len(result_dict) - no_label)
MRR = c_2_j / float(len(result_dict) - no_label) #
#print ""
#print(" evaluate on " + flag + " MAP: %f" % MAP)
#print(" evaluate on " + flag + ' MRR: %f' % MRR)
print(" evaluate on " + " MAP: %f" % MAP)
print(" evaluate on " + " MRR: %f" % MRR)

