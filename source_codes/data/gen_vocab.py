import re

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')

def accumulate(s, word_dic):
    lst = regex_multi_space.sub(' ', regex_drop_char.sub(' ', s.lower())).strip().split()
    for word in lst:
        if word not in word_dic:
            word_dic[word] = 0
        word_dic[word] += 1
    return ','.join(lst)

word_dic = {}

f = open('triples.train.small.tsv')
for line in f:
    line = line.strip('\n').split('\t')
    accumulate(line[0], word_dic)
    accumulate(line[1], word_dic)
    accumulate(line[2], word_dic)
    
f = open('top1000.dev.tsv')
for line in f:
    line = line.strip('\n').split('\t')
    accumulate(line[2], word_dic)
    accumulate(line[3], word_dic)
    
idx = 1
fout = open('vocab_gen.tsv', 'w')
for word in word_dic:
    if word_dic[word] >= 5:
        fout.write(word + '\t' + str(idx) + '\n')
        idx += 1
