# train
wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
tar -vxzf triples.train.small.tar.gz
python tokenize_train.py

# dev
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz
tar -vxzf top1000.dev.tar.gz
python tokenize_dev.py

