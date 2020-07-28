# Kernel-Based-Neural-Ranking-Models

### Data Preparation

All the data is located in this `/data` folder. You can run the script to automatically download and preprocess the data:

```shell
./data_preparation.sh
```

Within the above script, it downloads the data from MSMARCO website and preprocesses with the `tokenize_train.py` and `tokenize_dev.py`. The code uses the tokenization method provided by the MSMARCO BM25 Baseline, and converts the tokenized terms into indexes according to `vocab.tsv`. 

### Data Format

```shell
# Train
query \t doc_pos \t doc_neg

# Dev
qid \t did \t query \t doc
```

### File Description

The `vocab.tsv` was generated on the MSMARCO train & dev set with words appeared at least 5 times. You can also generate your own vocab file via `gen_vocab.py`.

The `idf.norm.tsv` is the normed idf value calculated on the whole MSMARCO train & dev corpora.


