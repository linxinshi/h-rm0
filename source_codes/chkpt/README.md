# Kernel-Based-Neural-Ranking-Models

### Check Points

This folder stores the trained models.

#### Conv-KNRM

The Conv-KNRM Ensemble Model (8 Conv-KNRM Models) can be downloaded from the following url:

https://drive.google.com/drive/folders/1ndkc2GBgKTB7nERX1roco1IVGD8FfTl7?usp=sharing

It is recommended to download all the files and store them in this `/chkpt/` folder.

#### Pretrained Embedding File

The embedding file `embed.npy` can also be found from the above link. This embedding is corresponding to `/data/vocab.tsv`. The embeddings are built from `glove.6b.300d`([here](https://www.kaggle.com/thanakomsn/glove6b300dtxt)). If the words are not contained in the glove, then the embeddings are randomly initialized.

