## Soft Kernel-based Ranking on A Statistical Manifold
This repository contains resources developed within the following paper:

    Xinshi Lin and Wai Lam. “Soft Kernel-based Ranking on A Statistical Manifold”, SIGIR 2020

## requirements
Python 3.4+

Pytorch 1.2+

This implementation works on both Windows and Linux. 

Tested on a AMD Radeon RX480 8GB(with ROCM platform installed) and a NVIDIA GTX 1080Ti.

## usage
python main.py -train_data ../data/train.txt -val_data ../data/dev.txt -embed ../chkpt/embed.npy -task SRM -batch_size 64 -save_model best -vocab_size 315370

*since large file cannot be uploaded freely in github, dev.txt/train.txt/embed.npy are currently unavailable in this repository. For request of these files, please send mail to me or refer to [here](https://github.com/thunlp/Kernel-Based-Neural-Ranking-Models/tree/master/data). 

dev.txt and train.txt are from the MS-MARCO dataset. embed.npy is the glove embeddings in numpy 2D-array format, each row is a vector corresponding to a word in the source_codes/data/vocab.tsv.

## License
Beer-ware or Snack-ware License

## Contact
Xinshi Lin (xslin@se.cuhk.edu.hk)

## Acknowledgement
This project is based on an existing github project https://github.com/thunlp/Kernel-Based-Neural-Ranking-Models.
