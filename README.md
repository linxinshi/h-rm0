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

## License
Beer-ware or Snack-ware License

## Acknowledgement
This project is developed based on an existing github project https://github.com/thunlp/Kernel-Based-Neural-Ranking-Models.
