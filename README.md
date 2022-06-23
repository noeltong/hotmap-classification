# Hotmap Classification

This repository is a PyTorch implementation of hotmap classification.

## Preliminaries

An Anaconda environment is needed. We first create a virtual env by
```
conda create -n cls python=3.9 --yes
```
and then activate it by
```
conda activate cls
```

After that, a `torch==1.11.0` is suggested to be installed first.
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch --yes
```
and other requirements are listed in `requirements.txt`, which can be installed by a simple script
```
pip install -r requirements.txt
```

## Usage of `wandb`

In this project, we used `wandb` to log information during training process. We install `wandb` by
```
pip install wandb
```
After the installation, we need to login to `wandb` using
```
wandb login
```

## Training data

The training data is not available for public. However, you can still have your own data in `.mat`, stored in `./data` folder. Don't forget to check `./utils/dataset.py` to ensure there's no problem in loading the `.mat` file.

## Logs and checkpoints

We also used `tensorboard` to record the loss and accuracy during training. The logs are stored in `./log`.

We save checkppoints every 5 epochs in `./ckpt`.