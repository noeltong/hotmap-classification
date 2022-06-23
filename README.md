# Hotmap Classification

This repository is a PyTorch implementation of Hotmap classification.

## Preliminaries

An Anaconda environment is needed. We firstly create a virtual env by
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
