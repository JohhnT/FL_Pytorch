#!/bin/sh

python3 main.py -c configs/MNIST/mnist-krum-mean.json -o out/mnist-krum-mean
python3 main.py -c configs/MNIST/mnist-krum-mean-ext.json -o out/mnist-krum-mean-ext
python3 main.py -c configs/BreastCancer/breastcancer-krum-mean-ext.json -o out/breastcancer-krum-mean-ext
python3 main.py -c configs/BreastCancer/breastcancer-krum-mean.json -o out/breastcancer-krum-mean
python3 main.py -c configs/MNIST/mnist-krum-krum.json -o out/mnist-krum-krum
python3 main.py -c configs/MNIST/mnist-krum-krum-ext.json -o out/mnist-krum-krum-ext
python3 main.py -c configs/BreastCancer/breastcancer-krum-krum.json -o out/breastcancer-krum-krum
python3 main.py -c configs/BreastCancer/breastcancer-krum-krum-ext.json -o out/breastcancer-krum-krum-ext
