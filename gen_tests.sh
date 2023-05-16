#!/bin/sh

python3 main.py -c configs/MNIST/mnist-krum-mean.json -o out/mnist-krum-mean-compromised-50
python3 main.py -c configs/MNIST/mnist-krum-mean-ext.json -o out/mnist-krum-mean-compromised-50-ext
python3 main.py -c configs/BreastCancer/breastcancer-krum-mean-ext.json -o out/breastcancer-krum-mean-compromised-50-ext
python3 main.py -c configs/BreastCancer/breastcancer-krum-mean.json -o out/breastcancer-krum-mean-compromised-50
python3 main.py -c configs/MNIST/mnist-krum-krum.json -o out/mnist-krum-krum-compromised-50
python3 main.py -c configs/MNIST/mnist-krum-krum-ext.json -o out/mnist-krum-krum-compromised-50-ext
python3 main.py -c configs/BreastCancer/breastcancer-krum-krum.json -o out/breastcancer-krum-krum-compromised-50
python3 main.py -c configs/BreastCancer/breastcancer-krum-krum-ext.json -o out/breastcancer-krum-krum-compromised-50-ext
