#!/bin/sh

python3 main.py -c configs/MNIST/mnist-median-mean.json -o out/mnist-median-mean-compromised-50
python3 main.py -c configs/MNIST/mnist-median-mean-ext.json -o out/mnist-median-mean-compromised-50-ext
python3 main.py -c configs/BreastCancer/breastcancer-median-mean-ext.json -o out/breastcancer-median-mean-compromised-50-ext
python3 main.py -c configs/BreastCancer/breastcancer-median-mean.json -o out/breastcancer-median-mean-compromised-50
python3 main.py -c configs/MNIST/mnist-median-krum.json -o out/mnist-median-krum-compromised-50
python3 main.py -c configs/MNIST/mnist-median-krum-ext.json -o out/mnist-median-krum-compromised-50-ext
python3 main.py -c configs/BreastCancer/breastcancer-median-krum.json -o out/breastcancer-median-krum-compromised-50
python3 main.py -c configs/BreastCancer/breastcancer-median-krum-ext.json -o out/breastcancer-median-krum-compromised-50-ext
