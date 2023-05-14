#!/bin/sh

python3 main.py -c configs/MNIST/mnist-median-krum.json -o out/mnist-median-krum-threshold-0001
python3 main.py -c configs/MNIST/mnist-median-krum-ext.json -o out/mnist-median-krum-threshold-0001-ext
python3 main.py -c configs/BreastCancer/breastcancer-median-krum.json -o out/breastcancer-median-krum-threshold-0001
python3 main.py -c configs/BreastCancer/breastcancer-median-krum-ext.json -o out/breastcancer-median-krum-threshold-0001-ext
