#!/bin/sh

./collect-by-pair mnist-median-krum
./collect-by-pair mnist-median-krum-compromised-10
./collect-by-pair mnist-median-krum-compromised-5
./collect-by-pair mnist-median-krum-threshold-0001
./collect-by-pair mnist-median-mean
./collect-by-pair mnist-median-mean-compromised-10
./collect-by-pair mnist-median-mean-compromised-5
./collect-by-pair mnist-median-mean-compromised-50

./collect-by-pair breastcancer-median-krum
./collect-by-pair breastcancer-median-krum-compromised-10
./collect-by-pair breastcancer-median-krum-compromised-5
./collect-by-pair breastcancer-median-krum-threshold-0001
./collect-by-pair breastcancer-median-mean
./collect-by-pair breastcancer-median-mean-compromised-10
./collect-by-pair breastcancer-median-mean-compromised-5
./collect-by-pair breastcancer-median-mean-compromised-50
