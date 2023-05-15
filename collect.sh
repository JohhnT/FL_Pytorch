#!/bin/sh

experiment=$1

mkdir "extracted/$experiment"
cat "out/$experiment" | grep "training acc" | head -n100 | awk '{print $4}' | awk -F ',' '{print $1}' > "extracted/$experiment/training acc"
cat "out/$experiment" | grep "training acc" | head -n100 | awk '{print $6}' | awk -F ',' '{print $1}' > "extracted/$experiment/test acc"
cat "out/$experiment" | grep "training acc" | head -n100 | awk '{print $8}' > "extracted/$experiment/test loss"
#head -n5 "extracted/$experiment/*"
#./plot.sh $experiment training_acc
#./plot.sh $experiment test_acc
#./plot.sh $experiment test_loss

