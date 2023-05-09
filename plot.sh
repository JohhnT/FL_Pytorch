#!/bin/sh

scope=$1
target=$2

plot_file="extracted/$scope/$target"

gnuplot -e "set terminal png; set output 'plots/$scope-$target.png'; set ylabel '$target'; set xlabel 'Round'; plot '$plot_file' title '$target of $scope' with lines; set terminal x11; set output; replot;"
