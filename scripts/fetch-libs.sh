#!/bin/bash


mkdir -p lib/ || exit
cd lib/ || exit

git clone https://github.com/google/benchmark.git || exit
