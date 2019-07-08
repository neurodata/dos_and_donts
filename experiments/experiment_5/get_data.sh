#!/bin/bash

# set -ex

mkdir -p ./data
cd ./data

for idx in {1..10}; do
    if [ $idx -ne 10 ]; then
        curl -O "http://users.cecs.anu.edu.au/~bdm/data/graph${idx}.g6"
    elif [ $idx -eq 10 ]; then
        curl -O "http://users.cecs.anu.edu.au/~bdm/data/graph${idx}.g6.gz";
        gunzip graph${idx}.g6
    fi
done
