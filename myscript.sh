#!/bin/bash

for nClust in {1..140}
do
    for S in {1..100}
    do
        cat iris-data.txt | ./split.bash 10 python3 kmeans.py $S $nClust >> iris-results-cluster-$nClust.txt

done
done