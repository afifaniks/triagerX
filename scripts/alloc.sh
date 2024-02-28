#!/bin/bash

if [ "$1" = "gpu" ]
then
    echo "Allocating GPU partition..."
    salloc --mem=50G -t 04:59:00 -p disa --gres=gpu:1
else
    echo "Allocating CPU partition..."
    salloc --mem=50G -c 1 -N 1 -n 1 -t 04:00:00 -p cpu2019-bf05
fi