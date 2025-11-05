#!/bin/sh

module load cuda/12.1.1
module load nvhpc/24.5


# allocate GPU

salloc --gpus=1

echo "Finished loading CUDA modules and allocating GPU resources"
