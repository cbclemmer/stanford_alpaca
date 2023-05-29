#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -t 60:00
#SBATCH --gres=gpu:v100-32:1

if [ ! -e "fine_tuned_model" ]; then
  echo "Error: Must have fine tuned model in fine_tuned_model/ directory"
  exit 1
fi

#Set protoc to python
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python3 test_model.py

echo "DONE"