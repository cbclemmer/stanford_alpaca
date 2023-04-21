#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -t 60:00
#SBATCH --gres=gpu:v100-32:1

if [ ! -e "fine_tuned_model" ]; then
  echo "Error: Must have fine tuned model in fine_tuned_model/ directory"
  exit 1
fi

module load AI/pytorch_23.02-1.13.1-py3

export PATH="/opt/packages/cuda/v11.7.1/bin:$PATH"
export LD_LIBRARY_PATH="/opt/packages/cuda/v11.7.1/lib64:$LD_LIBRARY_PATH"

pip install -r requirements.txt
pip install deepspeed
pip install accelerate
# pip freeze


#Set protoc to python
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python3 test_model.py

echo "DONE"
