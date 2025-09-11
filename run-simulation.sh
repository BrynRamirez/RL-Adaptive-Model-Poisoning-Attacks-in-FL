#!/bin/bash
### Sets the job's name.
#SBATCH --job-name=Fed-RL_Simulation

### Sets the job's output file and path.
#SBATCH --output=Fed-RL.out.%j

### Sets the job's error output file and path.
#SBTACH --error=Fed-RL.err.%j

### Requested number of nodes for this job. Can be a single number or a range.
#SBATCH -N 1


### Requested partition (group of nodes, i.e. compute, bigmem, gpu, etc.) for the resource allocation.
#SBATCH -p kimq

### Requested number of gpus
#SBATCH --gres=gpu:1

### Limit on the total run time of the job allocation.
#SBATCH --time=12:00:00

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Activating Fed-RL Environment"
source ~/Fed-RL/venv/bin/activate

echo "Running server.py"
python3 ~/Fed-RL/federated_learning/server.py

echo "Deactivating Python Environment"
deactivate

echo "Done."
