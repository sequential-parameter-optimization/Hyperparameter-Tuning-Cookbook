#!/bin/bash
 
### Vergabe von Ressourcen
#SBATCH --job-name=Test
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#----
#SBATCH --partition=gpu

nvidia-smi

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_spot.pkl>"
    exit 1
fi

SPOT_PKL=$1

module load conda

conda activate spot312

python startPython.py "$SPOT_PKL"
