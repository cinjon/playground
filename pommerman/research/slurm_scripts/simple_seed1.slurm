#!/bin/bash
#SBATCH --job-name=simple_seed1
#SBATCH --output=slurm_logs/simple_seed1.out
#SBATCH --error=slurm_logs/simple_seed1.err
#SBATCH --qos=batch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raileanu@cs.nyu.edu
module purge
OMP_NUM_THREADS=1 python main.py --num-processes 16 --how-train simple --save-dir /home/raileanu/pomme_logs/trained_models --log-interval 10 --save-interval 1000 --run-name simple --seed 1
