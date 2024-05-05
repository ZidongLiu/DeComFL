#!/bin/bash -l

#SBATCH --job-name=rge_main_test_slurm_1       # Name of your job
#SBATCH --time=0-1:00:00               # Time limit, format is Days-Hours:Minutes:Seconds

#SBATCH --ntasks=1                      # 1 task (default of 1 CPU per task)
#SBATCH --mem=32g                        # 32GB of RAM for the whole job
#SBATCH --gres=gpu:v100:1

#SBATCH --output=%x_%j.out	# Output file
#SBATCH --error=%x_%j.err	# Error file

#SBATCH --mail-user=slack:@zl4063	# Slack username to notify
#SBATCH --mail-type=BEGIN			# Type of slack notifications to send
#SBATCH --account=fl-het          # Slurm account
#SBATCH --partition=debug             # Partition to run on
#SBATCH --cpus-per-task=2

module load python/3.10.12-gcc-11.2.0-ubv7zcio

python rge_main.py --log-to-tensorboard=rge_main_test_slurm_1 --checkpoint-update-plan=every5 --epoch=20 --num-pert=5 --grad-estimate-method=rge-forward