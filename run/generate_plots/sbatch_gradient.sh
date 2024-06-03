#!/bin/bash
#SBATCH --job-name=gradient-plot
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu
#SBATCH --output=/gscratch/ark/hari/finetune/sruns/%j.out

# I use source to initialize conda into the right environment.
source activate testenv5
bash run/generate_plots/sst2.sh gradient
bash run/generate_plots/snli.sh gradient