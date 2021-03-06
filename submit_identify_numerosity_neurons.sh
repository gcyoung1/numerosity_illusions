#!/bin/bash
#
#SBATCH --job-name=identify_numerosity_neurons
#SBATCH --output=jobs/identify_numerosity_neurons_%j.txt
#
#SBATCH --time=3:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem 192G

source activate numerosity_illusions

python -m scripts.analysis.identify_numerosity_neurons --model_directory alexnet_random --dataset_directory no_illusion --layer features_12 --selection_method variance
