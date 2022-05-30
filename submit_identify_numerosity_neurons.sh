#!/bin/bash
#
#SBATCH --job-name=gen_stimuli
#SBATCH --output=jobs/gen_stimuli_%j.txt
#
#SBATCH --time=3:00:00

source activate numerosity_illusions

python -m scripts.analysis.identify_numerosity_neurons --model_directory alexnet_random --dataset_directory test --layer features_12 --selection_method variance
