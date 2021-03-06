#!/bin/bash
#
#SBATCH --job-name=save_layers
#SBATCH --output=jobs/save_layers_%j.txt
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH -p gpu
#SBATCH -G 2

source activate numerosity_illusions

#features_0 features_1 features_2 features_3 features_4 features_5 features_6 features_7 features_8 features_9 features_10 features_11 classifier_1 classifier_2 classifier_4 classifier_5 classifier_6
for stim_dir in `ls data/stimuli`;do echo $stim_dir; python -m scripts.models.save_layers --model alexnet --stimulus_directory $stim_dir --layers features_12  --num_workers 1; python -m scripts.models.save_layers --model alexnet --pretrained --stimulus_directory $stim_dir --layers features_12 --num_workers 1;done

