import time
import argparse
import os
import numpy as np

from . import utility_functions as utils

if __name__ == '__main__':

    start_time = time.time()

    #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Save the tuning curves of a particular set of numerosity neurons on a particular dataset.')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of experiment')
    parser.add_argument('--model_directory', type=str, required=True,
                        help='Directory in experiment directory to find dataset directories in')
    parser.add_argument('--layer', type=str, required=True,
                        help='Layer to save tuning curves for.')
    parser.add_argument('--numerosity_neurons_dataset_name', type=str, required=True,
                        help='Name of dataset directory to look for numerosity neurons in. This determines which neurons in the layer have tuning curves saved for them.')
    parser.add_argument('--selection_method', type=str, choices=['variance','dewing_variance','anova', 'anova_corrected', 'anova1way', 'anova1way_corrected'], required=True,
                        help='Which selection method to use the numerosity neurons of.')
    parser.add_argument('--activations_dataset_name', type=str, required=True,
                        help='Name of dataset directory to use the activations of. This determines which activations (ie which dataset the activations are in response to) are used to create the tuning curves for the numerosity neurons specified above.')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)

    # Get path to model directory
    models_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/models', args.experiment_name, args.model_directory, args.layer)

    # Load numerosity neurons
    numerosity_neuron_path = os.path.join(models_path, args.numerosity_neurons_dataset_name)
    numerosities = np.load(os.path.join(numerosity_neuron_path, 'numerosities.npy'))
    # Allow pickle since subarrays are different lengths
    sorted_numerosity_neurons = np.load(os.path.join(numerosity_neuron_path,f"{args.numerosity_neurons_dataset_name}_{args.selection_method}_numerosityneurons.npy"), allow_pickle=True)

    # Load activations
    activations_path = os.path.join(models_path, args.activations_dataset_name)
    df = utils.getActivationDataFrame(activations_path,'activations')
    average_activations = df.groupby(['numerosity'],as_index=False).mean()

    # Save the tuning curves of the numerosity neurons on these activations
    save_path = os.path.join(activations_path, f"{args.numerosity_neurons_dataset_name}_{args.selection_method}_")
    tuning_curves, std_errs = utils.getTuningCurves(sorted_numerosity_neurons,numerosities,average_activations)
    np.save(save_path+"tuning_curves", tuning_curves)
    np.save(save_path+"std_errs", std_errs)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
