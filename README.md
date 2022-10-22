1. Run `conda env create -f env.yml` to create the numerosity_illusions conda environment
2. Run `cp sample_yaml.yml YOUREXPERIMENTNAME.yml` to create a yaml file for your experiment. Modify it with a text editor to set the parameters for your experiment. By default numerosity neurons are identified using criteria from Stoianovand Zorzi, 2012 -- ie numerosity neurons are those for which numerosity explains more than 0.10 variance, other factors explain less than 0.01.
3. Run `python -m scripts.pipeline.generate_pipeline YOUREXPERIMENTNAME.yml`. This will create three directories with the name of your experiment: one in dat
a/stimuli/ to hold the image datasets you create, one in data/models/ to hold the activations of the model, and one in experiment_runs/ to hold the sbatch scripts which will run the experiment. The following sbatch files will be created:
      1. *1_submit_gen_stimuli.sbatch*: creates the stimuli.
      2. *2_submit_save_layers.sbatch*: saves the activations. This will create two folders in numerosity_illusions/data/models, one for pretrained Alexnet and one for randomly initialized Alexnet. Within each of these folders, a folder for each of the layers to save is created. Finally, within the layer folders, there are folders for each of the datasets which were created. An activations.csv file is created in each of these folders which contains the activations of the neurons in that layer in response to each of the images in the dataset for that folder.
      3. *3_identify_numerosity_neurons.sbatch*: identifies the numerosity neurons. This will run a multi-way ANOVA on the numerosity_neurons dataset for each layer with factors 'numerosity', 'size', and 'spacing'. The following outputs are saved:
      	 - *anova_dict.pkl*: contains the results of the anova for each neuron. 
	 - *numerosities.npy*: contains a numpy array of the numerosities in the dataset.
	 - *{selection_method}_numerosityneurons.npy*: contains an array of arrays, one for each numerosity in the dataset, where each list contains the indices of the numerosity neurons in that layer which are maximally sensitive to that numerosity. The lists are sorted in ascending order by numerosity. Note that there is no guarantee that these lists are the same length, hence a numpy object array is used.
	 - *{selection_method}_tuning_curves.npy*, *{selection_method}_tuning_curves.npy*: array of tuning curves and standard errors for each numerosity in the dataset. That is, for every neuron in the layer, we first calculate the mean activation of that neuron averaged across all images with the same numerosity in the dataset. This array of average activations in response to each numerosity is called a tuning curve. Then, for each numerosity, we calculate the average tuning curve across all of the numerosity neurons maximally sensitive to that numerosity, as well as the standard error for each numerosity.
	 - *variance_explained.png*: scatterplot of the amount of variance explained by numerosity vs each correlated variable ie Spacing and Size. Numerosity neurons are plotted in red, all others in black.
	 - *numerosity_histogram.png*: histogram of the number of neurons maximally sensitive to each numerosity.
      4. *4_submit_save_tuning.sbatch*: saves the tuning curves for the activations datasets. That is, the same process described above is done, only now the activations we average across are not the same as the ones used to identify the numerosity neurons.
      5. *5_submit_plot_tuning.sbatch*: plots the tuning curves for the numerosity_neurons dataset against those for the activations datasets.
4. Run `` bash scheduler.sh `ls experiment_runs/YOUREXPERIMENTNAME/*.sbatch` `` to schedule all the sbatch scripts in order, dependent on the completion of the previous script. This assumes you can submit jobs to Slurm.
