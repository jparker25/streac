################################################################################
# process_neural_data.py
# Gathers all neural spike train data for parallelization and processing.
# Author: John E. Parker
################################################################################

# import python modules
import os, pickle

# import user made modules
from neuron import neural_cell as nc
from helpers import *

def get_trial_data_parallel(data_direc,group,cells,save_direc):
    """
    Function that gathers neurons for parallel trial analysis

    :param data_direc: string of the data_directory from analyze_data.py
    :param group: The group to start to analyze (subdirectory in data_direc)
    :param cells: list of diectories corresponding to neurons in group
    :param save_direc: Destination of where to save the processed neurons
    :return neurons: list of neuron objects genereated by information in data_direc
    """
    direc = f"{data_direc}/{group}" # Create temp variable of the data directory and group
    out_dir = check_direc(f"{save_direc}/{group}") # Create directory for processed neurons
    neurons = [] # Empty list to be filled with all neuron objects
    for cell_dir in cells: # Iterate through all cell directories in cells
        neuron = nc(direc,group,cell_dir) # Create a neuron object
        neuron.set_save_direc(out_dir) # Set the save directory for the neuron object
        neurons.append(neuron) # Add the neuron object to list of neurons
    return neurons # Return list of neurons

def analyze_trial_parallel(neuron,bin_width,trial_percentile,average_percentile,baseline_start,baseline_length,trial_start,trial_length,in_bin_threshold,ex_bin_threshold):
    """
    Analyzes single neuron trials.

    :param neuron: Neuron object to analyze.
    :param bin_width: Width of the bins.
    :param trial_percentile: Between 0 and 100 indicating the trial percentile for inhibition and excitation identification.
    :param average_percentile: Between 0 and 100 indicating the average percentile for inhibition and excitation identification.
    :param baseline_start: Number representing time before stimulus on-set to begin the baseline
    :param baseline_length: Length of baseline.
    :param trial_start: Number representing time from stimulus on-set to begin the trial period
    :param trial_length: Length of trial.
    :param in_bin_threshold: Integer of bins that consider a trial inhibited.
    :param ex_bin_threshold: Integer of bins that consider a trial excited.
    """
    neuron.gather_data() # Gather all data of neuron from data_directory
    # Generate and analyze trial data using input parameters
    neuron.gather_trials(baseline_start,baseline_length,trial_start,trial_length,bin_width,trial_percentile,average_percentile,in_bin_threshold,ex_bin_threshold)
    neuron.gather_pre_post() # Generate and analyze pre/post periods 
    neuron.save_data() # Save the updated neuron object

def grab_analyze_avg_neurons(save_direc):
    """
    Creates a list of neuron objects for average resposne analysis.

    :param save_direc: Location of where neuron objects to be processed are stored.
    :return neuron_data: List containing sublists of neuron objects pertaining to each group.
    """
    neuron_data = [] # Empty list to be filled with sublits of objects
    for dir in os.listdir(save_direc): # Iterate through the save_direc to gather all neuron objects
        if dir != "comparisons" and dir != ".DS_Store" and os.path.isdir(f"{save_direc}/{dir}"): # Confirm subdirectory is a group
            neurons = [] # For the group, create a sublist for individual neurons
            for d in os.listdir(f"{save_direc}/{dir}"): # Iterate through directories and append nueuron object to list
                if d[0:6] == "Neuron": # Make sure directory is neuron directory
                    neuronFile = open(f"{save_direc}/{dir}/{d}/neuron.obj","rb") # Read in neuron object file
                    neurons.append(pickle.load(neuronFile)) # Read in neuron object and append to neurons list
                    neuronFile.close() # Close neuron object file
            neuron_data.append(neurons) # Append sublist of neurons to neuron_data 
    return neuron_data # Return neuron_data

def analyze_avg_neurons_parallel(neuron):
    """
    Analyze the neuron to determine the average response.

    :param neuron: Neuron object to analyze for average response.
    """
    print(f"Started {neuron.cell_dir}") # Print which neuron is being processed
    neuron.class_dict = {"complete inhibition":0, "adapting inhibition": 1, "partial inhibition":2,"no effect":3,"excitation":4,"biphasic IE":5,"biphasic EI": 6} # Classification dictioary for storate
    neuron.classify_neuron() # Classify the neuron
    neuron.save_data() # Save all updated data to the neuron object and relevant files
    print(f"Finished {neuron.cell_dir}") # Print the neuron is finished processing
