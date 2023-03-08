################################################################################
# gather_spikes.def
# Gathers all spike train data for each cell and each trial
# Author: John E. Parker
################################################################################

# import python modules
import os, pickle,sys
import numpy as np
from pprint import pprint

# import user made modules
from neuron import neural_cell as nc
from helpers import *

def get_trial_data_parallel(data_direc,group,cells,save_direc):
    '''
    Function that analyzes each neuron based on given parameters.
        str mice -> string specifying the type of mice (6-OHDA or Naive)
        str delivery -> string specifying the type of delivery (PV-DIO or hsyn)
        str direc -> string path for where the data is stored to analyze
        str save_direc -> string path for the directory of where to store analyzed data
        arr bin_edges -> array of values specifying the bin edges, left side inclusive, right side exclusive (default 0 to 10 by 0.5s)
        int percentile -> integer of the percentile to compare against for significance (default 95)
    '''
    direc = f"{data_direc}/{group}"
    '''
    all_cells = []; # array that will hold all the paths each neuron data
    for path, subdirs, files in os.walk(direc): # for loop that walks through all the directories to identify ones with desired data
        for name in files:
            if name != ".DS_Store": # Only analyze the directories with the right names
                all_cells.append(os.path.join(path,name)) # If the name is valid, add to the all_cell array
    '''
    out_dir = check_direc(f"{save_direc}/{group}")
    neurons = []
    #for cell in range(1,cells+1): # For each cell analyze and classify the data
    for cell_dir in cells:
        neuron = nc(direc,group,cell_dir) # Create a neuron object
        neuron.set_save_direc(out_dir)
        neurons.append(neuron)
    return neurons

def analyze_trial_parallel(neuron,bin_width,trial_percentile,average_percentile,baseline_start,baseline_length,trial_start,trial_length,in_bin_threshold,ex_bin_threshold):
    neuron.gather_data()
    neuron.gather_trials(baseline_start,baseline_length,trial_start,trial_length,bin_width,trial_percentile,average_percentile,in_bin_threshold,ex_bin_threshold)
    neuron.gather_pre_post()
    neuron.save_data()

def grab_analyze_avg_neurons(save_direc):
    neuron_data = []
    for dir in os.listdir(save_direc):
        if dir != "comparisons" and dir != ".DS_Store" and os.path.isdir(f"{save_direc}/{dir}"):
            neurons = []
            for d in os.listdir(f"{save_direc}/{dir}"):
                if d[0:6] == "Neuron":
                    neuronFile = open(f"{save_direc}/{dir}/{d}/neuron.obj","rb")
                    neurons.append(pickle.load(neuronFile))
                    neuronFile.close()
            neuron_data.append(neurons)
    return neuron_data

def analyze_avg_neurons_parallel(neuron):
    print(f"Started {neuron.cell_dir}")
    neuron.class_dict = {"complete inhibition":0, "adapting inhibition": 1, "partial inhibition":2,"no effect":3,"excitation":4,"biphasic IE":5,"biphasic EI": 6}
    neuron.classify_neuron()
    neuron.save_data()
    print(f"Finished {neuron.cell_dir}")
