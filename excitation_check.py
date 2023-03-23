################################################################################
# stimulus_classification.py
# Classifies stimulus response compared to baseline.
# Author: John E. Parker
################################################################################

# import python modules
import numpy as np
import sys, os, pickle
from scipy.stats import binom
import sys

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# import user made modules
from helpers import *

def gaussian(x,s):
    """
    Function that takes in values x and standard deviation s to determine Gaussian distribution function
    
    :param x: x-values to evaluate guassian at each point
    :param s: standard deviation for the gaussian distribution function
    :return gaussian function evaluated with parameter s at values of x
    """
    # Return the gaussian distribution function with standard deviation s at values x
    return (1/(np.sqrt(2*np.pi)*s))*np.exp(-x**2/(2*s**2))

def kernel(values,spacing,bandwidth=25/1000):
    """
    Function that is the kernel for the spike density function, returns the SDF.

    :param values: array of values to evaluate SDF at, should be spikes
    :param spacing: grid to compute SDF at, should be a desired domain
    :keyword param bandwidth: bandwidth of gaussian kernel, default is 25/1000
    """
    res = np.zeros(len(spacing)) # Create empty grid that will store the SDF values, length of desired grid
    for j in range(len(values)): # Iterate through each data point and find the SDF via the Gaussian kernel
        res += gaussian((spacing-values[j]),bandwidth) # Evaluate the Gaussian kernel at every point for the given value
    return res # Return an array that is the SDF values along the domain spacing


def avg_excitation_check(neuron,baselines,stimuli,percentile=99,bin_width=0.5):
    fig,ax = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create figure that will have 2 rows and 1 column
    results = np.zeros(len(stimuli[0].bin_edges)-1) # Create empty array that will store the classifications of each bin

    sdf_bls = np.asarray([baseline.sdf for baseline in baselines])
    sdf_stims = np.asarray([stimulus.sdf for stimulus in stimuli])
    avg_sdf_stim = np.mean(sdf_stims,axis=0)
    avg_sdf_bl = np.mean(sdf_bls,axis=0)

    bl_areas = []
    for baseline in baselines:
        for bi in range(len(baselines[0].bin_edges)-1):
            st,ed = np.where(baseline.t <= baseline.bin_edges[bi])[0][-1],np.where(baseline.t < baseline.bin_edges[bi+1])[0][-1]
            bl_areas.append(np.trapz(baseline.sdf[st:ed+1],x=baseline.t[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array

    for i in range(len(stimuli[0].bin_edges)-1):
        st,ed = np.where(stimuli[0].t <= stimuli[0].bin_edges[i])[0][-1],np.where(stimuli[0].t < stimuli[0].bin_edges[i+1])[0][-1]
        if np.percentile(bl_areas,percentile) < np.trapz(avg_sdf_stim[st:ed+1],x=stimuli[0].t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
            ax[1].fill_between(stimuli[0].t[st:ed+1],avg_sdf_stim[st:ed+1],color="blue",alpha=0.5) # Fill the area of the SDF that is beyond the percentile of the baseline areas
            results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
    ax[0].hist(bl_areas,bins=20,color="gray",edgecolor="k") # Create a histogram of all the baseline SDF areas
    ax[0].vlines(np.percentile(bl_areas,percentile),ax[0].get_ylim()[0],ax[0].get_ylim()[1],color="k",linestyle="dashed") # Draw a verticle line where the percentile threshold is on the histogram
    ax[0].set_xlabel("Baseline SDF Area Bins"); ax[0].set_ylabel("Count") # Label the histogram axes
    ax[1].plot(stimuli[0].t,avg_sdf_stim,color="blue") # Plot the SDF as a function of t
    ax[1].vlines(stimuli[0].bin_edges,0,ax[1].get_ylim()[1],color="k",linestyle="dashed") # Draw vertical lines for bin edges on the SDF stimulus plot
    ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("SDF (spikes/s)") # Label the SDF axes
    plt.suptitle("Average Baseline SDF Bin Area Histogram and Average Stimulus SDF") # Title the figure
    makeNice([ax[0],ax[1]]) # Clean the figure
    plt.savefig(f"{neuron.cell_dir}/avg_sdf_shuffling.pdf") # Save the figure
    plt.close() # Close the figure
    return results # Return the array containing the values of the bins excited (1) or not (0)

def trial_excitation_check(baseline_train,stim_data_train,trial_direc,bin_edges=list(range(11)),shuffles=10,percentile=99,bin_width=0.5):
    '''
    Function that determines how the stimulus responds as compared to the baseline with the light on. Based on comparing ISI peak values for an ISI function.
        arr baseline -> array of spike times for the baseline
        arr stimulus -> array of spike times for the stimulus
        str trial_direc -> string that is the path of the trial directory to store and grab all data for the current trial
        arr bin_edges -> array that defines the edges of each bin for the stimulus, left side inclusive, right side exclusive, default integers 0 to 10
        int shuffles -> number of times to shuffle baseline values to recreate a new ISI function
        int percentile -> integer defining what the percentile should be to check the classification for significance
    '''
    fig,ax = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create figure that will have 2 rows and 1 column
    results = np.zeros(len(stim_data_train.bin_edges)-1) # Create empty array that will store the classifications of each bin
    bl_isi = np.diff(baseline_train.spikes) # Find the baseline ISI values
    bl_shuffles = [np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)] # Shuffle the bl ISI values for shuffles number of times
    bl_spikes = [[baseline_train.spikes[0]+sum(bl_shuff[0:i]) if i > 0 else baseline_train.spikes[0] for i in range(len(bl_isi))] for bl_shuff in bl_shuffles] # Recreate the spikes based on the shuffled ISI values
    bl_sdf_fs = [kernel(bl_spike,baseline_train.t) for bl_spike in bl_spikes]
    bl_areas = [] # Empty array that will store the areas of each bin of the shuffled baseline
    for i in range(shuffles): # Loop through all the shuffles
        for bi in range(len(baseline_train.bin_edges)-1): # In each shuffle, loop throuhg each bin
            st,ed = np.where(baseline_train.t <= baseline_train.bin_edges[bi])[0][-1],np.where(baseline_train.t < baseline_train.bin_edges[bi+1])[0][-1]
            #bl_areas.append(np.trapz(baseline_train.sdf[st:ed+1],x=baseline_train.t[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array
            bl_areas.append(np.trapz(bl_sdf_fs[i][st:ed+1],x=baseline_train.t[st:ed+1]))

    for i in range(len(stim_data_train.bin_edges)-1): # Iterate through each bin
        st,ed = np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],np.where(stim_data_train.t < stim_data_train.bin_edges[i+1])[0][-1]
        if np.percentile(bl_areas,percentile) < np.trapz(stim_data_train.sdf[st:ed+1],x=stim_data_train.t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
            ax[1].fill_between(stim_data_train.t[st:ed+1],stim_data_train.sdf[st:ed+1],color="blue",alpha=0.5) # Fill the area of the SDF that is beyond the percentile of the baseline areas
            results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
    ax[0].hist(bl_areas,bins=20,color="gray",edgecolor="k") # Create a histogram of all the baseline SDF areas
    ax[0].vlines(np.percentile(bl_areas,percentile),ax[0].get_ylim()[0],ax[0].get_ylim()[1],color="k",linestyle="dashed") # Draw a verticle line where the percentile threshold is on the histogram
    ax[0].set_xlabel("Shuffled Baseline SDF Area Bins"); ax[0].set_ylabel("Count") # Label the histogram axes
    ax[1].plot(stim_data_train.t,stim_data_train.sdf,color="blue",label="Stim ISI Interpolation") # Plot the SDF as a function of t
    ax[1].vlines(stim_data_train.bin_edges,ax[1].get_ylim()[0],ax[1].get_ylim()[1],color="k",linestyle="dashed",alpha=0.5) # Draw vertical lines for bin edges on the SDF stimulus plot
    ax[1].scatter(stim_data_train.spikes,np.zeros(len(stim_data_train.spikes)),marker="|",color="k")
    ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("SDF (spikes/s)") # Label the SDF axes
    plt.suptitle("Baseline SDF Bin Area Histogram and Stimulus SDF") # Title the figure
    makeNice([ax[0],ax[1]]) # Clean the figure
    plt.savefig(trial_direc+"/sdf_shuffling.pdf") # Save the figure
    plt.close() # Close the figure
    return results # Return the array containing the values of the bins excited (1) or not (0)
