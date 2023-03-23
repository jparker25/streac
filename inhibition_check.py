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

def moving_mean(x,window=250):
    """
    Create a moving mean over a specified window.

    :param x: Array to create moving mean out of.
    :keyword param window: Size of the window for the average, default 250.
    """
    movmean = np.zeros(x.shape[0]) # Create an array that is the same size as the input to store moving average
    for i in range(window//2,x.shape[0]-window//2): # Iterate through all points that have full half window around it
        movmean[i] = np.mean(x[i-(window//2):i+(window//2)+(window%2)]) # Find the moving mean at the point in question
    for i in range(0,window//2): # Iterate through points at the front and find moving mean 
        movmean[i] = np.mean(x[0:i+window//2+(window%2)]) # Find the moving mean at the point in question
    for i in range(x.shape[0]-window//2,x.shape[0]): # Iterate through points at the back and find moving mean
        movmean[i] = np.mean(x[i-window//2:]) # Fin the moving mean at the point in question
    return movmean # Return moving mean
        
def isi_function(spikes,t,avg=250):
    """
    Generate the ISIF for a spike train based on sample points and a window.

    :param spikes: Array of spike train to create ISIF.
    :param t: Array to time for to sample ISIF
    :keyword param avg: Window of moving average, default 250.
    :return isif: Moving average of interpolated ISI function.
    """
    isis = np.diff(spikes) # Create ISIs for spike train
    tEnd = t[-1] - spikes[-1] # Find the time from the last spike to the end of train
    interp = np.interp(t,spikes[:-1],np.diff(spikes)) # Degree 1 interpolation onto 't' for spike times and ISIs
    if isis[-1] <= tEnd: # If the last ISI is smaller than the time to the end of recording
        interp[t >= spikes[-1]] = tEnd # Set all times of interpolation after last spike to the total time remaining
    else: # If the last ISI is larger than the time to the end of the recording
        interp[t >= spikes[-1]] = np.mean([tEnd,isis[-1]]) # Average the last ISI and tEnd and set all values after last spike
    if isis[0] <= spikes[0]: # If the first ISI is smaller than the first spike time
        interp[t <= spikes[0]] = np.mean([spikes[0],isis[0]]) # Set all times before the first spike to the average of the first spike and first ISI
    else: # If the first ISI is larger than the first spike
        interp[t <= spikes[0]] = spikes[0] # Set all times before the first spike to the time of the first spike
    isif = moving_mean(interp,window=avg) # Find the moving average of the interpolated ISI
    return isif # Return the ISIF


def avg_inhibition_check(neuron,baselines,stimuli,percentile=99,bin_width=0.5):
    fig,ax = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create figure that will have 2 rows and 1 column
    results = np.zeros(len(stimuli[0].bin_edges)-1) # Create empty array that will store the classifications of each bin

    isi_bls = np.asarray([baseline.isif for baseline in baselines if baseline.spikes.size > 1])
    isi_stims = np.asarray([stimulus.isif for stimulus in stimuli if stimulus.spikes.size > 1])
    avg_isi_stim = np.mean(isi_stims,axis=0)

    bl_areas = []
    for isi_bl in isi_bls:
        for i in range(len(baselines[0].bin_edges)-1):
            st,ed = np.where(baselines[0].t <= baselines[0].bin_edges[i])[0][-1],np.where(baselines[0].t < baselines[0].bin_edges[i+1])[0][-1]
            bl_areas.append(np.trapz(isi_bl[st:ed+1],x=baselines[0].t[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array

    for i in range(len(stimuli[0].bin_edges)-1):
        st,ed = np.where(stimuli[0].t <= stimuli[0].bin_edges[i])[0][-1],np.where(stimuli[0].t < stimuli[0].bin_edges[i+1])[0][-1]
        if np.percentile(bl_areas,percentile) < np.trapz(avg_isi_stim[st:ed+1],x=stimuli[0].t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
            ax[1].fill_between(stimuli[0].t[st:ed+1],avg_isi_stim[st:ed+1],color="blue",alpha=0.5) # Fill the area of the SDF that is beyond the percentile of the baseline areas
            results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
    ax[0].hist(bl_areas,bins=20,color="gray",edgecolor="k") # Create a histogram of all the baseline SDF areas
    ax[0].vlines(np.percentile(bl_areas,percentile),ax[0].get_ylim()[0],ax[0].get_ylim()[1],color="k",linestyle="dashed") # Draw a verticle line where the percentile threshold is on the histogram
    ax[0].set_xlabel("Baseline ISI Area Bins"); ax[0].set_ylabel("Count") # Label the histogram axes
    ax[1].plot(stimuli[0].t,avg_isi_stim,color="blue") # Plot the SDF as a function of t
    ax[1].vlines(stimuli[0].bin_edges,0,ax[1].get_ylim()[1],color="k",linestyle="dashed") # Draw vertical lines for bin edges on the SDF stimulus plot
    ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("ISI Function") # Label the SDF axes
    plt.suptitle("Average Baseline ISI Bin Area Histogram and Average Stimulus ISI Function") # Title the figure
    makeNice([ax[0],ax[1]])
    plt.savefig(f"{neuron.cell_dir}/avg_isi_shuffling.pdf") # Save the figure
    plt.close() # Close the figure
    return results # Return the array containing the values of the bins excited (1) or not (0)

def trial_inhibition_check(baseline_train,stim_data_train,trial_direc,shuffles=10,percentile=99):
    fig,ax = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create figure that will have 2 rows and 1 column
    results = np.zeros(len(stim_data_train.bin_edges)-1) # Create empty array that will store the classifications of each bin
    isi_stim = stim_data_train.isif# Create SDF for baseline and stimulus spikes
    bl_isi = np.diff(baseline_train.spikes) # Find the baseline ISI values
    bl_shuffles = [np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(shuffles)] # Shuffle the bl ISI values for shuffles number of times
    bl_spikes = [[baseline_train.spikes[0]+sum(bl_shuff[0:i]) if i > 0 else baseline_train.spikes[0] for i in range(len(bl_isi))] for bl_shuff in bl_shuffles] # Recreate the spikes based on the shuffled ISI values
    bl_isi_fs = [isi_function(bl_spike,baseline_train.t) for bl_spike in bl_spikes]

    bl_areas = [] # Empty array that will store the areas of each bin of the shuffled baseline
    for i in range(shuffles): # Loop through all the shuffles
        for bi in range(len(baseline_train.bin_edges)-1): # In each shuffle, loop throuhg each bin
            st,ed = np.where(baseline_train.t <= baseline_train.bin_edges[bi])[0][-1],np.where(baseline_train.t < baseline_train.bin_edges[bi+1])[0][-1]
            bl_areas.append(np.trapz(bl_isi_fs[i][st:ed+1],x=baseline_train.t[st:ed+1])) # Find the area of the baseline SDF of the bin and append to array

    first_bin_edge = np.where(stim_data_train.bin_edges <= stim_data_train.spikes[0])[0][-1] # Find the bin edge that is before the first stimulus spike
    last_bin_edge = np.where(stim_data_train.bin_edges < stim_data_train.spikes[-1])[0][-1] # Find the bin edge that is before the last stimulus spike

    for i in range(len(stim_data_train.bin_edges)-1): # Iterate through each bin
        if i == first_bin_edge:
            st,ed = np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],np.where(stim_data_train.t < stim_data_train.bin_edges[i+1])[0][-1]
            isi_stim_bin = stim_data_train.isif[st:ed+1] # Find the SDF of the bin
            if stim_data_train.spikes[0] >= np.percentile(bl_areas,percentile):
                ax[1].fill_between(stim_data_train.t[0:st],stim_data_train.isif[0:st],color="blue",alpha=0.5)
                results[0:i] += 1
            if np.percentile(bl_areas,percentile) < np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
                ax[1].fill_between(stim_data_train.t[st:ed+1],isi_stim_bin,color="blue",alpha=0.5) # Fill the area of the SDF that is beyond the percentile of the baseline areas
                results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
        elif i == last_bin_edge:
            st,ed = np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],np.where(stim_data_train.t < stim_data_train.bin_edges[i+1])[0][-1]
            isi_stim_bin = stim_data_train.isif[st:ed+1] # Find the SDF of the bin
            if stim_data_train.time-stim_data_train.spikes[-1] >= np.percentile(bl_areas,percentile) and i != len(stim_data_train.bin_edges)-2:
                results[i+1:] += 1
                ax[1].fill_between(stim_data_train.t[ed+1:],stim_data_train.isif[ed+1:],color="blue",alpha=0.5)
            if np.percentile(bl_areas,percentile) < np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
                ax[1].fill_between(stim_data_train.t[st:ed+1],isi_stim_bin,color="blue",alpha=0.5) # Fill the area of the SDF that is beyond the percentile of the baseline areas
                results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
        elif i > first_bin_edge and i < last_bin_edge:
            st,ed = np.where(stim_data_train.t <= stim_data_train.bin_edges[i])[0][-1],np.where(stim_data_train.t < stim_data_train.bin_edges[i+1])[0][-1]
            isi_stim_bin = stim_data_train.isif[st:ed+1] # Find the SDF of the bin
            if np.percentile(bl_areas,percentile) < np.trapz(isi_stim_bin,x=stim_data_train.t[st:ed+1]): # See if area of bin is beyond the percentile of the baseline shuffled areas
                ax[1].fill_between(stim_data_train.t[st:ed+1],isi_stim_bin,color="blue",alpha=0.5) # Fill the area of the SDF that is beyond the percentile of the baseline areas
                results[i] += 1 # Make the classification of the bin a 1 indicating it is excited
    ax[0].hist(bl_areas,bins=20,color="gray",edgecolor="k") # Create a histogram of all the baseline SDF areas
    ax[0].vlines(np.percentile(bl_areas,percentile),ax[0].get_ylim()[0],ax[0].get_ylim()[1],color="k",linestyle="dashed") # Draw a verticle line where the percentile threshold is on the histogram
    ax[0].set_xlabel("Shuffled Baseline ISI Area Bins"); ax[0].set_ylabel("Count") # Label the histogram axes
    ax[1].plot(stim_data_train.t,stim_data_train.isif,color="blue",label="Stim ISI Interpolation") # Plot the SDF as a function of t
    ax[1].vlines(stim_data_train.bin_edges,ax[1].get_ylim()[0],ax[1].get_ylim()[1],color="k",linestyle="dashed",alpha=0.5) # Draw vertical lines for bin edges on the SDF stimulus plot
    ax[1].scatter(stim_data_train.spikes,np.zeros(len(stim_data_train.spikes)),marker="|",color="k")
    ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("ISI Function") # Label the SDF axes
    plt.suptitle("Baseline ISI Bin Area Histogram and Stimulus ISI Function") # Title the figure
    makeNice([ax[0],ax[1]])
    plt.savefig(trial_direc+"/isi_shuffling.pdf") # Save the figure
    plt.close() # Close the figure
    return results # Return the array containing the values of the bins excited (1) or not (0)

