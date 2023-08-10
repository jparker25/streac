################################################################################
# stimulus_classification.py
# Classifies stimulus response compared to baseline.
# Author: John E. Parker
################################################################################

# import python modules
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# import user made modules
from helpers import *
import excitation_check as exch
import inhibition_check as inch

def consecutive_bins(res_bins):
    """
    Function that determines lenght of longest consecutive bins.

    :param res_bins: array of 0's and 1's indicate which bins are valid (1s)
    :return longest: array of longest consecutive bins
    :return long_len: length of longest
    """
    consecs = [] # Empty array to fill with sets of consecutive bins
    for i in range(len(res_bins)-1): # Iterate through all bins
        if res_bins[i] == 1: # Only keep bin if a 1
            tmp_consec = [i+1] # Add bin to a temp array
            valid = True # Valid if conseuctive
            for j in range(i+1,len(res_bins)): # Iterate through remaining bins
                if res_bins[j] == 1 and res_bins[j-1] == 1 and valid: # Only store if adjacent bin is a 1
                    tmp_consec.append(j+1)
                else: # No longer store and set valid to false
                    valid = False
            consecs.append(tmp_consec)
    long_len = 0 # Longest consecutive by default is 0
    longest = [] # Array to find longest consecutive bins
    for con in consecs: # Iterate through all consecutive bin lengths
        if len(con) > long_len: # Update length and longest if exceeds current length
            long_len = len(con)
            longest = con
    # return longest consecutive array and length
    return longest, long_len 


def classify_response(res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,con_in_bin_threshold,con_ex_bin_threshold,bin_edges):
    """
    Function that classifies trial response based on number of inhibited and excited bins.

    :param res_inhibit: binary list of bins that are inhibited (1 inhibited, 0 not)
    :param res_excite: binary list of bins that are excited (1 excited, 0 not)
    :param in_bin_threshold: number of bins for spike train to be inhibited response
    :param ex_bin_threshold: number of bins for spike train to be excited response
    :param con_in_bin_threshold: number of consecutive bins for spike train to be inhibited response
    :param con_ex_bin_threshold: number of consecutive bins for spike train to be excited response
    :param bin_edges: list of the bin edges for the spike train
    :return list of number of NE, IN, and EX bins, response type, and values for each bin 
    """
    res = np.asarray([len(bin_edges)-1-sum(res_excite)-sum(res_inhibit), sum(res_inhibit), sum(res_excite)]) # Create array that counts number of no effect, inhibited, and excited bins
    class_results = [] # Results that stores the integer corresponding to the bin classification
    for i in range(len(res_excite)): # Loop through all the bins and determine the type of classification for the bin
        val = int(res_excite[i]+res_inhibit[i]) # Sum the results of the excitation and inhibition for the corresponding bin
        if val == 1: # If val is one, then either excited or inhibited
            class_results.append(1) if res_inhibit[i] == 1 else class_results.append(2) # Append 1 if inhibited or 2 if excited for class_results
        elif val == 2: # If val is two then both excited and inhibited
            class_results.append(val+1) # Append 3 to class_results
        else: # If val is neither 1 nor 2, then must be 0 and no effect
            class_results.append(val) # Append val as 0 to class_results for no effect bin

    consec_in, len_consec_in = consecutive_bins(res_inhibit) # Find consecutive inhibited bins
    consec_ex, len_consec_ex = consecutive_bins(res_excite) # Find consecutive excited bins

    type = None # Variable to given response type

    #### ADAPTING INHIBITION ####
    if  ((con_ex_bin_threshold == 0 or len_consec_ex < con_ex_bin_threshold) and np.sum(res_excite) < ex_bin_threshold) and ((len_consec_in >= con_in_bin_threshold and con_in_bin_threshold > 0) or np.sum(res_inhibit) >= in_bin_threshold) and np.sum(res_inhibit[len(res_inhibit)//2:]) < np.sum(res_inhibit[:len(res_inhibit)//2]):
        type = "adapting inhibition"
    
    #### PARTIAL INHIBITION ####
    elif  ((con_ex_bin_threshold == 0 or len_consec_ex < con_ex_bin_threshold) and np.sum(res_excite) < ex_bin_threshold) and ((len_consec_in >= con_in_bin_threshold and con_in_bin_threshold > 0) or np.sum(res_inhibit) >= in_bin_threshold):
        type = "partial inhibition"

    #### EXCITATION  ####
    elif  ((con_ex_bin_threshold > 0 and len_consec_ex >= con_ex_bin_threshold) or np.sum(res_excite) >= ex_bin_threshold) and ((con_in_bin_threshold == 0 or len_consec_in < con_in_bin_threshold) and np.sum(res_inhibit) < in_bin_threshold):
        type = "excitation"

    #### BPIE  ####
    elif  ((con_ex_bin_threshold > 0 and len_consec_ex >= con_ex_bin_threshold) or np.sum(res_excite) >= ex_bin_threshold) and ((con_in_bin_threshold > 0 and len_consec_in >= con_in_bin_threshold) or np.sum(res_inhibit) >= in_bin_threshold) and np.where(res_excite == 1)[0][-1] >= np.where(res_inhibit == 1)[0][-1]:
        type = "biphasic IE"
    
    #### BPEI ####
    elif  ((con_ex_bin_threshold > 0 and len_consec_ex >= con_ex_bin_threshold) or np.sum(res_excite) >= ex_bin_threshold) and ((con_in_bin_threshold > 0 and len_consec_in >= con_in_bin_threshold) or np.sum(res_inhibit) >= in_bin_threshold) and np.where(res_excite == 1)[0][-1] <= np.where(res_inhibit == 1)[0][-1]:
        type = "biphasic EI"

    ### NO EFFECT ####
    # If no other response, then no effect
    else: 
        type = "no effect"
    # Return designated count of number of each bin, response type, and bin results
    return res, type, class_results

def avg_classify_response(neuron,res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,con_in_bin_threshold,con_ex_bin_threshold,bin_edges):
    """
    Function that classifies average response based on number of inhibited and excited bins.

    :param neuron: neuron class to determine average response
    :param res_inhibit: binary list of bins that are inhibited (1 inhibited, 0 not)
    :param res_excite: binary list of bins that are excited (1 excited, 0 not)
    :param in_bin_threshold: number of bins for spike train to be inhibited response
    :param ex_bin_threshold: number of bins for spike train to be excited response
    :param con_in_bin_threshold: number of consecutive bins for spike train to be inhibited response
    :param con_ex_bin_threshold: number of consecutive bins for spike train to be excited response
    :param bin_edges: list of the bin edges for the spike train
    :return list of number of NE, IN, and EX bins, response type, and values for each bin 
    """
    res = np.asarray([len(bin_edges)-1-sum(res_excite)-sum(res_inhibit), sum(res_inhibit), sum(res_excite)]) # Create array that counts number of no effect, inhibited, and excited bins
    class_results = [] # Results that stores the integer corresponding to the bin classification
    for i in range(len(res_excite)): # Loop through all the bins and determine the type of classification for the bin
        val = int(res_excite[i]+res_inhibit[i]) # Sum the results of the excitation and inhibition for the corresponding bin
        if val == 1: # If val is one, then either excited or inhibited
            class_results.append(1) if res_inhibit[i] == 1 else class_results.append(2) # Append 1 if inhibited or 2 if excited for class_results
        elif val == 2: # If val is two then both excited and inhibited
            class_results.append(val+1) # Append 3 to class_results
        else: # If val is neither 1 nor 2, then must be 0 and no effect
            class_results.append(val) # Append val as 0 to class_results for no effect bin

    consec_in, len_consec_in = consecutive_bins(res_inhibit) # Find consecutive inhibited bins
    consec_ex, len_consec_ex = consecutive_bins(res_excite) # Find consecutive excited bins

    type = None # Variable to given response type

    #### ADAPTING INHIBITION ####
    if  ((con_ex_bin_threshold == 0 or len_consec_ex < con_ex_bin_threshold) and np.sum(res_excite) < ex_bin_threshold) and ((len_consec_in >= con_in_bin_threshold and con_in_bin_threshold > 0) or np.sum(res_inhibit) >= in_bin_threshold) and np.sum(res_inhibit[len(res_inhibit)//2:]) < np.sum(res_inhibit[:len(res_inhibit)//2]):
        type = "adapting inhibition"
    
    #### PARTIAL INHIBITION ####
    elif  ((con_ex_bin_threshold == 0 or len_consec_ex < con_ex_bin_threshold) and np.sum(res_excite) < ex_bin_threshold) and ((len_consec_in >= con_in_bin_threshold and con_in_bin_threshold > 0) or np.sum(res_inhibit) >= in_bin_threshold):
        type = "partial inhibition"

    #### EXCITATION  ####
    elif  ((con_ex_bin_threshold > 0 and len_consec_ex >= con_ex_bin_threshold) or np.sum(res_excite) >= ex_bin_threshold) and ((con_in_bin_threshold == 0 or len_consec_in < con_in_bin_threshold) and np.sum(res_inhibit) < in_bin_threshold):
        type = "excitation"

    #### BPIE  ####
    elif  ((con_ex_bin_threshold > 0 and len_consec_ex >= con_ex_bin_threshold) or np.sum(res_excite) >= ex_bin_threshold) and ((con_in_bin_threshold > 0 and len_consec_in >= con_in_bin_threshold) or np.sum(res_inhibit) >= in_bin_threshold) and np.where(res_excite == 1)[0][-1] >= np.where(res_inhibit == 1)[0][-1]:
        type = "biphasic IE"
    
    #### BPEI ####
    elif  ((con_ex_bin_threshold > 0 and len_consec_ex >= con_ex_bin_threshold) or np.sum(res_excite) >= ex_bin_threshold) and ((con_in_bin_threshold > 0 and len_consec_in >= con_in_bin_threshold) or np.sum(res_inhibit) >= in_bin_threshold) and np.where(res_excite == 1)[0][-1] <= np.where(res_inhibit == 1)[0][-1]:
        type = "biphasic EI"

    ### NO EFFECT ####
    # If no other response, then no effect
    else: 
        type = "no effect"
    # Return designated count of number of each bin, response type, and bin results
    return res, type, class_results

def trial_sdf_isi_functions_classification(neuron,baseline, stim_data,trial_direc,isif_inhibition,percentile=99,in_bin_threshold=3,ex_bin_threshold=3,con_in_bin_threshold=2,con_ex_bin_threshold=2):
    '''
    Function that determines how the stimulus responds as compared to the baseline with the light on. Based on comparing ISI peak values for an ISI function.

    :param neuron: neuron class to determine trial response
    :param baseline: array of spike times for the baseline
    :param stim_data: array of spike times for the stimulus
    :param trial_direc: string that is the path of the trial directory to store and grab all data for the current trial
    :param isif_inhibition: boolean to determine to use ISIF or SDF
    :keyword param percentile: integer defining what the percentile should be to check the classification for significance
    :keyword param in_bin_threshold: number of bins for spike train to be inhibited response
    :keyword param ex_bin_threshold: number of bins for spike train to be excited response
    :keyword param con_in_bin_threshold: number of consecutive bins for spike train to be inhibited response
    :keyword param con_ex_bin_threshold: number of consecutive bins for spike train to be excited response
    '''
    if len(stim_data.spikes) <= 5: # If the stimulus has 5 or less spikes, classify as complete inhibition
        return np.array([0,len(stim_data.bin_edges)-1,0]),"complete inhibition", np.ones(len(stim_data.bin_edges)-1) # Return all bins as inhibited, classification as complete inhibition, and array of bin classes as inhibitied
    # Call the function that checks for excited bins, returns array of 0 (not excited) or 1 (excited) corresponding to bin classifications
    res_excite = exch.trial_excitation_check(neuron,baseline,stim_data,trial_direc,percentile=percentile)
    # Call the function that checks for inhibited bins, returns array of 0 (not inhibited) or 1 (inhibited) corresponding to bin classifications
    res_inhibit = inch.trial_inhibition_check_isif(neuron,baseline,stim_data,trial_direc,percentile=percentile) if isif_inhibition else inch.trial_inhibition_check_sdf(neuron,baseline,stim_data,trial_direc,percentile=100-percentile) 

    np.savetxt(trial_direc+"/results_excitation.txt",res_excite,delimiter="\t",newline="\n",fmt="%f",header="Stimulus bin results for excitation check. 1 means bin excited.") # Save the results of excitation check
    np.savetxt(trial_direc+"/results_inhibition.txt",res_inhibit,delimiter="\t",newline="\n",fmt="%f",header="Stimulus bin results for inhibition check. 1 means bin inhibited.") # Save the results of inhibition check
    # Determine the response
    return classify_response(res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,con_in_bin_threshold,con_ex_bin_threshold,stim_data.bin_edges)

def average_sdf_isi_functions_classification(neuron,bin_width=0.5,percentile=90,in_bin_threshold=3,ex_bin_threshold=3,con_in_bin_threshold=2,con_ex_bin_threshold=2):
    '''
    Function that determines how the average response after processing the SDF and ISIF to find average excited and inhibitory periods.

    :param neuron: neuron class to determine trial response
    :keyword param bin_width: width of bins
    :keyword param percentile: integer defining what the percentile should be to check the classification for significance
    :keyword param in_bin_threshold: number of bins for spike train to be inhibited response
    :keyword param ex_bin_threshold: number of bins for spike train to be excited response
    :keyword param con_in_bin_threshold: number of consecutive bins for spike train to be inhibited response
    :keyword param con_ex_bin_threshold: number of consecutive bins for spike train to be excited response
    '''
    baselines = []; stimuli = []; trial_results = [] # Empty arrays to store trial information
    for trial in range(1,neuron.trials+1): # Iterate throug all trials
        # Store baseline spike train object
        baselines.append(pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb")))
        # Store stimulus spike train object
        stimuli.append(pickle.load(open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")))
        # Store bin results for each trials
        trial_results.append(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/bin_results.txt"))
    cmap = ["gray","cyan","red","green"] # Color map to code for classification of bin (NE, IN, EX, both)
    fig,axe = plt.subplots(1,1,figsize=(12,6),dpi=300) # Create a new figure
    for trial in range(neuron.trials): # Iterate thorugh all trials
        axe.scatter(baselines[trial].spikes-baselines[trial].time,np.ones(len(baselines[trial].spikes))*(trial+1),marker="|",color="k",s=200) # Plot the baseline raster
        axe.scatter(stimuli[trial].spikes,np.ones(len(stimuli[trial].spikes))*(trial+1),marker="|",color="blue",s=200) # Plot the stimulus raster
        for i in range(len(stimuli[trial].bin_edges)-1): # Iterate through each bin and color code by response
            axe.add_patch(Rectangle(xy=(np.mean([stimuli[trial].bin_edges[i],stimuli[trial].bin_edges[i+1]])-bin_width/4,trial+0.5-bin_width/4),width=bin_width/2,height=bin_width/2,color=cmap[int(trial_results[trial][i])],fill=True))
    axe.hlines(neuron.trials+0.5,0,stimuli[0].time,color='k',linewidth=5,label="Light") # horizontal line that signifies light on period
    axe.vlines(stimuli[0].bin_edges,0,neuron.trials+1,color="k",linestyle="dashed",alpha=0.25,linewidth=0.5) # Draw vertical lines for bin edges on the SDF stimulus plot
    axe.vlines(baselines[0].bin_edges-baselines[0].time,0,neuron.trials+1,color="k",linestyle="dashed",alpha=0.25,linewidth=0.5) # Draw vertical lines for bin edges on the SDF baselien plot
    axe.set_ylim([0,neuron.trials+0.75]) # Set ylims
    axe.set_xlim([-neuron.baseline_length,neuron.trial_length]) # Set xlims
    axe.set_yticks([k for k in range(1,neuron.trials+1)]) # Set yticks
    axe.spines['right'].set_visible(False) # Remove spines
    axe.spines['top'].set_visible(False) # Remove spines
    plt.suptitle("All Baseline (black) and Trials (blue)") # Set title
    fig.text(0.06, 0.5, 'Trials', ha='center', va='center', rotation='vertical') # Label figure
    fig.text(0.5, 0.04, 't', ha='center', va='center')
    makeNice(axe) # Clean up figure
    plt.savefig(f"{neuron.cell_dir}/all_trial_spike_trains.pdf") # Save the figure in trial direc
    plt.close() # Close the figure

    # Only for low rate stimulus neurons
    if (np.mean([len(stimulus.spikes) for stimulus in stimuli]) <= 5) and (np.mean([len(baseline.spikes) for baseline in baselines]) <= 5): # No effect if average spikes below 5 in baseline and stimulus
        return np.array([len(stimuli[0].bin_edges)-1,0,0]),"no effect", np.zeros(len(stimuli[0].bin_edges)-1)
    if (np.mean([len(stimulus.spikes) for stimulus in stimuli]) <= 5): # No effect if average spikes below 5 in stimulus
        return np.array([0,len(stimuli[0].bin_edges)-1,0]),"complete inhibition", np.ones(len(stimuli[0].bin_edges)-1)

    res_excite = exch.avg_excitation_check(neuron,baselines,stimuli,percentile=percentile) # Find average excited bins
    res_inhibit = inch.avg_inhibition_check(neuron,baselines,stimuli,percentile=percentile) # Find average inhibited bins
    np.savetxt(neuron.cell_dir+"/avg_results_excitation.txt",res_excite,delimiter="\t",newline="\n",fmt="%f",header="Average Stimulus bin results for excitation check. 1 means bin excited.") # Save the results of excitation check
    np.savetxt(neuron.cell_dir+"/avg_results_inhibition.txt",res_inhibit,delimiter="\t",newline="\n",fmt="%f",header="Average Stimulus bin results for inhibition check. 1 means bin inhibited.") # Save the results of inhibition check
    # Return average classificaition
    return avg_classify_response(neuron, res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,con_in_bin_threshold,con_ex_bin_threshold,stimuli[0].bin_edges) # Determine 
