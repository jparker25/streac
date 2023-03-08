################################################################################
# stimulus_classification.py
# Classifies stimulus response compared to baseline.
# Author: John E. Parker
################################################################################

# import python modules
import numpy as np
import sys, os, pickle
from scipy.stats import norm, mannwhitneyu
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.stats import binom
import sys



from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.signal import savgol_filter

# import user made modules
from helpers import *
#import process_neural_data as pnd

def moving_mean(x,window=250):
    movmean = np.zeros(x.shape[0])
    shift = 1 if (window%2) else 0
    for i in range(window//2-shift,x.shape[0]-window//2):
        movmean[i] = np.mean(x[i-(window//2-shift):i+(window//2)])
    for i in range(0,window//2-shift):
        left = x[0:i]
        right = x[i:i+window//2]
        movmean[i] = (np.sum(left)+np.sum(right)) / (left.shape[0]+right.shape[0])
    for i in range(x.shape[0]-window//2,x.shape[0]):
        left = x[i-window//2:i]
        right = x[i:]
        movmean[i] = (np.sum(left)+np.sum(right)) / (left.shape[0]+right.shape[0])
    return movmean
        


def isi_function(spikes,t,avg=250,extend=True):
    isis = np.diff(spikes)
    tEnd = t[-1] - spikes[-1]
    '''
    if isis[-1] <= tEnd:
        isis = np.append(isis,tEnd)
    else:
        isis = np.append(isis,np.mean([isis[-1],tEnd]))
    '''
    #isif = np.convolve(np.interp(t,spikes[:-1],np.diff(spikes)),np.ones(avg)/avg,mode='same')
    #isif = np.convolve(np.interp(t,spikes,isis),np.ones(avg)/avg,mode='same')
    interp = np.interp(t,spikes[:-1],np.diff(spikes))
    if isis[-1] <= tEnd:
        interp[t >= spikes[-1]] = tEnd
    else:
        interp[t >= spikes[-1]] = np.mean([tEnd,isis[-1]])
    interp[t <= spikes[0]] = isis[0]
    isif = moving_mean(interp,window=avg)
    #isif = np.convolve(interp,np.ones(avg)/avg,mode='same') 
    '''
    if extend and spikes[0] > t[1]:
        first_spike = np.where(t < spikes[0])[0][-1] #if len(spikes) > 1 else np.where(t < spikes[0])[0][-1]
        isif[0:first_spike+1] = np.ones(first_spike+1)*isif[first_spike]
    '''
    return isif

def gaussian(x,s):
    '''
    Function that takes in values x and standard deviation s to determine Gaussian distribution function
        arr x -> x-values to evaluate guassian at each point
        float s -> standard deviation for the gaussian distribution function
    '''
    # Return the gaussian distribution function with standard deviation s at values x
    return (1/(np.sqrt(2*np.pi)*s))*np.exp(-x**2/(2*s**2))

def kernel(values,spacing,bandwidth=25/1000):
    '''
    Function that is the kernel for the spike density function, returns the SDF.
        arr values -> array of values to evaluate SDF at, should be spikes
        arr spacing -> grid to compute SDF at, should be a desired domain
        int bandwidth -> number of points that the SDF should be spread over, default is 1
    '''
    res = np.zeros(len(spacing)) # Create empty grid that will store the SDF values, length of desired grid
    for j in range(len(values)): # Iterate through each data point and find the SDF via the Gaussian kernel
        res += gaussian((spacing-values[j]),bandwidth) # Evaluate the Gaussian kernel at every point for the given value
    return res # Return an array that is the SDF values along the domain spacing

def classify_response(res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,bin_edges):
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
    consec_in = np.asarray([[i+1,i+2] for i in range(len(res_inhibit)-1) if sum(res_inhibit[i:i+2]) == 2])
    consec_ex = np.asarray([[i+1,i+2] for i in range(len(res_excite)-1) if sum(res_excite[i:i+2]) == 2])

    type = None

    #### ADAPTING INHIBITION ####
    if (np.sum(res_inhibit[:len(res_inhibit)//2]) >= in_bin_threshold and np.sum(res_inhibit[len(res_inhibit)//2:]) < np.sum(res_inhibit[:len(res_inhibit)//2]) and np.sum(res_excite) < ex_bin_threshold):
        type = "adapting inhibition"
    elif (len(consec_in) == 1 and len(consec_ex) == 0 and consec_in[0,0] < int(len(res_inhibit)/2)):
        type = "adapting inhibition"

    #### PARTIAL INHIBITION ####
    elif (np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) < ex_bin_threshold):
        type = "partial inhibition"
    elif (len(consec_in) == 1 and len(consec_ex) == 0):
        type = "partial inhibition"

    #### EXCITATION  ####
    elif (np.sum(res_inhibit) < in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold):
        type = "excitation"
    elif (len(consec_in) == 0 and len(consec_ex) == 1):
        type = "excitation"

    #### BPIE  ####
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1] >= np.where(res_inhibit == 1)[0][-1]:
        type = "biphasic IE"
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) < ex_bin_threshold and consec_ex[0,1] >= np.where(res_inhibit == 1)[0][-1]+1 and len(consec_ex) == 1 and len(consec_in) == 0:
        type = "biphasic IE"
    elif np.sum(res_inhibit) < in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1]+1 >= consec_in[0,1] and len(consec_ex) == 0 and len(consec_in) == 1:
        type = "biphasic IE"
    elif len(consec_ex) == 1 and len(consec_in) == 1 and consec_ex[0,1] < consec_in[0,1]:
        type = "biphasic IE"
    #### BPEI ####
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1] <= np.where(res_inhibit == 1)[0][-1]:
        type = "biphasic EI"
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) < ex_bin_threshold and consec_ex[0,1] <= np.where(res_inhibit == 1)[0][-1]+1 and len(consec_ex) == 1 and len(consec_in) == 0:
        type = "biphasic EI"
    elif np.sum(res_inhibit) < in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1]+1 <= consec_in[0,1] and len(consec_ex) == 0 and len(consec_in) == 1:
        type = "biphasic EI"
    elif len(consec_ex) == 1 and len(consec_in) == 1 and consec_ex[0,1] > consec_in[0,1]:
        type = "biphasic EI"

    ### NO EFFECT ####
    else:
        type = "no effect"

    return res, type, class_results
    '''
    if np.sum(res_inhibit[:len(res_inhibit)//2]) >= in_bin_threshold and np.sum(res_inhibit[len(res_inhibit)//2:]) < np.sum(res_inhibit[:len(res_inhibit)//2]) and np.sum(res_excite) < ex_bin_threshold:
        return res, "adapting inhibition",class_results
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) < ex_bin_threshold:
        return res, "partial inhibition",class_results
    elif np.sum(res_inhibit) < in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold:
        return res, "excitation",class_results
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1] >= np.where(res_inhibit == 1)[0][-1]:
        return res, "biphasic IE",class_results
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_inhibit == 1)[0][-1] >= np.where(res_excite == 1)[0][-1]:
        return res, "biphasic EI",class_results
    else:
        return res, "no effect",class_results # Return the number of bins no effect, inhbited, and excited, the classification, and the corresponding bin class integers
    '''

def avg_classify_response(neuron,res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,bin_edges):

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

    li_true, li_bins = local_inhibition_check(neuron)
    #print(li_bins)
    consec_in = np.asarray([[i+1,i+2] for i in range(len(res_inhibit)-1) if sum(res_inhibit[i:i+2]) == 2])
    consec_ex = np.asarray([[i+1,i+2] for i in range(len(res_excite)-1) if sum(res_excite[i:i+2]) == 2])

    type = None

    #### ADAPTING INHIBITION ####
    if (np.sum(res_inhibit[:len(res_inhibit)//2]) >= in_bin_threshold and np.sum(res_inhibit[len(res_inhibit)//2:]) < np.sum(res_inhibit[:len(res_inhibit)//2]) and np.sum(res_excite) < ex_bin_threshold):
        type = "adapting inhibition"
    elif (len(consec_in) == 1 and len(consec_ex) == 0 and consec_in[0,0] < int(len(res_inhibit)/2)):
        type = "adapting inhibition"

    #### PARTIAL INHIBITION ####
    elif (np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) < ex_bin_threshold):
        type = "partial inhibition"
    elif (len(consec_in) == 1 and len(consec_ex) == 0):
        type = "partial inhibition"

    #### EXCITATION  ####
    elif (np.sum(res_inhibit) < in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold):
        type = "excitation"
    elif (len(consec_in) == 0 and len(consec_ex) == 1):
        type = "excitation"

    #### BPIE  ####
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1] >= np.where(res_inhibit == 1)[0][-1]:
        type = "biphasic IE"
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) < ex_bin_threshold and consec_ex[0,1] >= np.where(res_inhibit == 1)[0][-1]+1 and len(consec_ex) == 1 and len(consec_in) == 0:
        type = "biphasic IE"
    elif np.sum(res_inhibit) < in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1]+1 >= consec_in[0,1] and len(consec_ex) == 0 and len(consec_in) == 1:
        type = "biphasic IE"
    elif len(consec_ex) == 1 and len(consec_in) == 1 and consec_ex[0,1] < consec_in[0,1]:
        type = "biphasic IE"
    #### BPEI ####
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1] <= np.where(res_inhibit == 1)[0][-1]:
        type = "biphasic EI"
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) < ex_bin_threshold and consec_ex[0,1] <= np.where(res_inhibit == 1)[0][-1]+1 and len(consec_ex) == 1 and len(consec_in) == 0:
        type = "biphasic EI"
    elif np.sum(res_inhibit) < in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1]+1 <= consec_in[0,1] and len(consec_ex) == 0 and len(consec_in) == 1:
        type = "biphasic EI"
    elif len(consec_ex) == 1 and len(consec_in) == 1 and consec_ex[0,1] > consec_in[0,1]:
        type = "biphasic EI"

    ### NO EFFECT ####
    else:
        type = "no effect"

    ######### LOCAL INHIBITION CHECK ##########
    if type == "excitation" and li_true:
        ### BIPHASIC RESPONSE ###
        if li_bins[1] <= np.where(res_excite == 1)[0][-1]+1:
            type = "biphasic IE"
        elif li_bins[1] >= np.where(res_excite == 1)[0][-1]+1:
            type = "biphasic EI"
        else:
            type = -1
            print("ERROR: LOCAL INHIBITION AND EXCITATION")
    elif type == "no effect" and li_true:
        ### ADAPTING INHIBITION
        if li_bins[0] < int(len(res_inhibit)/2):
            type = "adapting inhibition"
        else:
            type = "partial inhibition"


    return res, type, class_results




    '''
    if np.sum(res_inhibit[:len(res_inhibit)//2]) >= in_bin_threshold and np.sum(res_inhibit[len(res_inhibit)//2:]) < np.sum(res_inhibit[:len(res_inhibit)//2]) and np.sum(res_excite) < ex_bin_threshold:
        return res, "adapting inhibition",class_results
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) < ex_bin_threshold:
        return res, "partial inhibition",class_results
    elif np.sum(res_inhibit) < in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold:
        return res, "excitation",class_results
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_excite == 1)[0][-1] >= np.where(res_inhibit == 1)[0][-1]:
        return res, "biphasic IE",class_results
    elif np.sum(res_inhibit) >= in_bin_threshold and np.sum(res_excite) >= ex_bin_threshold and np.where(res_inhibit == 1)[0][-1] >= np.where(res_excite == 1)[0][-1]:
        return res, "biphasic EI",class_results
    else:
        return res, "no effect",class_results # Return the number of bins no effect, inhbited, and excited, the classification, and the corresponding bin class integers
    '''

def local_inhibition_check(neuron):
    in_bin_threshold = 2;
    bl_bins = int(neuron.baseline_length/neuron.bin_width)
    total_poss = int(neuron.trials*(bl_bins-(in_bin_threshold-1)))
    trial_success = 0;
    new_trial_success = False;
    success = 0;

    for trial in range(1,neuron.trials+1):#neuron.trials+1):
        stimFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb")
        stim = pickle.load(stimFile)
        stimFile.close()

        new_trial_success = False;
        for bin in range(1,bl_bins+1-(in_bin_threshold-1)):
            files = [open(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/bin_{bin+n}/baseline_spike_train_bin_{bin+n:02d}.obj","rb") for n in range(0,in_bin_threshold)]
            freqs = np.asarray([pickle.load(files[n]).freq for n in range(0,in_bin_threshold)])
            for f in range(len(files)):
                files[f].close()
            if np.sum(freqs) == 0 and not new_trial_success:
                trial_success += 1;
                success += 1
                new_trial_success = True;
            elif np.sum(freqs) == 0:
                success += 1
    pmf_vals = np.asarray([binom.pmf(k,neuron.trials,success/total_poss) for k in range(neuron.trials+1)])

    #print(np.where(pmf_vals < 0.05))

    max_pmf = np.where(pmf_vals == pmf_vals.max())[0][0]
    #print(max_pmf)
    bl_trial_prob = np.where(pmf_vals[max_pmf:] < 0.05)[0][0]+max_pmf

    trial_bins = int(neuron.trial_length/neuron.bin_width)
    success_trials = np.zeros(trial_bins - (in_bin_threshold -1))
    bin_sets = np.asarray([list(range(i,i+in_bin_threshold)) for i in range(1,trial_bins)])
    for bs in range(len(bin_sets)):
        for trial in range(1,neuron.trials):
            file1 = open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/bin_{bin_sets[bs][0]}/stimulus_spike_train_bin_{bin_sets[bs][0]:02d}.obj","rb")
            b1 = pickle.load(file1).freq
            file1.close()

            file2 = open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/bin_{bin_sets[bs][1]}/stimulus_spike_train_bin_{bin_sets[bs][1]:02d}.obj","rb")
            b2 = pickle.load(file2).freq
            file2.close()
            if b1+b2 == 0:
                success_trials[bs] += 1

    '''
    print(bl_trial_prob)
    print(success_trials)
    print(success_trials.max())
    print(success,total_poss, success/total_poss,trial_success)
    print(pmf_vals)
    '''


    if np.max([bl_trial_prob,4]) < success_trials.max():
        '''
        plt.plot(range(neuron.trials+1),pmf_vals,marker="o")
        plt.hlines(0.05,0,neuron.trials,color="k",linestyle="dashed")
        plt.xlabel("Trials"); plt.ylabel("PMF");
        plt.show()
        '''
        return True, bin_sets[np.where(success_trials == success_trials.max())[0][0]]
    return False, -1

def trial_sdf_isi_functions_classification(baseline, stim_data,trial_direc,bin_width=0.5,percentile=95,in_bin_threshold=3,ex_bin_threshold=3):
    '''
    Function that determines how the stimulus responds as compared to the baseline with the light on. Based on comparing ISI peak values for an ISI function.
        arr baseline -> array of spike times for the baseline
        arr stimulus -> array of spike times for the stimulus
        str trial_direc -> string that is the path of the trial directory to store and grab all data for the current trial
        arr bin_edges -> array that defines the edges of each bin for the stimulus, left side inclusive, right side exclusive, default integers 0 to 10
        float alpha -> alpha to check if p-value is less than for significance (Bonferroni corrected) for Mann Whitney U test
        int shuffles -> number of times to shuffle baseline values to recreate a new ISI function
        int percentile -> integer defining what the percentile should be to check the classification for significance
    '''
    if len(stim_data.spikes) <= 5: # If the stimulus has 5 or less spikes, classify as complete inhibition
        return np.array([0,len(stim_data.bin_edges)-1,0]),"complete inhibition", np.ones(len(stim_data.bin_edges)-1) # Return all bins as inhibited, classification as complete inhibition, and array of bin classes as inhibitied
    # Call the function that checks for excited bins, returns array of 0 (not excited) or 1 (excited) corresponding to bin classifications
    res_excite = trial_excitation_check(baseline,stim_data,trial_direc,)
    # Call the function that checks for inhibited bins, returns array of 0 (not inhibited) or 1 (inhibited) corresponding to bin classifications
    res_inhibit = trial_inhibition_check(baseline,stim_data,trial_direc,)
    np.savetxt(trial_direc+"/results_excitation.txt",res_excite,delimiter="\t",newline="\n",fmt="%f",header="Stimulus bin results for excitation check. 1 means bin excited.") # Save the results of excitation check
    np.savetxt(trial_direc+"/results_inhibition.txt",res_inhibit,delimiter="\t",newline="\n",fmt="%f",header="Stimulus bin results for inhibition check. 1 means bin inhibited.") # Save the results of inhibition check
    return classify_response(res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,stim_data.bin_edges)

def average_sdf_isi_functions_classification(neuron,bin_width=0.5,percentile=90,in_bin_threshold=3,ex_bin_threshold=3):
    baselines = []; stimuli = []; trial_results = []
    for trial in range(1,neuron.trials+1):
        baselineFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb")
        baselines.append(pickle.load(baselineFile))
        baselineFile.close()

        stimuliFile = open(f"{neuron.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")
        stimuli.append(pickle.load(stimuliFile))
        stimuliFile.close()

        trial_results.append(np.loadtxt(f"{neuron.cell_dir}/trial_{trial:02d}/bin_results.txt"))
    cmap = ["gray","cyan","red","green"] # Color map to code for classification of bin
    fig,axe = plt.subplots(1,1,figsize=(12,6),dpi=300)
    for trial in range(neuron.trials):
        axe.scatter(baselines[trial].spikes-baselines[trial].time,np.ones(len(baselines[trial].spikes))*(trial+1),marker="|",color="k",s=200)
        axe.scatter(stimuli[trial].spikes,np.ones(len(stimuli[trial].spikes))*(trial+1),marker="|",color="blue",s=200)
        for i in range(len(stimuli[trial].bin_edges)-1):
            #axe.scatter(np.mean([stimuli[trial].bin_edges[i],stimuli[trial].bin_edges[i+1]]),trial+0.5,color=cmap[int(trial_results[trial][i])],marker='x') # Plot a colored X corresponding to the type of bin classification
            axe.add_patch(Rectangle(xy=(np.mean([stimuli[trial].bin_edges[i],stimuli[trial].bin_edges[i+1]])-bin_width/4,trial+0.5-bin_width/4),width=bin_width/2,height=bin_width/2,color=cmap[int(trial_results[trial][i])],fill=True))
    axe.hlines(neuron.trials+0.5,0,stimuli[0].time,color='k',linewidth=5,label="Light")
    axe.vlines(stimuli[0].bin_edges,0,neuron.trials+1,color="k",linestyle="dashed",alpha=0.25,linewidth=0.5) # Draw vertical lines for bin edges on the SDF stimulus plot
    axe.vlines(baselines[0].bin_edges-baselines[0].time,0,neuron.trials+1,color="k",linestyle="dashed",alpha=0.25,linewidth=0.5) # Draw vertical lines for bin edges on the SDF baselien plot
    axe.set_ylim([0,neuron.trials+0.75])
    axe.set_xlim([-neuron.baseline_length,neuron.trial_length])
    axe.set_yticks([k for k in range(1,neuron.trials+1)])
    axe.spines['right'].set_visible(False)
    axe.spines['top'].set_visible(False)
    plt.suptitle("All Baseline (Blue) and Trials (Orange)")
    fig.text(0.06, 0.5, 'Trials', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 't', ha='center', va='center')
    makeNice(axe)
    plt.savefig(f"{neuron.cell_dir}/all_trial_spike_trains.pdf") # Save the figure in trial direc
    plt.close() # Close the figure

    if (np.mean([len(stimulus.spikes) for stimulus in stimuli]) <= 5) and (np.mean([len(baseline.spikes) for baseline in baselines]) <= 5):
        return np.array([len(stimuli[0].bin_edges)-1,0,0]),"no effect", np.zeros(len(stimuli[0].bin_edges)-1)
    if (np.mean([len(stimulus.spikes) for stimulus in stimuli]) <= 5):
        return np.array([0,len(stimuli[0].bin_edges)-1,0]),"complete inhibition", np.ones(len(stimuli[0].bin_edges)-1)


    '''
    if (np.mean([stimulus.freq for stimulus in stimuli]) <= 0.5) and (np.mean([baseline.freq for baseline in baselines]) <= 0.5):
        return np.array([len(stimuli[0].bin_edges)-1,0,0]),"no effect", np.zeros(len(stimuli[0].bin_edges)-1)
    if (np.mean([stimulus.freq for stimulus in stimuli]) <= 0.5):
        return np.array([0,len(stimuli[0].bin_edges)-1,0]),"complete inhibition", np.ones(len(stimuli[0].bin_edges)-1)
    '''

    res_excite = avg_excitation_check(neuron,baselines,stimuli,percentile=percentile,bin_width=bin_width)
    res_inhibit = avg_inhibition_check(neuron,baselines,stimuli,percentile=percentile,bin_width=bin_width)
    np.savetxt(neuron.cell_dir+"/avg_results_excitation.txt",res_excite,delimiter="\t",newline="\n",fmt="%f",header="Average Stimulus bin results for excitation check. 1 means bin excited.") # Save the results of excitation check
    np.savetxt(neuron.cell_dir+"/avg_results_inhibition.txt",res_inhibit,delimiter="\t",newline="\n",fmt="%f",header="Average Stimulus bin results for inhibition check. 1 means bin inhibited.") # Save the results of inhibition check
    res, type, class_results = avg_classify_response(neuron, res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,stimuli[0].bin_edges)
    '''
    res, type, class_results = classify_response(res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,stimuli[0].bin_edges)
    if type != "partial inhibition" and type != "adapting inhibition" and type != "biphasic IE" and type != "biphasic EI":
        if local_inhibition_check(neuron,in_bin_threshold=in_bin_threshold):
            type = "local inhibition"
    '''
    #return  classify_response(res_inhibit,res_excite,in_bin_threshold,ex_bin_threshold,stimuli[0].bin_edges)
    return res, type, class_results

def avg_inhibition_check(neuron,baselines,stimuli,percentile=95,bin_width=0.5):
    fig,ax = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create figure that will have 2 rows and 1 column
    results = np.zeros(len(stimuli[0].bin_edges)-1) # Create empty array that will store the classifications of each bin

    isi_bls = np.asarray([baseline.isif for baseline in baselines if baseline.spikes.size > 1])
    isi_stims = np.asarray([stimulus.isif for stimulus in stimuli if stimulus.spikes.size > 1])
    avg_isi_stim = np.mean(isi_stims,axis=0)
    avg_isi_bl = np.mean(isi_bls,axis=0)

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

def trial_inhibition_check(baseline_train,stim_data_train,trial_direc,shuffles=10,percentile=95):
    fig,ax = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create figure that will have 2 rows and 1 column
    results = np.zeros(len(stim_data_train.bin_edges)-1) # Create empty array that will store the classifications of each bin
    isi_stim = stim_data_train.isif# Create SDF for baseline and stimulus spikes
    bl_isi = np.diff(baseline_train.spikes) # Find the baseline ISI values
    bl_shuffles = [np.random.permutation(bl_isi) if x > 0 else bl_isi for x in range(10)] # Shuffle the bl ISI values for shuffles number of times
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


def avg_excitation_check(neuron,baselines,stimuli,percentile=95,bin_width=0.5):
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
    makeNice([ax[0],ax[1]])
    plt.savefig(f"{neuron.cell_dir}/avg_sdf_shuffling.pdf") # Save the figure
    plt.close() # Close the figure
    return results # Return the array containing the values of the bins excited (1) or not (0)

def trial_excitation_check(baseline_train,stim_data_train,trial_direc,bin_edges=list(range(11)),shuffles=10,percentile=95,bin_width=0.5):
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
    makeNice([ax[0],ax[1]])
    plt.savefig(trial_direc+"/sdf_shuffling.pdf") # Save the figure
    plt.close() # Close the figure
    return results # Return the array containing the values of the bins excited (1) or not (0)
