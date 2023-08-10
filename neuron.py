################################################################################
# neuron.py
# Neuron class/object for storing all relevant data
# Author: John E. Parker
################################################################################

# import python modules
import numpy as np
from matplotlib import pyplot as plt
import pickle

# import user modules
import stimulus_classification as stimclass
from helpers import *
from spike_train import spike_train as st

# Class to describe neuron for further analysis
class neural_cell:
    def __init__(self,data_direc,group,src,average_shuffling,isif_sdf_threshold,mu,sigma):
        """
        Initialize neuron object for further analysis of spike trains. Arguments:
        
        :param data_direc: numeric value of number of neurons being analyzed
        :param group: string describing the type of neuron
        :param src: directory specifying where the data comes from
        :param average_shuffling: Boolean to determine if to use shuffling with average resposne
        :param isif_sdf_threshold: Value to determine cutoff with ISIF or SDF with inhibition
        :param mu: Number of points for ISIF for moving mean
        :param sigma: Bandwidth of SDF function
        """
        self.data_direc = data_direc # Set the neuron's data_direc to data_direc
        self.group = group # Set the neuron's group
        self.cell_num = int(src.split("_")[1]) # Store cell number as attribute
        self.src = f"{self.data_direc}/{src}" # Store src directory as attribute
        self.average_shuffling = average_shuffling # Store boolean for average shuffling
        self.isif_sdf_threshold = isif_sdf_threshold # Store value for trial Hz of determing isif or sdf for inhibition
        self.mu = mu # Store mu  for ISIF
        self.sigma = sigma # Store sigma for SDF


    def set_save_direc(self,save_direc):
        """
        Function to designate where the directory is to save the neuron object
        :param save_direc: path of save_direc directory to set as neuron attribute
        """
        self.cell_dir = check_direc(f"{save_direc}/Neuron_{self.cell_num:04d}") # Store save directory as neural_cell attribute

    def gather_data(self):
        """
        Function that looks at src directory and parses spike train data. Finds when the light is cut on, and what
        spikes correspond to each trial.
        """
        self.spikes = np.loadtxt(f"{self.src}/spikes.txt",ndmin=1) # Save the spike data
        self.light_on = np.loadtxt(f"{self.src}/light_on.txt",ndmin=1) # Save the spike data # Save the light on data
        self.trials = self.light_on.shape[0] # Set the number of trials as the number of times the light was cut on
        np.savetxt(f"{self.cell_dir}/spikes.txt",self.spikes,delimiter="\t",newline="\n",fmt="%f",header="All spike data from neuron.") # Save all the spike data
        np.savetxt(f"{self.cell_dir}/light_on.txt",self.light_on,delimiter="\t",newline="\n",fmt="%f",header="All times that light stimulus begins.") # Save all the times the light was cut on
        self.meta_data = {} # Create emtpy dictionary for metadata
        self.read_in_meta_data() # Read in metadata from source


    def read_in_meta_data(self):
        """
        Function that reads in meta data of neuron.
        """
        # Open meta_data.txt file
        for line in open(f"{self.data_direc}/Neuron_{self.cell_num:04d}/meta_data.txt").readlines():
            if line[0] != "#": # Skip lines with #
                splits = line.split(":") # Split data by column
                self.meta_data[splits[0]] = eval(splits[1]) if splits[0] == 'cell_num' or splits[0] =='distance' else splits[1][1:-1] # Store evaluated data

    def save_meta_data(self):
        """
        Function that grabs all meta data and save it to a text file in cell_dir.
        """
        # Iterate thorugh new variables and add to meta_data dictionary
        for key in self.__dict__: 
            if key != "spikes" and key != "light_on" and key != "meta_data": # Omit certain variables
                self.meta_data[key] = self.__dict__[key] # Store new variables
        with open(f"{self.cell_dir}/meta_data.txt","w") as f: # Open file meta_data.txt in cell_dir.
            f.write("# Meta Data for Neuron\n") # Write first line
            for key in sorted(self.meta_data): # Iterate through and write each meta_data variable to file
                    f.write(f"{key}:\t{self.meta_data[key]}\n") 

    def save_data(self):
        """
        Saves self and all attributes in save_direc.
        """
        self.save_meta_data() # Save the meta_data
        filehandler = open(f"{self.cell_dir}/neuron.obj","wb") # Open a new file for the object
        pickle.dump(self,filehandler) # Dump the object into the new file
        filehandler.close() # Close the file for the new object

    def plot_trial(self,trial,trial_direc,baseline_data,stim_data,bin_results,classification):
        """
        Function that plots the baseline and trial spike data as raster plot, and CV and firing rate.

        :param trial: integer specifying the current trial number
        :param trial_direc: path of trial_direc to store figure
        :param baseline_data: array of baseline spikes
        :param stim_data: array of stimulated data spikes
        :param bin_results: array detailing how each bin was classified
        :param classification: str of the classification for the current trial
        """
        _, axes = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create a two row figure
        ax = [axes[0],axes[1]] # Create list of axes
        # Plot the firing rate for each bin of baseline.
        ax[0].plot((np.arange(1,len(baseline_data.bins)+1,1)-len(baseline_data.bins))*baseline_data.bin_width,[b.freq for b in baseline_data.bins],marker='o',color="k")
        # Plot the firing rate for eaach bin of stimulus.
        ax[0].plot(np.arange(1,len(stim_data.bins)+1,1)*stim_data.bin_width,[b.freq for b in stim_data.bins],marker='o',color="blue")
        # Plot the CV for each bin of baseline.
        ax[1].plot((np.arange(1,len(baseline_data.bins)+1,1)-len(baseline_data.bins))*baseline_data.bin_width,[b.cv for b in baseline_data.bins],marker='o',color="k")
        # Plot the CV for each bin of stimulus.
        ax[1].plot(np.arange(1,len(stim_data.bins)+1,1)*stim_data.bin_width,[b.cv for b in stim_data.bins],marker='o',color="blue")
        ax[1].set_ylabel("CV"); ax[0].set_ylabel("Firing Rate"); # Set ylabels as firing rate and CV.
        ax[1].set_xlabel("Bins") # Set xlabel as bins.
        plt.suptitle(f"Trial {trial}, {classification}, FR (top) and CV (bottom)") # Set a title.
        makeNice(ax) # Clean the axes.
        plt.savefig(f"{trial_direc}/freq_cv.pdf") # Save the figure
        plt.close() # Close the figure.

        _, ax = plt.subplots(1,1,figsize=(12,4),dpi=300) # Create the figure for raster plots.
        ax.scatter(baseline_data.spikes-baseline_data.time,np.ones(len(baseline_data.spikes)),marker="|",color="k",label="Baseline") # Plot all the baseline data
        ax.scatter(stim_data.spikes,np.ones(len(stim_data.spikes)),marker="|",color="blue",label="Light On") # Plot all the stimulus spike data 
        cmap = ["gray","cyan","red","green"] # Color map to code for classification of bin (NE, IN, EX, BOTH)
        for i in range(len(bin_results)): # Iterate through each result of the bin classification
            ax.scatter(np.mean([stim_data.bin_edges[i],stim_data.bin_edges[i+1]]),0.95,color=cmap[int(bin_results[i])],marker='x') # Plot a colored X corresponding to the type of bin classification
        ax.hlines(1.05,0,stim_data.time,color='k',linewidth=5,label="Light") # Create a line to respresent the stimulus.
        ax.vlines(stim_data.bin_edges,0.9,1.1,color='k',linestyle='dashed',alpha=0.5,linewidth=0.5) # Create a vertical dashed line on the bin edges
        ax.vlines(baseline_data.bin_edges-baseline_data.time,0.9,1.1,color='k',linestyle='dashed',alpha=0.5,linewidth=0.5) # Create a vertical dashed line on the bin edges
        ax.set_xlabel("Time (s)"); ax.set_ylim([0.9,1.1]); ax.set_yticks([]); plt.legend() # Specify x labels, and y limits and tick marks
        plt.title(f"Trial {trial}, {classification}, Bl Freq: {baseline_data.freq:.2f}, Stim Freq: {stim_data.freq:.2f}") # Write the title, include ISI information.
        makeNice(ax) # Clean the axes
        plt.savefig(f"{trial_direc}/raster.pdf") # Save the figure to the direcotry corresponding to the neurons trial.
        plt.close() # Close the figure.



    def gather_trials(self,baseline_start,baseline_length,trial_start,trial_length,bin_width,trial_percentile,average_percentile,in_bin_threshold,ex_bin_threshold,con_in_bin_threshold,con_ex_bin_threshold):
        """
        Iterate through each trial and analyze classification. Save results and store appropriate data.

        :param baseline_start: Number representing time before stimulus on-set to begin the baseline
        :param baseline_length: Length of baseline.
        :param trial_start: Number representing time from stimulus on-set to begin the trial period
        :param trial_length: Length of trial.
        :param bin_width: Width of the bins.
        :param trial_percentile: Between 0 and 100 indicating the trial percentile for inhibition and excitation identification.
        :param average_percentile: Between 0 and 100 indicating the average percentile for inhibition and excitation identification.
        :param in_bin_threshold: Integer of bins that consider a trial inhibited.
        :param ex_bin_threshold: Integer of bins that consider a trial excited.
        :param con_in_bin_threshold: Integer of consecutive bins that consider a trial inhibited.
        :param con_ex_bin_threshold: Integer of consecutive bins that consider a trial excited.
        """
        np.random.seed(44)
        self.bin_width = bin_width # Set the attribute of bin_width
        self.classifications = [] # Set an empty array to hold trial classifications
        self.classification_vals = [] # Set an empty array to hold numerical trial classifications
        self.baseline_length = baseline_length; # Set the attribute of baseline_length
        self.baseline_start = baseline_start # Set the attribute of baseline_start
        self.trial_length = trial_length; # Set the attribute of trial_length
        self.trial_start = trial_start; # Set the attribute of trial_start
        self.trial_percentile = trial_percentile; self.average_percentile = average_percentile; # Set the attributes for trial and average percentiles
        self.in_bin_threshold = in_bin_threshold; self.ex_bin_threshold = ex_bin_threshold; # Set the IBT and EBT attributes
        self.con_in_bin_threshold = con_in_bin_threshold; self.con_ex_bin_threshold = con_ex_bin_threshold; # Set the IBT and EBT attributes
        self.trial_excitation_areas = [] # List of all areas found by SDF in the baseline
        self.trial_inhibition_areas = [] # List of all areas found by ISIF (or SDF) in the baseline

        baseline_stats = np.zeros((self.trials,5)) # Empty array for storing baseline trial stats of freq, cv, isis, first spike, last spike
        stim_stats = np.zeros((self.trials,5)) # Empty array for storing baseline trial stats of freq, cv, isis, first spike, last spike
        
        baseline_trials = [] # array to store baseline_trial spike trains
        stim_trials = [] # array to store stim_trial spike trains
        trial_direcs = [] # array to store trial_directories

        self.class_dict = {"complete inhibition":0, "adapting inhibition": 1, "partial inhibition":2,"no effect":3,"excitation":4,"biphasic IE":5,"biphasic EI": 6} # Set the classification dictionary
        for tl in range(1,self.trials+1): # Iterate through all trials and gather trial data
            trial_direc = check_direc(f"{self.cell_dir}/trial_{tl:02d}") # Create a trial directory
            trial_direcs.append(trial_direc) # append trial directory to array

            # Find the stimulus spikes and make a spike train object
            stim_data = st(self.spikes[(self.spikes >= self.light_on[tl-1]+self.trial_start) & (self.spikes < self.light_on[tl-1]+self.trial_start+self.trial_length)]-(self.light_on[tl-1]+self.trial_start),self.trial_length,self.bin_width,"stimulus",self.mu,self.sigma)
            # Find the baselin spikes and make a spike train object
            baseline_data = st(self.spikes[(self.spikes >= self.light_on[tl-1]-self.baseline_start) & (self.spikes < self.light_on[tl-1]-self.baseline_start+self.baseline_length)]-(self.light_on[tl-1]-self.baseline_start),self.baseline_length,self.bin_width,"baseline",self.mu,self.sigma)

            spike_trains = {"stimulus":stim_data,"baseline":baseline_data} # Create a dictionary of stimulus and baseline data
            for key in spike_trains: # Iterate through dictionary and run methods
                spike_trains[key].set_name(f"{key}_spike_train") # Set the name
                spike_trains[key].set_save_direc(f"{trial_direc}/{key}_data") # Create a location to save the data
                spike_trains[key].get_bins() # Generate bin analysis
                spike_trains[key].save_data() # Save the spike train data

            baseline_stats[tl-1] = np.asarray([baseline_data.freq,baseline_data.cv, baseline_data.avg_isi,baseline_data.first_spike,baseline_data.last_spike]) # Fill the baseline trial data array
            stim_stats[tl-1] = np.asarray([stim_data.freq,stim_data.cv, stim_data.avg_isi,stim_data.first_spike,stim_data.last_spike]) # Fill the stimulus trial data array

            baseline_trials.append(baseline_data) # Append baseline to all baseline data
            stim_trials.append(stim_data) # Append baseline to all baseline data

        self.baseline_freq = np.mean(baseline_stats[:,0]) # Determine the trial average baseline firing rate
        self.baseline_cv = np.mean(baseline_stats[:,1]) # Determine the trial average baseline CV
        self.baseline_avg_isi = np.mean(baseline_stats[:,2]) # Determine the trial baseline average ISI
        self.baseline_first_spike = np.mean(baseline_stats[:,3]) # Determine the trial average first spike time
        self.baseline_last_spike = np.mean(baseline_stats[:,4]) # Determine the trial average last spike time

        self.stim_freq = np.mean(stim_stats[:,0]) # Determine the trial average stimulus firing rate
        self.stim_cv = np.mean(stim_stats[:,1]) # Determine the trial average stimulus CV
        self.stim_avg_isi = np.mean(stim_stats[:,2]) # Determine the trial average stimulus average ISI
        self.stim_first_spike = np.mean(stim_stats[:,3]) # Determine the trial average stimulus first spike time
        self.stim_last_spike = np.mean(stim_stats[:,4]) # Determine the trial average stimulus last spike time

        # Set isif_inhibition depending on if the average firing rate in the baseline is above isif_sdf_threshold
        self.isif_inhibition = False if np.min(baseline_stats[:,0] > self.isif_sdf_threshold) else True

        # Iterate throuh all trials
        for tl in range(1,self.trials+1):
            # Determine the classification of the trial
            _,classification, bin_classes = stimclass.trial_sdf_isi_functions_classification(self,baseline_trials[tl-1], stim_trials[tl-1],trial_direcs[tl-1],self.isif_inhibition,percentile=self.trial_percentile,in_bin_threshold=self.in_bin_threshold,ex_bin_threshold=self.ex_bin_threshold,con_in_bin_threshold=self.con_in_bin_threshold,con_ex_bin_threshold=self.con_ex_bin_threshold)
            self.classifications.append(classification) # Append the trial classification to the classificaiton attribute array
            np.savetxt(trial_direcs[tl-1]+"/bin_results.txt",np.asarray(bin_classes),delimiter="\t",newline="\n",fmt="%d",header="Class result for each bin.") # Save the classification of each bin as a file
            self.plot_trial(tl,trial_direc,baseline_trials[tl-1],stim_trials[tl-1],bin_classes,classification) # Call plot trial to visualize the classification by bin
            self.classification_vals.append(self.class_dict[classification]) # Append the trial classification values to the classification value array
            
        self.classification_vals = np.asarray(self.classification_vals) # Change classifcation value array to numpy array
        self.average_classification_val = np.mean(self.classification_vals) # Find the average classification value and set attribute
        self.avg_bin_results = np.loadtxt(f"{self.cell_dir}/trial_01/bin_results.txt") / self.trials # Find the average bin identification result (first bin contribution)
        for trial in range(2,self.trials+1): # Iterate through each bin to find the average
            self.avg_bin_results += np.loadtxt(f"{self.cell_dir}/trial_{trial:02d}/bin_results.txt")/self.trials

    def gather_pre_post(self):
        """
        Determine properties of spikes before trials start and spikes after trials end.
        """
        # Generate spike train of data before first trial
        pre_exp = st(self.spikes[np.where((self.spikes < self.light_on[0]))],self.light_on[0],self.bin_width,"pre_experiment_spikes",self.mu,self.sigma)
        pre_exp.set_name(f"pre_exp_spike_train") # Set the name
        pre_exp.set_save_direc(f"{self.cell_dir}/pre_exp_data") # Set the save directory
        pre_exp.get_bins() # Gather the bins 
        pre_exp.save_data() # Gather data of spike train and save it
        self.pre_exp_freq = pre_exp.freq # Set the firing rate of before trial as neural attribute
        self.pre_exp_cv = pre_exp.cv # Set the CV of before trial as neural attribute
        self.pre_exp_avg_isi = pre_exp.avg_isi # Set the avg ISI of before trial as neural attribute

        # Generate spike train of data after last trial
        post_exp = st(self.spikes[np.where(((self.spikes >= (self.light_on[-1]+self.trial_length)) & (self.spikes < (self.light_on[-1]+self.trial_length+self.light_on[0]))))]-(self.light_on[-1]+self.trial_length),self.light_on[0],self.bin_width,"post_experiment_spikes",self.mu,self.sigma)
        post_exp.set_name(f"post_exp_spike_train") # Set the name
        post_exp.set_save_direc(f"{self.cell_dir}/post_exp_data") # Set the save directory
        post_exp.get_bins() # Gather the bins
        post_exp.save_data() # Gather data of spike train and save it
        self.post_exp_freq = post_exp.freq # Set the firing rate before trial as neural attribute
        self.post_exp_cv = post_exp.cv # Set the CV before trial as neural attribute
        self.post_exp_avg_isi = post_exp.avg_isi # Set the avg ISI before trial as neural attribute

    def classify_neuron(self):
        """
        Identify the average response of the neuron.
        """
        # Run the average response classifciation algorithm.
        np.random.seed(44)
        _,classification, bin_classes = stimclass.average_sdf_isi_functions_classification(self,bin_width=self.bin_width,percentile=self.average_percentile,in_bin_threshold=self.in_bin_threshold,ex_bin_threshold=self.ex_bin_threshold,con_in_bin_threshold=self.con_in_bin_threshold,con_ex_bin_threshold=self.con_ex_bin_threshold)
        self.neural_response = classification # Set the average response as neural attribute
        np.savetxt(self.cell_dir+"/avg_bin_results.txt",np.asarray(bin_classes),delimiter="\t",newline="\n",fmt="%d",header="Avg Class result for each bin.") # Save the classification of each bin as a file
        self.neural_response_val = self.class_dict[classification] # Set the average response value as neural attribute
        self.plot_average() # Plot the average firing rate and CV
        
    def plot_average(self):
        """
        Plot the average firing rate and CV over the trials, with the standard deviations for each bin.
        """
        baselines = []; stimuli = []; # Empty arrays to store baseline and stimulus data
        for trial in range(1,self.trials+1): # Iterate through all trials
            baselineFile = open(f"{self.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb") # Pull the trial baseline data file
            baselines.append(pickle.load(baselineFile)) # Store the baseline data
            baselineFile.close() # Close the trai baseline data file

            stimuliFile = open(f"{self.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb") # Pull the trial stimulus data file
            stimuli.append(pickle.load(stimuliFile)) # Store the stimulus data
            stimuliFile.close() # Close the trial stimulus data file

        avg_bl = np.zeros((len(baselines[0].bins),2)) # Create empty array for trial average baseline (firing rate ,cv)
        avg_stim = np.zeros((len(stimuli[0].bins),2)) # Create empty array for trial average stimulus (firing rate, cv)

        for baseline in baselines: # Itreate through trial baseline data
            for bi in range(len(baseline.bins)): # For each bin store the firing rate and CV
                avg_bl[bi,0] += baseline.bins[bi].freq/len(baselines) # Add the average for baseline firing rate
                avg_bl[bi,1] += baseline.bins[bi].cv/len(baselines) # Add the average for stimulus firing rate
        for stimulus in stimuli: # Iterate through trial stimulus data
            for bi in range(len(stimulus.bins)): # For each bin store the firing rate and CV
                avg_stim[bi,0] += stimulus.bins[bi].freq/len(stimuli) # Add the average for stimulus firing rate
                avg_stim[bi,1] += stimulus.bins[bi].cv/len(stimuli) # Add the average for stimulus CV
 
        yerr_bl = np.zeros((len(baselines[0].bins),len(baselines),2)) # Create the baseline error array (firing rate, cv)
        for bi in range(len(baselines[0].bins)): # For each bin store the frequency and CV
            for bl in range(len(baselines)): # Iterate through all the baselines
                yerr_bl[bi,bl,0] = baselines[bl].bins[bi].freq # Add the bin frequeny to the array
                yerr_bl[bi,bl,1] = baselines[bl].bins[bi].cv # Add the bin CV to the array

        yerr_stim = np.zeros((len(stimuli[0].bins),len(stimuli),2)) # Create the stimulus error array (firing rate, cv)
        for bi in range(len(stimuli[0].bins)): # For each bin store the frequency and CV
            for bl in range(len(stimuli)): # Iterate through all the stimuli
                yerr_stim[bi,bl,0] = stimuli[bl].bins[bi].freq # add the bin frequency to the array
                yerr_stim[bi,bl,1] = stimuli[bl].bins[bi].cv # Add the bin CV to the array

        _,ax = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create a two row figure to plot bin average firing rate and CV with standard deviations
        ax[0].plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k") # Plot the bin average baseline firing rate
        ax[0].plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue") # Plot the bin average stimulus firing rate
        ax[1].plot(np.arange(1,len(avg_bl[:,1])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,1],marker="o",color="k") # Plot the bin average baseline CV
        ax[1].plot(np.arange(1,len(avg_stim[:,1])+1,1)*stimuli[0].bin_width,avg_stim[:,1],marker="o",color="blue") # Plot the bin average stimulus CV

        # Plot the baseline firing rate standard deviation for each bin as a shaded area 
        ax[0].fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
        # Plot the stimulus firing rate standard deviation for each bin as a shaded area 
        ax[0].fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")

        # Plot the baseline CV standard deviation for each bin as a shaded area 
        ax[1].fill_between(np.arange(1,len(avg_bl[:,1])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,1]+np.asarray([np.std(yerr_bl[bi,:,1]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,1]-np.asarray([np.std(yerr_bl[bi,:,1]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
        # Plot the stimulus CV standard deviation for each bin as a shaded area 
        ax[1].fill_between(np.arange(1,len(avg_stim[:,1])+1,1)*stimuli[0].bin_width,avg_stim[:,1]+np.asarray([np.std(yerr_stim[bi,:,1]) for bi in range(yerr_stim.shape[0])]),avg_stim[:,1]-np.asarray([np.std(yerr_stim[bi,:,1]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")

        ax[1].set_ylabel("CV"); ax[0].set_ylabel("Firing Rate"); # Label the y-axes
        ax[1].set_xlabel("Bins") # Label the x-axes
        
        plt.suptitle(f"Average Response:, {self.neural_response}, FR (top) and CV (bottom)") # Give the figure a label
        makeNice([ax[0],ax[1]]) # Clean the figure up
        plt.savefig(f"{self.cell_dir}/avg_freq_cv.pdf") # Save the figure
        plt.close() # Close the figure
