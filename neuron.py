################################################################################
# neuron.py
# Object that stores relevant data for neural recrodings.
# Author: John E. Parker
################################################################################

# import python modules
import numpy as np
from matplotlib import pyplot as plt
import datetime, pickle

# import user modules
import stimulus_classification as stimclass
from helpers import *
from spike_train import spike_train as st

# Class to describe neuron for further analysis
class neural_cell:
    #def __init__(self,cell_num,mouse_type,delivery,src):
    def __init__(self,data_direc,group,src):
        '''
        Initialize neuron object for further analysis of spike trains. Arguments:
            int cell_num -> numeric value of number of neurons being analyzed
            str mouse_type -> string describing mouse, 6-OHDA or Naive
            str delivery -> string describing delivery, PV-DIO or hsyn
            str src -> directory specifying where the data comes from
        '''
        self.data_direc = data_direc
        self.group = group
        self.cell_num = int(src.split("_")[1]) # Store cell number as neural_cell attribute
        self.src = f"{self.data_direc}/{src}" # Store src directory as neural_cell attribute

    '''
    def set_recording_attributes(self):
        str_to_parse = self.src.split("/")[-3:]
        first_split = str_to_parse[0].split("_")
        self.ag_number = first_split[0]
        self.mouse_number = first_split[1][first_split[1].index("#")+1:] if first_split[1][first_split[1].index("#")+1:][0] != 'N' else first_split[1][first_split[1].index("#")+1:]
        self.date = datetime.datetime.strptime(first_split[-1],'%m%d%y').strftime('%d-%b-%y')
        second_split = str_to_parse[1].split("_")
        self.trajectory = second_split[0][second_split[0].index("y")+1:]
        self.laser_strength = second_split[-7]
        self.laser_duration = second_split[-6][0:2]
        self.channel = str_to_parse[-1][:str_to_parse[-1].index(".")+1]
    '''

    def set_save_direc(self,save_direc):
        '''
        Function to designate where the directory is to save the neuron object
            str save_direc -> path of save_direc directory to set as neuron attribute
        '''
        self.cell_dir = check_direc(f"{save_direc}/Neuron_{self.cell_num:04d}") # Store save directory as neural_cell attribute

    def gather_data(self):
        '''
        Function that looks at src directory and parses spike train data. Finds when the light is cut on, and what
        spikes correspond to each trial.
        '''
        '''
        with open(self.src) as f: # Open the src file
            lines = f.readlines() # Read in all the lines in the src file
            blanks = [line for line in range(len(lines)) if lines[line] == "\n"] # Find where all the blank lines are
            spikes = np.asarray([eval(spike) for spike in lines[blanks[0]+1:blanks[1]]]) # Gather all the spike data, data until second empty line
            light_on = None
            if blanks[2]-blanks[1] == 2: # Gather the light on data, data between second and third empty line, case if multiple empty lines in a row
                light_on = np.asarray([eval(light) for light in lines[blanks[2]+1:blanks[3]]]) # Store times that stimulation begain into light_on
            else: # Gather the light on data, data between second and third empty line
                light_on = np.asarray([eval(light) for light in lines[blanks[3]+1:blanks[4]]]) # Store times that stimulation begain into light_on
        self.trials = len(light_on) # Set the number of trials as the number of times the light was cut on
        self.spikes = spikes # Save the spike data
        self.light_on = light_on # Save the light on data
        '''

        
        self.spikes = np.loadtxt(f"{self.src}/spikes.txt") # Save the spike data
        self.light_on = np.loadtxt(f"{self.src}/light_on.txt") # Save the spike data # Save the light on data
        self.trials = self.light_on.shape[0] # Set the number of trials as the number of times the light was cut on
        np.savetxt(f"{self.cell_dir}/spikes.txt",self.spikes,delimiter="\t",newline="\n",fmt="%f",header="All spike data from neuron.") # Save all the spike data
        np.savetxt(f"{self.cell_dir}/light_on.txt",self.light_on,delimiter="\t",newline="\n",fmt="%f",header="All times that light stimulus begins.") # Save all the times the light was cut on

    def set_location(self):
        '''
        Function that determines location (left/right) in SNr and distance (mm). Parses src directory.
        '''
        parse = self.src.split('/') # Split the src directory by '/'
        if "right" in parse[8]: # Set the orientation to right if src has right in it
            self.orientation = "right"
        else: # Set the orientation to left if src has left in it
            self.orientation = "left"
        self.distance = eval(parse[8].split("_")[1][:-2]) # Determine what the mm distance by further splitting the src direcotry by "_"

    def save_data(self):
        '''
        Saves self and all characteristics in save_direc.
        '''
        self.save_meta_data()
        filehandler = open(f"{self.cell_dir}/neuron.obj","wb")
        pickle.dump(self,filehandler)
        filehandler.close()

    def save_meta_data(self):
        '''
        Function that grabs all meta data and save it to a text file in cell_dir.
        '''
        with open(f"{self.cell_dir}/meta_data.txt","w") as f: # Open file meta_data.txt in cell_dir.
            f.write("# Meta Data for neuron\n")
            for key in sorted(self.__dict__):
                if key != "spikes" and key != "light_on":
                    f.write(f"{key}:\t{self.__dict__[key]}\n") # Write the cell number

    def plot_trial(self,trial,trial_direc,baseline_data,stim_data,bin_results,classification):
        '''
        Function that plots the baseline and trial spike data as raster plot. Designates specific bin classification,
        trial classification, and ISI information.
            int trial -> integer specifying the current trial number
            str trial_direc -> path of trial_direc to store figure
            arr full_baseline -> array of baseline spikes
            arr stim_Data -> array of stimulated data spikes
            arr class_results -> array detailing how each bin was classified
            str class10s -> str of the classification for the current trial
        '''
        fig, axes = plt.subplots(2,1,figsize=(8,6),dpi=300)
        ax = [axes[0],axes[1]]
        ax[0].plot((np.arange(1,len(baseline_data.bins)+1,1)-len(baseline_data.bins))*baseline_data.bin_width,[b.freq for b in baseline_data.bins],marker='o',color="k")
        ax[0].plot(np.arange(1,len(stim_data.bins)+1,1)*stim_data.bin_width,[b.freq for b in stim_data.bins],marker='o',color="blue")
        ax[1].plot((np.arange(1,len(baseline_data.bins)+1,1)-len(baseline_data.bins))*baseline_data.bin_width,[b.cv for b in baseline_data.bins],marker='o',color="k")
        ax[1].plot(np.arange(1,len(stim_data.bins)+1,1)*stim_data.bin_width,[b.cv for b in stim_data.bins],marker='o',color="blue")
        ax[1].set_ylabel("CV"); ax[0].set_ylabel("Firing Rate");
        ax[1].set_xlabel("Bins")
        plt.suptitle(f"Trial {trial}, {classification}, FR (top) and CV (bottom)") # Write the title, include ISI information.
        makeNice(ax)
        plt.savefig(f"{trial_direc}/freq_cv.pdf")
        plt.close()

        fig, ax = plt.subplots(1,1,figsize=(12,4),dpi=300) # Create the figure
        ax.scatter(baseline_data.spikes-baseline_data.time,np.ones(len(baseline_data.spikes)),marker="|",color="k",label="Baseline") # Plot all the baseline data (blue)
        ax.scatter(stim_data.spikes,np.ones(len(stim_data.spikes)),marker="|",color="blue",label="Light On") # Plot all the stimulus spike data (orange)
        cmap = ["gray","cyan","red","green"] # Color map to code for classification of bin
        for i in range(len(bin_results)): # Iterate through each result of the bin classification
            ax.scatter(np.mean([stim_data.bin_edges[i],stim_data.bin_edges[i+1]]),0.95,color=cmap[int(bin_results[i])],marker='x') # Plot a colored X corresponding to the type of bin classification
        ax.hlines(1.05,0,stim_data.time,color='k',linewidth=5,label="Light")
        ax.vlines(stim_data.bin_edges,0.9,1.1,color='k',linestyle='dashed',alpha=0.5,linewidth=0.5) # Create a vertical dashed line on the bin edges
        ax.vlines(baseline_data.bin_edges-baseline_data.time,0.9,1.1,color='k',linestyle='dashed',alpha=0.5,linewidth=0.5) # Create a vertical dashed line on the bin edges
        ax.set_xlabel("Time (s)"); ax.set_ylim([0.9,1.1]); ax.set_yticks([]); plt.legend() # Specify x labels, and y limits and tick marks
        plt.title(f"Trial {trial}, {classification}, Bl Freq: {baseline_data.freq:.2f}, Stim Freq: {stim_data.freq:.2f}") # Write the title, include ISI information.
        makeNice(ax)
        plt.savefig(f"{trial_direc}/raster.pdf") # Save the figure to the direcotry corresponding to the neurons trial.
        plt.close() # Close the figure.

    def plot_average(self):
        baselines = []; stimuli = []; 
        for trial in range(1,self.trials+1):
            baselineFile = open(f"{self.cell_dir}/trial_{trial:02d}/baseline_data/baseline_spike_train.obj","rb")
            baselines.append(pickle.load(baselineFile))
            baselineFile.close()

            stimuliFile = open(f"{self.cell_dir}/trial_{trial:02d}/stimulus_data/stimulus_spike_train.obj","rb")
            stimuli.append(pickle.load(stimuliFile))
            stimuliFile.close()


        avg_bl = np.zeros((len(baselines[0].bins),2))
        avg_stim = np.zeros((len(stimuli[0].bins),2))


        for baseline in baselines:
            freq = np.zeros(()); cv = []
            for bi in range(len(baseline.bins)):
                avg_bl[bi,0] += baseline.bins[bi].freq/len(baselines)
                avg_bl[bi,1] += baseline.bins[bi].cv/len(baselines)
        for stimulus in stimuli:
            for bi in range(len(stimulus.bins)):
                avg_stim[bi,0] += stimulus.bins[bi].freq/len(stimuli)
                avg_stim[bi,1] += stimulus.bins[bi].cv/len(stimuli)

        yerr_bl = np.zeros((len(baselines[0].bins),len(baselines),2))
        for bi in range(len(baselines[0].bins)):
            for bl in range(len(baselines)):
                yerr_bl[bi,bl,0] = baselines[bl].bins[bi].freq
                yerr_bl[bi,bl,1] = baselines[bl].bins[bi].cv

        yerr_stim = np.zeros((len(stimuli[0].bins),len(stimuli),2))
        for bi in range(len(stimuli[0].bins)):
            for bl in range(len(stimuli)):
                yerr_stim[bi,bl,0] = stimuli[bl].bins[bi].freq
                yerr_stim[bi,bl,1] = stimuli[bl].bins[bi].cv

        fig,ax = plt.subplots(2,1,figsize=(8,6),dpi=300)

        ax[0].plot(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0],marker="o",color="k")
        ax[0].plot(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0],marker="o",color="blue")
        ax[1].plot(np.arange(1,len(avg_bl[:,1])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,1],marker="o",color="k")
        ax[1].plot(np.arange(1,len(avg_stim[:,1])+1,1)*stimuli[0].bin_width,avg_stim[:,1],marker="o",color="blue")

        ax[0].fill_between(np.arange(1,len(avg_bl[:,0])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,0]+np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,0]-np.asarray([np.std(yerr_bl[bi,:,0]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
        ax[0].fill_between(np.arange(1,len(avg_stim[:,0])+1,1)*stimuli[0].bin_width,avg_stim[:,0]+np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]), avg_stim[:,0]-np.asarray([np.std(yerr_stim[bi,:,0]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")

        ax[1].fill_between(np.arange(1,len(avg_bl[:,1])+1,1)*baselines[0].bin_width-baselines[0].time,avg_bl[:,1]+np.asarray([np.std(yerr_bl[bi,:,1]) for bi in range(yerr_bl.shape[0])]),avg_bl[:,1]-np.asarray([np.std(yerr_bl[bi,:,1]) for bi in range(yerr_bl.shape[0])]),alpha=0.5,color="k")
        ax[1].fill_between(np.arange(1,len(avg_stim[:,1])+1,1)*stimuli[0].bin_width,avg_stim[:,1]+np.asarray([np.std(yerr_stim[bi,:,1]) for bi in range(yerr_stim.shape[0])]),avg_stim[:,1]-np.asarray([np.std(yerr_stim[bi,:,1]) for bi in range(yerr_stim.shape[0])]),alpha=0.5,color="blue")

        ax[1].set_ylabel("CV"); ax[0].set_ylabel("Firing Rate");
        ax[1].set_xlabel("Bins")
        
        plt.suptitle(f"Average Response:, {self.neural_response}, FR (top) and CV (bottom)") # Write the title, include ISI information.
        makeNice([ax[0],ax[1]])
        plt.savefig(f"{self.cell_dir}/avg_freq_cv.pdf")
        plt.close()


    def gather_trials(self,baseline_start,baseline_length,trial_start,trial_length,bin_width,trial_percentile,average_percentile,in_bin_threshold,ex_bin_threshold):
        '''
        Iterate through each trial and analyze classification. Save results and store appropriate data.
        '''
        self.bin_width = bin_width
        self.classifications = []
        self.classification_vals = []
        self.baseline_length = baseline_length;
        self.baseline_start = baseline_start

        self.trial_length = trial_length;
        self.trial_start = trial_start;

        self.trial_percentile = trial_percentile; self.average_percentile = average_percentile;
        self.in_bin_threshold = in_bin_threshold; self.ex_bin_threshold = ex_bin_threshold;

        baseline_stats = np.zeros((self.trials,5)) # trial by freq, cv, isis, first spike, last spike
        stim_stats = np.zeros((self.trials,5))

        self.class_dict = {"complete inhibition":0, "adapting inhibition": 1, "partial inhibition":2,"no effect":3,"excitation":4,"biphasic IE":5,"biphasic EI": 6}
        for tl in range(1,self.trials+1):
            trial_direc = check_direc(f"{self.cell_dir}/trial_{tl:02d}")

            trial_data = self.spikes[np.where((self.spikes >= self.light_on[tl-1]-self.baseline_start) & (self.spikes < self.light_on[tl-1]+self.trial_length))] - self.light_on[tl-1]

            stim_data = st(trial_data[np.where((trial_data >= self.trial_start) & (trial_data < self.trial_start+self.trial_length))]-self.trial_start,self.trial_length,self.bin_width,"stimulus") # Grab the stimulus spike data as the time 0-10 seconds
            baseline_data = st(trial_data[np.where((trial_data > -self.baseline_start) &  (trial_data < -(self.baseline_start-self.baseline_length)))]+self.baseline_start,self.baseline_length,self.bin_width,"baseline") # Grab the baseline spike data as all spikes before 0s

            spike_trains = {"stimulus":stim_data,"baseline":baseline_data}
            for key in spike_trains:
                spike_trains[key].set_name(f"{key}_spike_train")
                spike_trains[key].set_save_direc(f"{trial_direc}/{key}_data")
                spike_trains[key].get_bins()
                #spike_trains[key].get_bursts()
                spike_trains[key].save_data()

            baseline_stats[tl-1] = np.asarray([baseline_data.freq,baseline_data.cv, baseline_data.avg_isi,baseline_data.first_spike,baseline_data.last_spike]) 
            stim_stats[tl-1] = np.asarray([stim_data.freq,stim_data.cv, stim_data.avg_isi,stim_data.first_spike,stim_data.last_spike]) 

            bin_type_breakdown,classification, bin_classes = stimclass.trial_sdf_isi_functions_classification(baseline_data, stim_data,trial_direc,bin_width=self.bin_width,percentile=self.trial_percentile,in_bin_threshold=self.in_bin_threshold,ex_bin_threshold=self.ex_bin_threshold)
            self.classifications.append(classification)
            np.savetxt(trial_direc+"/bin_results.txt",np.asarray(bin_classes),delimiter="\t",newline="\n",fmt="%d",header="Class result for each bin.") # Save the classification of each bin as a file
            self.plot_trial(tl,trial_direc,baseline_data,stim_data,bin_classes,classification) # Call plot trial to visualize the classification by bin
            self.classification_vals.append(self.class_dict[classification])

        self.baseline_freq = np.mean(baseline_stats[:,0])
        self.baseline_cv = np.mean(baseline_stats[:,1])
        self.baseline_avg_isi = np.mean(baseline_stats[:,2])
        self.baseline_first_spike = np.mean(baseline_stats[:,3])
        self.baseline_last_spike = np.mean(baseline_stats[:,4])

        self.stim_freq = np.mean(stim_stats[:,0])
        self.stim_cv = np.mean(stim_stats[:,1])
        self.stim_avg_isi = np.mean(stim_stats[:,2])
        self.stim_first_spike = np.mean(stim_stats[:,3])
        self.stim_last_spike = np.mean(stim_stats[:,4])


        self.classification_vals = np.asarray(self.classification_vals)
        self.average_classification_val = np.mean(self.classification_vals)
        self.avg_bin_results = np.loadtxt(f"{self.cell_dir}/trial_01/bin_results.txt") / self.trials
        for trial in range(2,self.trials+1):
            self.avg_bin_results += np.loadtxt(f"{self.cell_dir}/trial_{trial:02d}/bin_results.txt")/self.trials

    def gather_pre_post(self):
        pre_exp = st(self.spikes[np.where((self.spikes < self.light_on[0]))],self.light_on[0],self.bin_width,"pre_experiment_spikes")
        pre_exp.set_name(f"pre_exp_spike_train")
        pre_exp.set_save_direc(f"{self.cell_dir}/pre_exp_data")
        pre_exp.get_bins()
        pre_exp.save_data()
        self.pre_exp_freq = pre_exp.freq
        self.pre_exp_cv = pre_exp.cv
        self.pre_exp_avg_isi = pre_exp.avg_isi

        post_exp = st(self.spikes[np.where(((self.spikes >= (self.light_on[-1]+self.trial_length)) & (self.spikes < (self.light_on[-1]+self.trial_length+self.light_on[0]))))]-(self.light_on[-1]+self.trial_length),self.light_on[0],self.bin_width,"post_experiment_spikes")
        post_exp.set_name(f"post_exp_spike_train")
        post_exp.set_save_direc(f"{self.cell_dir}/post_exp_data")
        post_exp.get_bins()
        post_exp.save_data()
        self.post_exp_freq = post_exp.freq
        self.post_exp_cv = post_exp.cv
        self.post_exp_avg_isi = post_exp.avg_isi


    def classify_neuron(self):
        bin_type_breakdown,classification, bin_classes = stimclass.average_sdf_isi_functions_classification(self,bin_width=self.bin_width,percentile=self.average_percentile,in_bin_threshold=self.in_bin_threshold,ex_bin_threshold=self.ex_bin_threshold)
        self.neural_response = classification
        np.savetxt(self.cell_dir+"/avg_bin_results.txt",np.asarray(bin_classes),delimiter="\t",newline="\n",fmt="%d",header="Avg Class result for each bin.") # Save the classification of each bin as a file
        self.neural_response_val = self.class_dict[classification]
        self.plot_average()

    def save_classification(self,direc,res10s, class10s):
        '''
        Saves the classification results for the current trials.
            str direc -> path to save the text file of the data
            arr res10s -> array contaning the number of no effect, inhibited, and excited bins
            str class10s -> str stating the classification for the trial
        '''
        with open(direc+"/results_classification_10s.txt","w") as f: # Open the directory and store data in results_classification_10s.txt file
            f.write("# Results - Bins no effect, Bins inhibited, Bins excited\n") # Header for the file
            f.write("Results:\t{0}\t{1}\t{2}\n".format(res10s[0],res10s[1],res10s[2])) # Save the number of bins as no effect, inhibited, and excited
            f.write("Classification:\t"+class10s) # Save the classification for the trial
