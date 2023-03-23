################################################################################
# spike_train.py
# Object that stores typical information regarding a neural spike train.
# Author: John E. Parker
################################################################################

# import python modules
import pickle
import numpy as np
from matplotlib import pyplot as plt

# import user modules
from helpers import *
from bin import bin
import excitation_check as exch
import inhibition_check as inch

def calc_cv(spike_train):
    """
    For a given spike train, calculate the CV.

    :param spike_train: float array of spike times for analysis to determine the CV
    :return CV value of spike train
    """
    if len(spike_train) > 1: # If the spike train has more than 1 spike, find the CV.
        isi = np.diff(spike_train) # Find the ISIs for the entire spike train
        return np.std(isi)/np.mean(isi) # Return the CV
    else: # Return the CV as 0 if the spike train has less than 2 spikes
        return 0

def freq(spike_train,time):
    """
    For a given spike train, calculate the frequency (firing rate).
    
    :param spike_train: array of spike times for analysis to determine the frequency
    :param time: float defining the total time over which the spike train was recorded
    :return firing rate of spike train
    """
    return len(spike_train)/time # firing rate is number of spikes divided by total time recorded

class spike_train:
    def __init__(self,spikes,time,bin_width,type):
        """
        Initialize spike train object given array of spike times, and recording time.

        :param spikes: float array of spike times
        :param time: float defining the total time over which the spike train was recorded
        :param bin_width: bin_width to use for defining bin edges and bins
        :param type: type of spike train
        """
        self.type = type # Set attribute
        self.time = time # Set attribute
        self.bin_width = bin_width # Set attribute
        self.bin_edges = np.arange(0,self.time+self.bin_width,self.bin_width) # Create bins by bin_width
        self.spikes = spikes # Set spikes
        self.first_spike = self.spikes[0] if len(self.spikes) > 0 else self.time # Find the first spike
        self.last_spike = self.spikes[-1] if len(self.spikes) > 0 else self.time # Find the last spike
        self.cv = calc_cv(self.spikes) # Calc CV and set attribute
        self.freq = freq(self.spikes,self.time) # Calc firing rate and set attribute
        self.avg_isi = np.mean(np.diff(self.spikes)) if len(self.spikes) > 1 else 0 # Calc average ISI and set attribute
        self.t = np.linspace(0,self.time,int(self.time*1000)) # Create time array as attribute
        self.sdf = exch.kernel(self.spikes,self.t) if len(self.spikes) > 1 else np.zeros(int(self.time*1000)) # Generate SDF
        if len(self.spikes) < 2: # Empty ISIF if less than 2 spikes
            #self.isif = np.zeros(1)
            self.isif = np.zeros(self.t.shape[0]) # Empty array for ISIF
        else: # Generate ISIF if more than 1 spike
            self.isif = inch.isi_function(self.spikes,self.t)


    def get_bins(self):
        """
        Function that splits spike train into bins.
        """
        bins = [] # Empty array to store bin objects
        for i in range(len(self.bin_edges)-1): # Iterate through all bins
            # Generate a bin object from spikes in the bin
            curr_bin = bin(self.bin_edges[i],self.bin_edges[i+1],self.spikes[np.where((self.spikes >= self.bin_edges[i]) & (self.spikes < self.bin_edges[i+1]))],i+1,f"{self.save_direc}/bin_{i+1}")
            curr_bin.set_name(f"{self.name}_bin_{i+1:02d}") # Set the name of the bin object
            curr_bin.save_data() # Save the bin object data
            bins.append(curr_bin) # Append the bin object to the bins array
        self.bins = bins # Set attribute of all bins
        _, ax = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create a figure of the bin firing rate and CV
        ax[0].plot(self.bin_edges[:-1],[b.freq for b in self.bins],marker="o",color="blue") # Plot the bin firing rate
        ax[1].plot(self.bin_edges[:-1],[b.cv for b in self.bins],marker="o",color="blue") # Plot the bin CV
        ax[1].set_ylabel("Bin CV"); ax[0].set_ylabel("Bin Firing Rate (Hz)"); # Set the ylabels
        ax[1].set_xlabel("Time (s)") # Set the xlabel
        plt.suptitle(f"{self.name} Firing Rate (top) and CV (bottom)") # Set the title
        makeNice([ax[0],ax[1]]) # Clean the figure
        plt.savefig(f"{self.save_direc}/freq_cv.pdf") # Save the figure
        plt.close() # Close the figure

    def set_name(self,name):
        """
        Sets the name of the spike train. Used for saving object.

        :param name: Name of the spike train.
        """
        self.name = name # Set attribute

    def set_save_direc(self,direc):
        """
        Sets the destination to save spike train object and data if desired.

        :param direc: Directory for save location
        """
        self.save_direc = direc # Set attirbute of save location
        check_direc(direc) # Check/create save location.

    def save_spikes(self):
        """
        Function that saves spike data to save_direc as txt file.
        """
        # Save spike data to text file.
        np.savetxt(f"{self.save_direc}/{self.name}.txt",self.spikes,delimiter="\t",newline="\n",fmt="%f",header=f"{self.name} spike times.")

    def save_meta_data(self):
        """
        Function that grabs all meta data and save it to a text file.
        """
        with open(f"{self.save_direc}/{self.name}_meta_data.txt","w") as f: # Open file meta_data.txt in cell_dir.
            f.write(f"# Meta Data for {self.name}\n") # Write the first line
            for key in self.__dict__: # Save all meta data except large arrays
                if key != "spikes" and key != "sdf" and key != "isif" and key != "t" and key != "bins": # Only save metadata without these labels
                    f.write(f"{key}:\t{self.__dict__[key]}\n") # Write metadata to file

    def save_data(self):
        """
        Saves self and all characteristics in save_direc.
        """
        self.save_meta_data() # Save the meta data
        self.save_spikes() # Save the spikes
        np.savetxt(f"{self.save_direc}/sdf.txt",self.sdf,newline="\n",delimiter="\t",header="# SDF of Spike Train") # Save the SDF to text file
        np.savetxt(f"{self.save_direc}/isif.txt",self.isif,newline="\n",delimiter="\t",header="# ISIF of Spike Train") # Save the ISIF to text file
        if len(self.spikes) > 1: # If spikes are greater than 1, plot the SDF and ISIF
            self.plot_fcns()
        fileHandler = open(f"{self.save_direc}/{self.name}.obj","wb") # Open the file for spike train object
        pickle.dump(self,fileHandler) # Save spike train object
        fileHandler.close() # Close spike train object

    def plot_fcns(self):
        """
        Create figure of ISIF and SDF for the spike train.
        """
        _, axe = plt.subplots(2,1,figsize=(8,6),dpi=300) # Create two row figure
        axe[0].plot(self.t,self.sdf,color="blue") # Plot SDF in blue
        axe[1].plot(self.t,self.isif,color="blue") # Plot ISIF in blue
        axe[0].scatter(self.spikes,np.zeros(len(self.spikes)),marker="|",color="k") # Plot spikes as vertical hashes
        axe[1].scatter(self.spikes,np.zeros(len(self.spikes)),marker="|",color="k") # Plot spikes as vertical hashes
        axe[0].set_xlabel("Time (s)"); axe[0].set_ylabel("SDF [spikes/s]") # Label the SDF axes
        axe[1].set_xlabel("Time (s)"); axe[1].set_ylabel("ISI Function [s]") # Label the SDF axes
        axe[0].vlines(self.bin_edges,axe[0].get_ylim()[0],axe[0].get_ylim()[1],color="k",linestyle="dashed",alpha=0.5) # Draw vertical lines for bin edges on the SDF stimulus plot
        axe[1].vlines(self.bin_edges,axe[1].get_ylim()[0],axe[1].get_ylim()[1],color="k",linestyle="dashed",alpha=0.5) # Draw vertical lines for bin edges on the SDF stimulus plot
        plt.suptitle(f"SDF and ISIF for Spike Train {self.name}") # Set the title
        makeNice([axe[0],axe[1]]) # Clean the figure
        plt.savefig(f"{self.save_direc}/sdf_isif_fig.pdf") # Save the figure
        plt.close() # Close the figure
