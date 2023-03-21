################################################################################
# spike_train.py
# Object that stores typical information regarding a neural spike train.
# Author: John E. Parker
################################################################################

# import python modules
import os, sys, pickle
import numpy as np
from matplotlib import pyplot as plt

# import user modules
from helpers import *
from bin import bin
import stimulus_classification as stimclass

def calc_cv(spike_train):
    '''
    For a given spike train, calculate the CV.
        arr spike_train -> float array of spike times for analysis to determine the CV
    '''
    if len(spike_train) > 1: # If the spike train has more than 1 spike, find the CV.
        isi = np.diff(spike_train) # Find the ISIs for the entire spike train
        return np.std(isi)/np.mean(isi) # Return the CV
    else: # Return the CV as 0 if the spike train has less than 2 spikes
        return 0

def freq(spike_train,time):
    '''
    For a given spike train, calculate the frequency.
        arr spike_train -> array of spike times for analysis to determine the frequency
        float time -> float defining the total time over which the spike train was recorded
    '''
    return len(spike_train)/time

class spike_train:
    def __init__(self,spikes,time,bin_width,type):
        '''
        Initialize spike train object given array of spike times, and recording time.
            arr spikes -> float array of spike times
            float time -> float defining the total time over which the spike train was recorded
        '''
        self.type = type
        self.time = time
        self.bin_width = bin_width
        self.bin_edges = np.arange(0,self.time+self.bin_width,self.bin_width)
        self.spikes = spikes
        self.first_spike = self.spikes[0] if len(self.spikes) > 0 else self.time
        self.last_spike = self.spikes[-1] if len(self.spikes) > 0 else self.time
        self.cv = calc_cv(self.spikes)
        self.freq = freq(self.spikes,self.time)
        self.avg_isi = np.mean(np.diff(self.spikes)) if len(self.spikes) > 1 else 0
        self.t = np.linspace(0,self.time,int(self.time*1000))
        self.sdf = stimclass.kernel(self.spikes,self.t) if len(self.spikes) > 1 else np.zeros(int(self.time*1000))
        if len(self.spikes) < 2:
            self.isif = np.zeros(1)
        else:
            self.isif = stimclass.isi_function(self.spikes,self.t)


    def get_bins(self):
        '''
        Function that splits spike train into bins.
            arr bin_edges -> array that defines the edges of each bin for the stimulus, left side inclusive, right side exclusive
        '''
        bins = []
        for i in range(len(self.bin_edges)-1):
            curr_bin = bin(self.bin_edges[i],self.bin_edges[i+1],self.spikes[np.where((self.spikes >= self.bin_edges[i]) & (self.spikes < self.bin_edges[i+1]))],i+1,f"{self.save_direc}/bin_{i+1}")
            curr_bin.set_name(f"{self.name}_bin_{i+1:02d}")
            curr_bin.save_data()
            bins.append(curr_bin)
        self.bins = bins
        _, ax = plt.subplots(2,1,figsize=(8,6),dpi=300)
        ax[0].plot(self.bin_edges[:-1],[b.freq for b in self.bins],marker="o",color="blue")
        ax[1].plot(self.bin_edges[:-1],[b.cv for b in self.bins],marker="o",color="blue")
        ax[1].set_ylabel("Bin CV"); ax[0].set_ylabel("Bin Firing Rate (Hz)");
        ax[1].set_xlabel("Time (s)")
        plt.suptitle(f"{self.name} Firing Rate (top) and CV (bottom)")
        makeNice([ax[0],ax[1]])
        plt.savefig(f"{self.save_direc}/freq_cv.pdf")
        plt.close()


    def set_name(self,name):
        '''
        Sets the name of the spike train. Used for saving object.
        '''
        self.name = name

    def set_save_direc(self,direc):
        '''
        Sets the destination to save spike train object and data if desired.
        '''
        self.save_direc = direc
        check_direc(direc)

    def save_spikes(self):
        '''
        Function that saves spike data to save_direc as txt file.
        '''
        np.savetxt(f"{self.save_direc}/{self.name}.txt",self.spikes,delimiter="\t",newline="\n",fmt="%f",header=f"{self.name} spike times.")

    def save_meta_data(self):
        '''
        Function that grabs all meta data and save it to a text file.
        '''
        with open(f"{self.save_direc}/{self.name}_meta_data.txt","w") as f: # Open file meta_data.txt in cell_dir.
            f.write(f"# Meta Data for {self.name}\n")
            for key in self.__dict__:
                if key != "spikes" and key != "sdf" and key != "isif" and key != "t" and key != "bins":
                    f.write(f"{key}:\t{self.__dict__[key]}\n")

    def plot_fcns(self):
        _, axe = plt.subplots(2,1,figsize=(8,6),dpi=300)
        axe[0].plot(self.t,self.sdf,color="blue")
        axe[1].plot(self.t,self.isif,color="blue")
        axe[0].scatter(self.spikes,np.zeros(len(self.spikes)),marker="|",color="k")
        axe[1].scatter(self.spikes,np.zeros(len(self.spikes)),marker="|",color="k")
        axe[0].set_xlabel("Time (s)"); axe[0].set_ylabel("SDF [spikes/s]") # Label the SDF axes
        axe[1].set_xlabel("Time (s)"); axe[1].set_ylabel("ISI Function [s]") # Label the SDF axes
        axe[0].vlines(self.bin_edges,axe[0].get_ylim()[0],axe[0].get_ylim()[1],color="k",linestyle="dashed",alpha=0.5) # Draw vertical lines for bin edges on the SDF stimulus plot
        axe[1].vlines(self.bin_edges,axe[1].get_ylim()[0],axe[1].get_ylim()[1],color="k",linestyle="dashed",alpha=0.5) # Draw vertical lines for bin edges on the SDF stimulus plot
        plt.suptitle(f"SDF and ISIF for Spike Train {self.name}")
        makeNice([axe[0],axe[1]])
        plt.savefig(f"{self.save_direc}/sdf_isif_fig.pdf")
        plt.close()


    def save_data(self):
        '''
        Saves self and all characteristics in save_direc.
        '''
        self.save_meta_data()
        self.save_spikes()
        np.savetxt(f"{self.save_direc}/sdf.txt",self.sdf,newline="\n",delimiter="\t",header="# SDF of Spike Train")
        np.savetxt(f"{self.save_direc}/isif.txt",self.isif,newline="\n",delimiter="\t",header="# ISIF of Spike Train")
        if len(self.spikes) > 1:
            self.plot_fcns()
        fileHandler = open(f"{self.save_direc}/{self.name}.obj","wb")
        pickle.dump(self,fileHandler)
        fileHandler.close()
