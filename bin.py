################################################################################
# bin.py
# Object that stores information regarding a bin of a spike train.
# Author: John E. Parker
################################################################################

# import python modules
import numpy as np
import pickle

# import user modules
from helpers import *
import spike_train as st

class bin:
    def __init__(self,left_bin_edge,right_bin_edge,spikes,position,save_direc):
        """
        Initialize bin object for further analysis. Multiple bins make up a spike train.
        :param left_bin_edge: float defining the left edge of the bin, inclusive
        :param right_bin_edge: float defining the right edge of the bin, exclusive
        :param spikes: float array of spike times existing in the bin
        :param position: integer that defines where the bin is in the spike train
        :param save_direc: directory to save bin
        """
        self.left_bin_edge = left_bin_edge # Set attribute
        self.right_bin_edge = right_bin_edge # Set attribute
        self.spikes = spikes # Set attribute
        self.position = position # Set attribute
        self.bin_width = self.right_bin_edge - self.left_bin_edge # Set the bin width
        self.cv = st.calc_cv(spikes) # Find the CV
        self.freq = st.freq(spikes,self.bin_width) # Find the firing rate 
        self.save_direc = check_direc(save_direc) # Set the save location
        self.isi = np.mean(np.diff(self.spikes)) if len(self.spikes) > 1 else 0 # Find the average isi

    def set_name(self,name):
        """
        Sets the name of the spike train. Used for saving object.

        :param name: String of the bin name
        """
        self.name = name # Set the attribute

    def save_data(self):
        """
        Saves self and all characteristics in save_direc.
        """
        self.save_meta_data() # Save meta data
        self.save_spikes() # Save the spikes
        fileHandler = open(f"{self.save_direc}/{self.name}.obj","wb") # Create file to save bin object
        pickle.dump(self,fileHandler) # Dump bin object data
        fileHandler.close() # Close file of bin object.

    def save_meta_data(self):
        """
        Function that grabs all meta data and save it to a text file.
        """
        with open(f"{self.save_direc}/{self.name}_meta_data.txt","w") as f: # Open file meta_data.txt in cell_dir.
            f.write(f"# Meta Data for {self.name}\n") # Write the first line
            for key in self.__dict__: # Iterate through all meta data and save, except spikes
                if key != "spikes": # If not spikes, save to file
                    f.write(f"{key}:\t{self.__dict__[key]}\n")

    def save_spikes(self):
        """
        Function that saves spike data to save_direc as txt file.
        """
        # Save spikes to the save directory and specify name of file.
        np.savetxt(f"{self.save_direc}/{self.name}.txt",self.spikes,delimiter="\t",newline="\n",fmt="%f",header=f"{self.name} spike times.")


