################################################################################
# bin.py
# Object that stores typical information regarding a bin of a spike train.
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
        '''
        Initialize bin object for further analysis. Multiple bins make up a spike train.
            float left_bin_edge -> float defining the left edge of the bin, inclusive
            float right_bin_edge -> float defining the right edge of the bin, exclusive
            arr spike_train -> float array of spike times existing in the bin
            int position -> integer that defines where the bin is in the spike train
        '''
        self.left_bin_edge = left_bin_edge
        self.right_bin_edge = right_bin_edge
        self.spikes = spikes
        self.position = position
        self.bin_width = self.right_bin_edge - self.left_bin_edge
        self.cv = st.calc_cv(spikes)
        self.freq = st.freq(spikes,self.bin_width)
        self.save_direc = check_direc(save_direc)
        self.isi = np.mean(np.diff(self.spikes)) if len(self.spikes) > 1 else 0

    def set_name(self,name):
        '''
        Sets the name of the spike train. Used for saving object.
        '''
        self.name = name

    def save_meta_data(self):
        '''
        Function that grabs all meta data and save it to a text file.
        '''
        with open(f"{self.save_direc}/{self.name}_meta_data.txt","w") as f: # Open file meta_data.txt in cell_dir.
            f.write(f"# Meta Data for {self.name}\n")
            for key in self.__dict__:
                if key != "spikes":
                    f.write(f"{key}:\t{self.__dict__[key]}\n")

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

    def save_data(self):
        '''
        Saves self and all characteristics in save_direc.
        '''
        self.save_meta_data()
        self.save_spikes()
        fileHandler = open(f"{self.save_direc}/{self.name}.obj","wb")
        pickle.dump(self,fileHandler)
        fileHandler.close()
