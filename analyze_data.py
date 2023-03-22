################################################################################
# analyze_data.py
# Runs classification scripts based on parameters to analyze neural spike trains
# Author: John E. Parker
################################################################################

# import python modules
import os, sys, pickle, argparse, multiprocessing
import pandas as pd

# import user made modules
import process_neural_data as pnd
from helpers import *

if __name__ == '__main__':
    # Parameters that can be used. Flags can be passed to script with proper entries.
    parser = argparse.ArgumentParser(description="Classification and data analysis of neural spiking.")
    parser.add_argument('-d','--data_direc',type=str,default="/Users/johnparker/streac/data",help="Directory where raw data is stored.",required=True)
    parser.add_argument('-r','--save_direc',type=str,default="/Users/johnparker/streac/results",help="Directory where processed data is stored.",required=True)

    parser.add_argument('-f','--param_file',type=str,default="",help="If provided, path to parameter file, otherwise default values used.")
    parser.add_argument('-b','--baseline',nargs=2,type=float,default=[10,10],help="Time before stimulus to start baseline and length of baseline.")
    parser.add_argument('-l','--stimulus',nargs=2,type=float,default=[0,10],help="Time near stimulus to begin recording and length of stimulus data.")
    parser.add_argument('-bw','--bin_width',type=float,nargs=1,default=[0.5],help="Width of bins.")
    parser.add_argument('-tpct','--trial_percentile',nargs=1,default=[99],type=float,help="Percentile to look for in trial baseline histogram.")
    parser.add_argument('-apct','--average_percentile',nargs=1,default=[90],type=float,help="Percentile to look for in average baseline histogram.")
    parser.add_argument('-ibt','--inhibit_bin_threshold',nargs=1,default=[3],type=int,help="Number of bins for inhibtion threshold.")
    parser.add_argument('-ebt','--excite_bin_threshold',nargs=1,default=[3],type=int,help="Number of bins for excitation threshold.")
    
    parser.add_argument('-mu','--mu',type=int,nargs=1,default=[250],help="Number of points for moving average in ISIF. Not implemented.")
    parser.add_argument('-sig','--sigma',type=float,nargs=1,default=[25/1000],help="Sigma (bandwidth) in SDF. Not implemented")

    parser.add_argument('-g','--generate_data',action="store_true",help="If given then will gather data in data_direc.")
    parser.add_argument('-ar','--average_response',action="store_true",help="If given then will gather average neural response in data_direc.")
    parser.add_argument('-pd','--pandas',action="store_true",help="If given then generates pandas dataframe and saves as CSV.")

    args = parser.parse_args() # Read in arguments from terminal

    if args.param_file != "": # If parameter file is given, read in arguments from parameter file.
        param_dict = {} # Create a parameter dictionary to be filled
        with open(args.param_file) as f: # Read parameter file
            for line in f: # Read each line in paramter file
                if line[0] != "#": # Skip commented lines
                    (key, val) = line.split(":") # Split the line by a ':'
                    if key == "baseline" or key =="stimulus": # Read in 'baseline' and 'stimulus' as lists
                        vals = val.split(",") # Split by comma
                        param_dict[key] = [eval(x) for x in vals] # Create numeric list
                    else: # Read in all other parametes
                        param_dict[key] = eval(val) # Evaluate the provided number
        print(param_dict) # Print the parameter dictionary generated from the file
        args.baseline = param_dict["baseline"] # Set baseline argument from file
        args.stimulus = param_dict["stimulus"] # Set stimulus argument from file
        args.bin_width = [param_dict["bin_width"]] # Set bin_width argument from file
        args.trial_percentile = [param_dict["trial_percentile"]] # Set trial_percentile from file
        args.average_percentile = [param_dict["average_percentile"]] # Set average_percentile from file
        args.inhibit_bin_threshold = [param_dict["inhibit_bin_threshold"]] # Set IBT from file
        args.excite_bin_threshold = [param_dict["excite_bin_threshold"]] # Set EBT from file
        #args.mu = [param_dict["mu"]] # not implemented yet # Set mu from file
        #args.sigma = [param_dict["sigma"]] # not implemented yet # Set sigma from file

    print(args) # Print arguments

    save_direc = check_direc(args.save_direc) # Create/check the save directory

    groups = [f.path.split("/")[-1] for f in os.scandir(args.data_direc) if f.is_dir()] # Read in all groups of data to analyze from data directory

    for group in groups: # Iterate through each group in data directory and set target results directory in the save_directory
        check_direc(f"{save_direc}/{group}")
    
    '''
    # Save the parameters that were run with the results directory
    with open(f"{save_direc}/parameters.txt",'w') as file: # Create new parameter file
        for key in sorted(args.__dict__): # Save each paramter to file
            file.write(f"{key}:\t{args.__dict__[key]}\n")
    '''

    if args.generate_data: # If flag is given, generate the trial data
        for group in groups: # Iterate through groups in data directory and analyze trials
            print(f"Started analyzing {group}...") # Print the group that is being processed
            cells = [f.path.split("/")[-1] for f in os.scandir(f"{args.data_direc}/{group}") if f.is_dir()] # Read in the cells to be processed
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-2) # Create a worker pool based on CPUs on machine
            neurons = pnd.get_trial_data_parallel(args.data_direc,group,cells,save_direc) # Gather all the neurons to be processed
            # Create a list to parallelized for trial analysis
            parallel = [(neuron,args.bin_width[0],args.trial_percentile[0],args.average_percentile[0],args.baseline[0],args.baseline[1],args.stimulus[0],args.stimulus[1],args.inhibit_bin_threshold[0],args.excite_bin_threshold[0]) for neuron in neurons] 
            pool.starmap(pnd.analyze_trial_parallel,parallel) # Analyze the trials of each cell, in parallel
            pool.close() # Close the worker pool
            pool.join() # Bring the workers back together
            print(f"Finished analyzing {group}...\n") # Print that the group trials have processed

    if args.average_response: # If flag is given, generate the average response data
        for neural_set in pnd.grab_analyze_avg_neurons(save_direc): # For each group, gather the neurons that need to be analyzed
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-2) # Gather the workers
            pool.map(pnd.analyze_avg_neurons_parallel,neural_set) # Analyze the average response in parallel
            pool.close() # Close the pool of workers
            pool.join() # Bring the workers back together

    if args.pandas: # If flag is given, generate a CSV of all the data
        neurons = [] # Empty list to be filled with all necessary data
        for group in groups: # Iterate through all processed groups
            for neuron_dir in os.listdir(f"{save_direc}/{group}"): # Find all the neuron dictories to report
                if "Neuron" in neuron_dir: # Confirm directory is a Neuron directory
                    neuralFile =open(f"{save_direc}/{group}/{neuron_dir}/neuron.obj","rb") # Read in the neuron objectf file
                    neuron = pickle.load(neuralFile) # Load neuron object to python
                    neuralFile.close() # Close the file
                    neurons.append(neuron.meta_data) # Report the metadata of the neuron for the CSV list
        df = pd.DataFrame(neurons) # Save all the neurons to a pandas DataFrame
        df.to_csv(f"{save_direc}/all_data.csv") # Create CSV out of data frame and save CSV
