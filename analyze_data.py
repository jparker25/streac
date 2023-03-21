################################################################################
# classify_data.def
# Runs classification scripts based on parameters to analyze neural spike trains
# Author: John E. Parker
################################################################################

# import python modules
#import pandas as pd
import os, sys, pickle
import argparse
import multiprocessing
import pandas as pd

# import user made modules
import process_neural_data as pnd
from helpers import *
#import properties

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification and data analysis of neural spiking.")
    parser.add_argument('-p','--data_direc',type=str,default="/Users/johnparker/streac/data",help="Directory where raw data is stored.",required=True)
    parser.add_argument('-s','--save_direc',type=str,default="/Users/johnparker/streac/results",help="Directory where processed data is stored.",required=True)

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

    args = parser.parse_args()

    if args.param_file != "":
        param_dict = {}
        with open(args.param_file) as f:
            for line in f:
                if line[0] != "#":
                    (key, val) = line.split(":")
                    if key == "baseline" or key =="stimulus":
                        vals = val.split(",")
                        param_dict[key] = [eval(x) for x in vals]
                    else:
                        param_dict[key] = eval(val)
        print(param_dict)
        args.baseline = param_dict["baseline"]
        args.stimulus = param_dict["stimulus"]
        args.bin_width = [param_dict["bin_width"]]
        args.trial_percentile = [param_dict["trial_percentile"]]
        args.average_percentile = [param_dict["average_percentile"]]
        args.inhibit_bin_threshold = [param_dict["inhibit_bin_threshold"]]
        args.excite_bin_threshold = [param_dict["excite_bin_threshold"]]
        #args.mu = [param_dict["mu"]] # not implemented yet
        #args.sigma = [param_dict["sigma"]] # not implemented yet

    print(args)

    save_direc = check_direc(args.save_direc)

    groups = [f.path.split("/")[-1] for f in os.scandir(args.data_direc) if f.is_dir()]

    for group in groups:
        check_direc(f"{save_direc}/{group}")
    
    '''
    with open(f"{save_direc}/parameters.txt",'w') as file:
        for key in sorted(args.__dict__):
            file.write(f"{key}:\t{args.__dict__[key]}\n")
    '''

    if args.generate_data:
        for group in groups:
            print(f"Started analyzing {group}...")
            cells = [f.path.split("/")[-1] for f in os.scandir(f"{args.data_direc}/{group}") if f.is_dir()]
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)
            neurons = pnd.get_trial_data_parallel(args.data_direc,group,cells,save_direc)
            parallel = [(neuron,args.bin_width[0],args.trial_percentile[0],args.average_percentile[0],args.baseline[0],args.baseline[1],args.stimulus[0],args.stimulus[1],args.inhibit_bin_threshold[0],args.excite_bin_threshold[0]) for neuron in neurons]
            pool.starmap(pnd.analyze_trial_parallel,parallel)
            pool.close()
            pool.join()
            print(f"Finished analyzing {group}...\n")

    if args.average_response:
        for neural_set in pnd.grab_analyze_avg_neurons(save_direc):
            pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)
            pool.map(pnd.analyze_avg_neurons_parallel,neural_set)
            pool.close()
            pool.join()

    if args.pandas:
        neurons = []
        for group in groups:
            for neuron_dir in os.listdir(f"{save_direc}/{group}"):
                if "Neuron" in neuron_dir:
                    neuralFile =open(f"{save_direc}/{group}/{neuron_dir}/neuron.obj","rb")
                    neuron = pickle.load(neuralFile)
                    neuralFile.close()
                    neurons.append(neuron.meta_data)
        df = pd.DataFrame(neurons)
        df.to_csv(f"{save_direc}/all_data.csv")
