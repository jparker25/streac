# Welcome to the Spike Train Stimulus Response Classification (STReaC) Toolbox!

The purpose of this toolbox is to analyze and classify spike train datasets, specifically comparing a baseline period to a response period.

Originally, this code was developed for analyzing neural responses to optogenetic stimulation. Much of the code retains some of this language.

This toolbox is currently under review and being actively managed, this README is still a work in progress. For further questions that are not addresed or other necessary clarification please contact the owner of this repository.

## Dependencies

All code was built using Python 3.11. We strongly advise using a virtual environment when running this code. Please see [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html) on how to set up and activate virtual environment on your machine.

Once set up, to install the necessary Python modules please run:

`$ pip install -r requirements.txt`

You are now ready to run the code!

## Running the STReaC program

One line is sufficient to run the program, assuming the proper flags and directory paths are supplied. To see a print out of all the options run:

`$ python3 analyze_data.py --help`

which should display
```
usage: analyze_data.py [-h] -d DATA_DIREC -r SAVE_DIREC [-f PARAM_FILE] [-b BASELINE BASELINE] [-l STIMULUS STIMULUS] [-bw BIN_WIDTH] [-tpct TRIAL_PERCENTILE] [-apct AVERAGE_PERCENTILE] [-ibt INHIBIT_BIN_THRESHOLD]
                       [-ebt EXCITE_BIN_THRESHOLD] [-cibt CONSECUTIVE_INHIBIT_BIN_THRESHOLD] [-cebt CONSECUTIVE_EXCITE_BIN_THRESHOLD] [-NoShuff] [-ISIFSDF ISIF_VS_SDF] [-mu MU] [-sig SIGMA] [-g] [-ar] [-pd]

Classification and data analysis of neural spiking.

options:
  -h, --help            show this help message and exit
  -d DATA_DIREC, --data_direc DATA_DIREC
                        Directory where raw data is stored.
  -r SAVE_DIREC, --save_direc SAVE_DIREC
                        Directory where processed data is stored.
  -f PARAM_FILE, --param_file PARAM_FILE
                        If provided, path to parameter file, otherwise default values used.
  -b BASELINE BASELINE, --baseline BASELINE BASELINE
                        Time before stimulus to start baseline and length of baseline.
  -l STIMULUS STIMULUS, --stimulus STIMULUS STIMULUS
                        Time near stimulus to begin recording and length of stimulus data.
  -bw BIN_WIDTH, --bin_width BIN_WIDTH
                        Width of bins.
  -tpct TRIAL_PERCENTILE, --trial_percentile TRIAL_PERCENTILE
                        Percentile to look for in trial baseline histogram.
  -apct AVERAGE_PERCENTILE, --average_percentile AVERAGE_PERCENTILE
                        Percentile to look for in average baseline histogram.
  -ibt INHIBIT_BIN_THRESHOLD, --inhibit_bin_threshold INHIBIT_BIN_THRESHOLD
                        Number of bins for inhibtion threshold.
  -ebt EXCITE_BIN_THRESHOLD, --excite_bin_threshold EXCITE_BIN_THRESHOLD
                        Number of bins for excitation threshold.
  -cibt CONSECUTIVE_INHIBIT_BIN_THRESHOLD, --consecutive_inhibit_bin_threshold CONSECUTIVE_INHIBIT_BIN_THRESHOLD
                        Number of bins for inhibtion threshold.
  -cebt CONSECUTIVE_EXCITE_BIN_THRESHOLD, --consecutive_excite_bin_threshold CONSECUTIVE_EXCITE_BIN_THRESHOLD
                        Number of bins for excitation threshold.
  -NoShuff, --no_average_shuffling
                        If given then no average shuffling.
  -ISIFSDF ISIF_VS_SDF, --isif_vs_sdf ISIF_VS_SDF
                        Any baseline FR below this, all trials use ISIF for inhibition.
  -mu MU, --mu MU       Number of points for moving average in ISIF. Not implemented.
  -sig SIGMA, --sigma SIGMA
                        Sigma (bandwidth) in SDF. Not implemented
  -g, --generate_data   If given then will gather data in data_direc.
  -ar, --average_response
                        If given then will gather average neural response in data_direc.
  -pd, --pandas         If given then generates pandas dataframe and saves as CSV

```

We will now go through all of these possible flags. Once our paper on the method is published, we will include the link to peer-reviewed paper.
