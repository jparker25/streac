### Welcome to the Spike Train Stimulus Response Classification (STReaC) Toolbox!

The purpose of this toolbox is to analyze and classify spike train datasets, specifically comparing a baseline period to a response period.

Originally, this code was developed for analyzing neural responses to optogenetic stimulation. Much of the code retains some of this language.

This toolbox is currently under review and being actively managed, this README is still a work in progress. For further questions that are not addresed or other necessary clarification please contact the owner of this repository.

<details> 
<summary>Dependencies</summary>

All code was built using Python 3.11. We strongly advise using a virtual environment when running this code. Please see [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html) on how to set up and activate virtual environment on your machine.

Once set up, to install the necessary Python modules please run:

`$ pip install -r requirements.txt`

You are now ready to run the code!
</details>

<details>
<summary>Running the STReaC program</summary>

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

We will now go through all of these possible flags. The order is focused on importance rather than the order listed above. Once our paper on the method is published, we will include the link to the peer-reviewed paper for more information.


<details>
<summary> --data_direc (required) </summary><blockquote>

Directory for where the spike train data is stored, short hand, `-d`. Spike train data should be subdivied into directories pertaining to each group. For example, if there are four groups that exist in `path_to_data_direc`:

```
path_to_data_direc
├── Group_1
├── Group_2
├── Group 3
└── Group 4
```
The group names are arbitrary and can be user defined. From here, each group should contained a list of `N` subdirectories, `Neuron_0001` through `Neuron_N` (e.g. `N=1200` then the last subdirectory would be `Neuron_1200`). Let's assume `Group_1` has three neurons, and within each subdirectory will be three text files, as shown below.

```
path_to_data_direc
├── Group_1
    ├── Neuron_0001
    │   ├── light_on.txt
    │   ├── meta_data.txt
    │   └── spikes.txt
    ├── Neuron_0002
    │   ├── light_on.txt
    │   ├── meta_data.txt
    │   └── spikes.txt
    └── Neuron_0003
        ├── light_on.txt
        ├── meta_data.txt
        └── spikes.txt
```
The `light_on.txt` file will be a list of times corresponding to the start of an experiment. For example, in optogenetic studies, these times would signfity when the light comes on, hence the name. The `spikes.txt` file contains a list of all spike times before, during and after stimulation all stimulation times for that particular neuron. Lastly, `meta_data.txt` contains quantites that may be of interest for the analys that are delimited by a `:`, see the example file below that contain keyword data pertaining to that particular neuron:
```
Mouse#:	3
hemisphere:	right
distance:	1.0
mouse:	experimental mice
cell_num:	1
channel:	10
```
Note that none of the above keywords are necessary and users can define their own keywords. 
</blockquote>
</details>

<details><summary> --save_direc (required) </summary><blockquote>

Path to the directory that will contain the results of the STReaC toolbox. This should a separate directory than what is provided to the `--data_direc` flag.
</blockquote>
</details>

<details><summary> --generate_data (optional) </summary><blockquote>

Flag to indicate that the toolbox should determine all trial responses for each neuron. 
</blockquote>
</details>

<details><summary>  --average_response (optional) </summary><blockquote>

Flag to indicate that the toolbox should determine all average responses of each neuron. This assumes trials responses have been determined already via the `--generate_data` flag.
</blockquote>
</details>

<details><summary>  --pandas (optional) </summary><blockquote>

Flag to provide if the toolbox should output a CSV in the directory specified by `--save_direc` containing a summary of results. This is strongly advised and can be used by a `pandas` data frame (hence the name) for further analysis in Python.
</blockquote>
</details>

<details>
<summary>  
--baseline (optional, default [10,10]) 
</summary> 
<blockquote>

Positive integers, first value read in as a negative number, corresponding to when to start the baseline time and when to end the basline time, relative to each trial's start time as indicated by `light_on.txt`. For example if a trial's light on time is at 22.5 s and `--baseline [10,10]` is the flag provided, then that baseline starts at 12.5s and ends at 22.5s. Figures will show this period as negative time.
</blockquote>
</details>


<details><summary>  --stimulus (optional, default [0,10]) </summary><blockquote>

Positive integers corresponding to when to start the response time and when to end the response time, relative to each trial's start time as indicated by `light_on.txt`. For example if a trial's light on time is at 22.5 s and `--stimulus [0,10]` is the flag provided, then that response starts at 22.5s and ends at 32.5s. Figures will show this period as positive time.
</blockquote>
</details>

<details><summary>  --bin_width  (optional, default [0.5]) </summary> <blockquote>

Width of bins to use when partitioning baseline and response periods with ISIF/SDF analysis.
</blockquote>
</details>

<details><summary>  --trial_percentile (optional, default [99]) </summary> <blockquote>

Percentile used when finding the threshold of baseline bin areas to compare with ISIF/SDF bin areas in determining the trial neural response.
</blockquote>
</details>

<details><summary> --average_percentile (optional, default [90]) </summary> <blockquote>

Percentile used when finding the threshold of baseline bin areas to compare with ISIF/SDF bin areas in determining the average neural response.
</blockquote>
</details>

<details><summary>  --inhibit_bin_threshold (optional, default [3]) </summary> <blockquote>

Number of inhibited bins required to be considered an inhibitory response. Must be greater than 0 unless consecutive inhibited bins are given, then possible to be 0.
</blockquote>
</details>

<details><summary>  --excite_bin_threshold  (optional, default [3]) </summary><blockquote>
Number of excited bins required to be considered an excitatory response. Must be greater than 0 unless consecutive excited bins are given, then possible to be 0.
</blockquote>
</details>

<details>
<summary> 
--consecutive_inhibit_bin_threshold (optional, default [2]) 
</summary>
<blockquote>

Integer number of consecutive inhibited bins required to be considered an inhibitory response. If this value is below the inhibited bin threshold, a valid inhibitory response (partial inhibition, adapting inhibition, biphasic IE, biphasic EI) can be comprised of consecutive or non-consecutive bins. If this value is equal or above the inhibited bin threshold, then only this number of consecutive inhibited bins determine a valid inhibitory response.
</blockquote>
</details>


<details><summary>  --consecutive_excite_bin_threshold (optional, default [2]) </summary> <blockquote>
Integer number of consecutive excited bins required to be considered an excitatory response. If this value is below the excited bin threshold, a valid excitatory response (excitation, biphasic IE, biphasic EI) can be comprised of consecutive or non-consecutive bins. If this value is equal or above the excited bin threshold, then only this number of consecutive excited bins determine a valid excitatory response.
</blockquote>
</details>

<details><summary>  --no_average_shuffling (optional, default False) </summary><blockquote>

During each trial analysis, spike times are shuffled to amplify the number of baseline samples. During the average response analysis, all of the trial data, including shuffled baseline samples, are used in determing periods of excitation and inhibition. However if this flag is provided, then the addtional shuffled samples are excluded.
</blockquote>
</details>

<details><summary>  --isif_vs_sdf (optional, default [25]) </summary><blockquote>

Threshold to determine whether to use the ISIF or SDF for detecting inhibition. For a neuron's set of trials, if any average baseline firing rate is below this threshold then the ISIF will be used for all detection of inhibition. If not, then the SDF will be used for all detection of inhibition.
</blockquote>
</details>

<details><summary> --mu (optional, default [250]) </summary><blockquote>

Number of points to form a moving mean of the interspike interval function (ISIF).</blockquote>
</details>

<details><summary>  --sigma (optional, default [25/1000]) </summary><blockquote>

Bandwidth used in construction of the spike density function (SDF).
</blockquote>
</details>

<details><summary> --param_file (optional) </summary><blockquote>

Parameter file for above optional flags. Ideal for running multiple runs or tracking what flags were provided to the toolbox. Example is given below, from the `parameters.txt` file included in this repository.
```
# default parameter file
baseline:10,10
stimulus:0,10
bin_width:0.5
trial_percentile:99
average_percentile:90
inhibit_bin_threshold:3
excite_bin_threshold:3
consecutive_inhibit_bin_threshold:2
consecutive_excite_bin_threshold:2
mu:250
sigma:25/1000
isif_sdf_threshold:24.25
average_shuffling:Yes
```
</blockquote>
</details>                        
</details>