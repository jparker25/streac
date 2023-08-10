################################################################################
# helpers.py
# Commonly used functions in program that do not belong to a single script.
# Author: John E. Parker
################################################################################

# import python modules
import os

def check_direc(direc):
    """
    Function to check if directory exists, if not, create directory.

    :param direc: (string) path to directory to make
    :return: returns string of directory name
    """
    os.system(f"mkdir -p {direc}")
    return direc # Return directory as string

def run_cmd(str):
    """
    Runs input from the command line. Prints out command

    :param str: string input to be passed to command line/terminal for execution.
    """
    print("Running cmd:") 
    print(str) # Print string command to execute
    os.system(str) # Execute command
    print() # Add a linebreak after execution

def makeNice(axes):
    """
    Reads in and clean figure axes.

    :param axes: list of axes or single axis of a figure.
    """
    if type(axes) == list: # Check if input is a list
        for axe in axes: # Iterate through list of axes 
            for i in ['left','right','top','bottom']: # Iterate through spines 
                if i != 'left' and i != 'bottom': # Removes top and right spine from axis
                    axe.spines[i].set_visible(False) 
                else: # Left and bottom axis
                    axe.spines[i].set_linewidth(3) # Increase spine width
                    axe.tick_params('both', width=0,labelsize=8) # Remove tick marks and set label size
    else: # If single axis
        for i in ['left','right','top','bottom']:  # Iterate through spines 
                if i != 'left' and i != 'bottom': # Removes top and right spine from axis
                    axes.spines[i].set_visible(False) 
                else: # Left and bottom axis
                    axes.spines[i].set_linewidth(3) # Increase spine width
                    axes.tick_params('both', width=0,labelsize=8) # Remove tick marks and set label size
        
