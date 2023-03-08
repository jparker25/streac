################################################################################
# helpers.py
# Commonly used functions in program that do not belong to a single script.
# Author: John E. Parker
################################################################################

# import python modules
import os

def check_direc(direc):
    '''
    Simple function to create directory if it does not exist.
    '''
    if not os.path.exists(direc):
        os.mkdir(direc)
    return direc

def run_cmd(str):
    '''
    Runs str from the command line.
    '''
    print("Running cmd:")
    print(str)
    os.system(str)
    print()

def makeNice(axes):
    if type(axes) == list:
        for axe in axes:
            for i in ['left','right','top','bottom']:
                if i != 'left' and i != 'bottom':
                    axe.spines[i].set_visible(False)
                    axe.tick_params('both', width=0,labelsize=8)
                else:
                    axe.spines[i].set_linewidth(3)
                    axe.tick_params('both', width=0,labelsize=8)
    else:
        for i in ['left','right','top','bottom']:
                if i != 'left' and i != 'bottom':
                    axes.spines[i].set_visible(False)
                    axes.tick_params('both', width=0,labelsize=8)
                else:
                    axes.spines[i].set_linewidth(3)
                    axes.tick_params('both', width=0,labelsize=8)
        
