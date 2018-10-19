# Masterproef2018Code
Python code for the master thesis about bat sounds.

**Tutorial**:

To run code, two things are needed: the file AD_functions from the code folder (which defines all the functions needed) and the files 'ppip-1µl1µA044_AAT.wav' and 'eser-1µl1µA030_ACH.wav' from the audio_data folder (to make templates). 

All scripts start in the following way:

>import os

>path='C:/Users/arne/Documents/Github/Masterproef2018Code/Code';

>os.chdir(path)

>import AD_functions as AD

Where 'path' must be replaced to refer to the directory that stores the necessary files. All functions can now be called using AD.funcName(). A full description of all functions can be found in the file 'func_descr' in the code folder.

The file 'Comparison_pip' gives an example of how the code can be used.
