# Masterproef2018Code
Python code for the master thesis about bat sounds.

**Tutorial**:

The code uses a specific folder structure which goes as follows:

-Top folder. This folder needs to be set as the current directory for the code to run. It has to contain the following (all of which can be found in the 'data folder'):
->AD_functions.py: contains all the python functions.
->templates_arrays and templates_rectangles folder: initial sets can be downloaded from Github. These can be expanded. Contains subfolders with codes for each specific bat.
->Audio_data: the data you want to analyse. Must all be of the same type (no mixes of TE and non-TE data).
->parameters.txt: a txt file with a number of adjustable parameters defined.

If one of these files is not in the right location, the full pathway can be specified as an optional argument which shares the name with the necessary folder. i.e. AD.write_output(***, Audio_data='C/users/documents/..., templates_arrays=***).
Any results will get written out to a folder 'results' that is created in the top folder by default. If you want to write them out anywhere else, you need to specify this in an optional argument 'results'.
Time-expanded data can be analysed by defining the optional argument 'channel' as 'l' or 'r' (left or right channel) and exp_factor as the time expansion factor (if this isn't given the default is used (10)).

The easiest way to run the code is with a script that starts in the following way:

>import os

>path='...';

>os.chdir(path)

>import AD_functions as AD

Where 'path' must be set to refer to the directory that stores the necessary files. All functions can now be called using AD.funcName(). A full description of all functions can be found in the file 'func_descr' in the code folder.

Parameters can be changed in the parameters.txt file. It is not recommended to adjust anything in the AD_functions.py file.

A number of python packages are used. The anaconda environment can be found on Github.