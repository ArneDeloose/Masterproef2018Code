# Masterproef2018Code
Python code for the master thesis about bat sounds.

**Tutorial**:

Copy the folder 'Data' from Github. This contains all the necessary files. The files to be analyses must be copied to the folder 'Audio_data' within the Data folder or the full pathway must be specified in the optional argument Audio_data=path.

The code can be run in two ways. For simple applications, there is a Jupyter notebook file prepared. Blocks with 'adapt' need to be adjusted to the data, other blocks can be run normally.

For more complex analysis, specific functions can be called by loading in the file 'AD_functions' in a Python environment. A full overview of every function is given in 'func_description'. 

Various parameters can be adjusted in the txt-file 'parameters'. 

Templates are stored in three folders: templates_arrays, templates_rectangles and templates_images. The third folder is present only for reference purposes. When a new template is created, hash-codes for the rectangle and region are written out (the image shares the hash code of the region).

A number of python packages are used. The anaconda environment can be found on Github as 'bats_environment'.
