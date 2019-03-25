# Masterproef2018Code
Python code for the master thesis about bat sounds.

**Tutorial**:

Download the Github folder by clicking the green 'clone or download' button at the top right corner. Download as ZIP and extract the files.

Install Anaconda and open the application. In the left menu, go to environments. At the bottom click 'import'. Pick a name and select the file 'bats_environment.yml'. 
Before running the code, make sure the green arrow in environment is on the bats_environment by clicking on it. This will ensure the necessary packages can be loaded and they remain compatible with new versions.

Within the Github 'Data' folder there is a folder called 'Audio_data'. Under standard conditions, the program assumes the files to be analyzed are in here.

To run the program, go to Anaconda and on the homescreen, select 'Jupyter Notebook'. This application will open in a browser and show folders. 
Navigate to the location of the downloaded folder and in the folder Data, open Notebook.ipynb. Follow the instructions within this file. To execute a block, click it and press SHIFT + ENTER.

--- 

For more advanced applications, specific functions can be called by loading in the file 'AD_functions' in a Python environment. A full overview of every function is given in 'func_description'. 

Various parameters can be adjusted in the txt-file 'parameters'. 

Templates are stored in three folders: templates_arrays, templates_rectangles and templates_images. The third folder is present only for reference purposes. 
When a new template is created, hash-codes for the rectangle and region are written out (the image shares the hash code of the region).
