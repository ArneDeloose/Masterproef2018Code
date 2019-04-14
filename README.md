# Masterproef2018Code
Python code for the master thesis about bat sounds.

**Tutorial**:

Download the Github folder by clicking the green 'clone or download' button at the top right corner. Download as ZIP and extract the files.

Install Anaconda and open the application. In the left menu, go to environments. At the bottom click 'import'. Pick a name and select the file 'bats_environment.yml'. 
Before running the code, make sure the green arrow in environment is on the bats_environment by clicking on it. This will ensure the necessary packages can be loaded and they remain compatible with new versions.

Within the Github 'Data' folder there is a folder called 'Audio_data'. Under standard conditions, the program assumes the files to be analyzed are located here. It is recommended to copy the data to be analysed here. Some example files are already there.

To run the program, go to Anaconda and on the homescreen, select 'Jupyter Notebook'. This application will open in a browser and show folders. 
Navigate to the location of the downloaded folder and in the folder Data, open the notebooks: Fitting.ipynb and Analysis.ipynb. More detailed instructions for each notebook are written inside. To execute a block, click it and press SHIFT + ENTER.

---

The file 'Fitting' is used to fit a new SOM and a DML matrix. A DML matrix is fitted with data in the Templates folders. A SOM is fitted with data provided by the user.

The file 'Analysis' requires a SOM and a DML matrix to be loaded in. This file contains a tool to visualise the audio data. New templates can be defined here as well. Keep in mind that a new SOM and DML must be fitted whenever a new regular template is defined (because the number of features changes).

The file 'Evaluate' can be used to visualise labeled datapoints on a SOM plot to estimate the performance. Labeled datapoints are stored as eval templates. 

---

For more information various files can be found in the 'documentation' folder.

--- 

For more advanced applications, the code can be loaded in a Python environment. All code is stored in five modules: AD1_Loading, AD2_Spectro, AD3_Features, AD4_SOM and AD5_MDS. The file 'Functions descriptions' in the folder 'documentation' contains a list of all functions with their description, input and output, so a specific function can be located easily. 
