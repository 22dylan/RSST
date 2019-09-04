This repository contains the code and executable for the Representative Storm Selection Tool. The python code can be ran, or a standalone executable can be generated. 

This tool was developed by Dylan R. Sanderson at the US army engineer research and development center sometime in the year of 2017. 

Within the compile_to_exe directory, there are two storm probability data files, as well as a spec file. 
The storm probabiltiy datafiles contain individual storm recurrence rates for the NACCS and S2G USACE studies.
The .spec file is used to create a stand alone executable. 

~~~~
To "complie" the code into a GUI: 

ensure "storm_selection_gui.py" and "storm_selection_gui.spec" are in the same directory
change details in "storm_selection_gui.spec" to meet needs (name, etc.)
if wrapping data in GUI, ensure data is in directory and in .spec file.
ensure correct conda environment is activated (e.g. python 3.X; 32 vs. 64 bit).
in the terminal, navigate to the directory where the .spec file is. 
run the following command: 
	pyinstaller cluster_simple_gui.spec

