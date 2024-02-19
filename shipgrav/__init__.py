"""
Introduction
---------------------

shipgrav is a Python package with utilities for reading and processing marine gravity data from UNOLS ships. At time of writing, the UNOLS fleet is transitioning away from BGM3 gravimeters to DGS AT1M meters managed by the Potential Field Pool Equipment (PFPE) facility. shipgrav is able to read files from both types of meters, as well as navigation data and other vessel feeds.

DGS gravimeters output two types of files: serial, or 'raw' files; and 'laptop' files. Raw files are written from the serial port, and contain counts values that can be calibrated to retrieve the gravity signal. In this documentation we use the terms 'serial' and 'raw' interchangeably.  What we refer to as laptop files are lightly processed onboard the meter and output with (biased) gravity values alongside other information.

Installation
------------

shipgrav's dependencies are

* Python 3.9+
* numpy
* scipy
* pandas 2.0+
* statsmodels
* tomli
* pyyaml

matplotlib is also required to run some of the example scripts.

To install and use shipgrav, using an environment managing tool is recommended. An exemplary way to do this using `conda <https://anaconda.org>`_ would be to use these commands: ::

    conda create --name shipgrav numpy scipy pandas statsmodels tomli pyyaml matplotlib
    conda activate shipgrav
    cd /path/to/put/source/files
    git clone https://github.com/hfmark/shipgrav.git
    cd shipgrav/
    pip install .

Modules and files
-----------------

shipgrav consists of the modules ``io``, ``nav``, ``grav``, and ``utils``, along with the file ``database.toml`` and a set of example scripts and data files. 

* ``io`` contains functions for reading different kinds of gravimeter files and associated navigation files.
* ``nav`` contains functions for handling coordinate systems.
* ``grav`` contains functions for processing gravity data and calculating various anomalies.
* ``utils`` is a catch-all of other things we need. 
* ``database.toml`` holds some ship-specific constants and other information for UNOLS vessels.
* the scripts in ``example-scripts`` walk through the steps of reading and processing UNOLS gravimeter data for a set of sample data files.

Data directories
----------------

You can organize your data however you like; shipgrav does not care as long as you tell it where to look. However, the example scripts are all set up using a particular organization system that you may like to emulate. You can see this structure in the data files that accompany the example scripts. 

Navigation data
---------------

Which navigation data should you use to process gravimeter data?

In an ideal world, the DGS meter pulls navigation info from the ship's feed and synchronizes it perfectly with acquisition such that the output files have the correct geographic coordinates in them at the start. In practice, this synchronization doesn't always work as expected (see ``example-scripts/dgs_raw_comp.py`` for a case where the serial files do not have GPS info). So, we like to take the timestamped navigation data directly from the ship's feed and match up the DGS timestamps to obtain more accurate coordinates.

The database file included in shipgrav lists the navigation talkers that we expect are good to use for specific UNOLS vessels. Find the files that contain those feeds, and you should be able to read in timestamped coordinates from them.

Example scripts
---------------

The scripts in the ``example-scripts`` directory use sample data files to run through some common workflows. The sample data files are not available on github because of file size limits but we're working on making them available for download elsewhere.

``dgs_bgm_comp.py`` reads data from DGS and BGM gravimeter files from R/V Thompson cruise TN400. The files are lightly processed to obtain the FAA (including syncing with navigation data for more accurate locations), and the FAA is plotted alongside corresponding satellite-derived FAA.

``dgs_raw_comp.py`` reads laptop and serial data from R/V Sally Ride cruise SR2312. The serial data are calibrated and compared to the laptop data. The laptop data are processed to FAA and plotted alongside satellite-derived FAA.

``dgs_ccp_calc.py`` reads an example set of laptop files provided by DGS, calculates the FAA and various kinematic variables, and fits for cross-coupling coefficients. The coefficients derived from the data are compared to values that were provided by DGS. The cross-coupling correction is applied and the data are plotted with and without correction.

``mru_coherence.py`` reads laptop data and other feeds from R/V Sally Ride cruise SR2312. The FAA is calculated, and MRU info is read to obtain time series of pitch, roll, and heave. Coherence is caluclated between those and each of the four monitors output by the gravimeter for the cross-coupling correction.

``interactive_line_pick.py`` reads laptop data and navigation data from R/V Sally Ride cruise SR2312. The script generates an interactive plot with a cursor for users to select segments of the time series data based on mapped locations, in order to extract straight line segments from a cruise track. The selected segments are written to files that can be re-read by the next script...

``RMBA_calc.py`` reads an example of data from a line segment (from the interactive line picker) and calcualtes the residual mantle bouger anomaly (RMBA) as well as estimated crustal thickness variations.

Help!
-----

``FileNotFound`` **errors:** check the filepaths in your scripts and make sure that (a) there are no typos, and (b) you are pointing toward the actual locations of your data files.

**Other file reading errors:** shipgrav does its best to read a variety of file formats from UNOLS gravimeters, but we can't read files that we don't know enough about ahead of time. In some cases, a file cannot be read because we don't yet know how to pass the file to the correct parsing function. Most primary i/o functions in shipgrav have an option where users can supply their own file-parsing function, so one option is to write such a function (following the examples in shipgrav for known vessel file formats) and plug that in via the appropriate kwarg (usually named ``ship_function``). You can also send an example file and information to PFPE so that we can update shipgrav.

**The anomaly I've calculated looks really weird:** a good first step is to compare your (lowpass filtered) FAA to satellite data (e.g., Sandwell et al. 2014, doi: 10.1126/science.1258213). If that looks very different, you can start checking whether the data is being read properly; whether the sample rate of the data is consistent with your expectations; whether there are anomalous spikes or dropouts in the data that need to be cleaned out; and whether the corrections used to calculate the FAA seem to have reasonable magnitudes.

**If you have some other question that's not answered here:** you can try contacting PFPE for specific assistance with processing UNOLS gravimeter data.

Testing
-------

shipgrav comes with a set of unit tests. To run them for yourself, navigate to the ``tests/`` directory and run ``__main__.py`` (in an environment with dependencies installed, naturally).


Contributing
------------

Do you have ideas for making this software better? Go ahead and `raise an issue <https://github.com/hfmark/shipgrav/issues>`_ on the github page or, if you're a savvy Python programmer, submit a pull request. You can also email PFPE.

"""
