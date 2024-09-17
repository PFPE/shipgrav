"""
Introduction
---------------------

shipgrav is a Python package with utilities for reading and processing marine gravity data from UNOLS ships. At time of writing, the UNOLS fleet is transitioning away from BGM3 gravimeters to DGS AT1M meters managed by the Potential Field Pool Equipment (PFPE) facility. shipgrav is able to read files from both types of meters, as well as navigation data and other vessel feeds. shipgrav functions can then be used to process gravity data to FAA, MBA, RMBA, and crustal thickness estimates.

DGS gravimeters output two types of files: serial, or 'raw' files; and 'laptop' files. Raw files are written from the serial port, and contain counts values that can be calibrated to retrieve the gravity signal. In this documentation we use the terms 'serial' and 'raw' interchangeably.  What we refer to as laptop files are lightly processed onboard the meter and output with (biased) gravity values alongside other information.

Installation
------------

shipgrav can be installed from `PyPI <https://pypi.org/project/shipgrav/>`_ using ``pip``. we recommend using an environment management tool like `conda <https://anaconda.org>`_. An exemplary set of commands to make a conda enviroment with shipgrav would be: ::

    conda create --name shipgrav numpy scipy pandas statsmodels tomli pyyaml matplotlib
    conda activate shipgrav
    pip install shipgrav

shipgrav's dependencies are

* Python 3.9+
* numpy
* scipy
* pandas 2.0+
* statsmodels
* tomli
* pyyaml
* matplotlib (optional, needed to run some of the example scripts)

The example scripts are `available on github <https://github.com/PFPE/shipgrav>`_. They are not packaged with the PyPI package and must be downloaded separately.

Modules and files
-----------------

shipgrav consists of the modules ``io``, ``nav``, ``grav``, and ``utils``, along with the file ``database.toml`` and a set of example scripts. 

* ``io`` contains functions for reading different kinds of gravimeter files and associated navigation files.
* ``nav`` contains functions for handling coordinate systems.
* ``grav`` contains functions for processing gravity data and calculating various anomalies.
* ``utils`` is a catch-all of other things we need. 
* ``database.toml`` holds some ship-specific constants and other information for UNOLS vessels.
* the scripts in ``example-scripts/`` walk through the steps of reading and processing UNOLS gravimeter data for a set of data files that are publicly available via `R2R (Rolling Deck to Repository) <https://www.rvdata.us>`_.

Data directories
----------------

You can organize your data however you like; shipgrav does not care as long as you tell it where to look. However, to run the example scripts, you will need to download some data and ensure that the scripts are pointed to the paths where you put that data. A shell script ``download_data.sh`` is provided in ``example-scripts/`` to help download and organize the data.

Ties and bias
-------------

The ``database.toml`` file contains some bias values for UNOLS vessels. These are provided for your convenience, but are not necessarily up-to-date with recent gravity ties.

Navigation data
---------------

Which navigation data should you use to process gravimeter data?

In an ideal world, the gravimeter pulls navigation info from the ship's feed and synchronizes it perfectly with acquisition such that the output files have the correct geographic coordinates in them at the start. In practice, this synchronization doesn't always work as expected (see ``example-scripts/dgs_raw_comp.py`` for a case where the serial files do not have GPS info). So, we like to take the timestamped navigation data directly from the ship's feed and match up the gravimeter timestamps to obtain more accurate coordinates.

The database file included in shipgrav lists the navigation talkers that we expect are good to use for specific UNOLS vessels. Find the files that contain those feeds, and you should be able to read in timestamped coordinates from them.

Example scripts
---------------

The scripts in the ``example-scripts/`` directory use publicly available data files to run through some common workflows for marine gravity processing. The data files can be downloaded from R2R and Zenodo. The ``download_data.sh`` shell script in ``example-scripts/`` uses ``curl``, ``tar``, and ``unzip`` to download all the data files for you and unpack them into a directory structure that matches what the examples expect. You can also download and unpack the files on your own from the sources listed below:

* https://doi.org/10.7284/151470 - TN400 BGM3 data
* https://doi.org/10.7284/151457 - TN400 nav data
* https://doi.org/10.7284/157179 - SR2312 DGS laptop data
* https://doi.org/10.7284/157188 - SR2312 nav data
* https://doi.org/10.7284/157177 - SR2312 mru data
* https://doi.org/10.5281/zenodo.12733929 - TN400 DGS raw and laptop data, SR2312 DGS raw data, R/V Ride DGS meter and Hydrins metadata, satellite FAA tracks for comparison, example file for RMBA calculation

If you are not using the download script provided, we recommend looking at what it does and mimicking the directory structure it sets up for the data files. Otherwise, you will have to edit the examples to point to the correct filepaths for wherever you put the data in your system.

``dgs_bgm_comp.py`` reads data from DGS and BGM gravimeter files from R/V Thompson cruise TN400. The files are lightly processed to obtain the FAA (including syncing with navigation data for more accurate locations), and the FAA is plotted alongside corresponding satellite-derived FAA.

.. image:: _static/TN400_FAA.png
   :alt: FAA for TN400 data from BGM3 and DGS, compared to satellite data.
   :height: 250px
   :align: center

``dgs_raw_comp.py`` reads laptop and serial data from R/V Sally Ride cruise SR2312. The serial data are calibrated and compared to the laptop data. The laptop data are processed to FAA and plotted alongside satellite-derived FAA.

.. image:: _static/SR2312_serial_laptop.png
   :alt: SR2312 raw gravity from serial and laptop files.
   :height: 250px
   :align: center

``dgs_ccp_calc.py`` reads DGS files from R/V Thompson cruise TN400, calculates the FAA and various kinematic variables, and fits for cross-coupling coefficients. The cross-coupling correction is applied and the data are plotted with and without correction.

.. image:: _static/TN400_ccp.png
   :alt: FAA for TN400, with and without cross-coupling correction applied.
   :height: 250px
   :align: center

``mru_coherence.py`` reads laptop data and other feeds from R/V Sally Ride cruise SR2312. The FAA is calculated, and MRU info is read to obtain time series of pitch, roll, and heave. Coherence is caluclated between those and each of the four monitors output by the gravimeter for the cross-coupling correction.

.. image:: _static/roll_coherence.png
   :alt: Coherence between monitors and roll for SR2312.
   :height: 250px
   :align: center

``interactive_line_pick.py`` reads laptop data and navigation data from R/V Sally Ride cruise SR2312. The script generates an interactive plot with a cursor for users to select segments of the time series data based on mapped locations, in order to extract straight line segments from a cruise track. The selected segments are written to files that can be re-read by the next script...

.. image:: _static/cursor.png
   :alt: Interactive line segment picker window.
   :height: 250px
   :align: center

``RMBA_calc.py`` reads an example of data from a line segment (from the interactive line picker) and calculates the residual mantle bouger anomaly (RMBA) as well as estimated crustal thickness variations.

.. image:: _static/rmba.png
   :alt: RMBA for a segment of SR2312, with back-calculated "recovered" signal.
   :height: 250px
   :align: center

Help!
-----

``FileNotFound`` **errors:** check the filepaths in your scripts and make sure that (a) there are no typos, and (b) you are pointing toward the actual locations of your data files.

**Other file reading errors:** shipgrav does its best to read a variety of file formats from UNOLS gravimeters, but we can't read files that we don't know enough about ahead of time. In some cases, a file cannot be read because we don't yet know how to pass the file to the correct parsing function. Most primary i/o functions in shipgrav have an option where users can supply their own file-parsing function, so one option is to write such a function (following the examples in shipgrav for known vessel file formats) and plug that in via the appropriate kwarg (usually named ``ship_function``). You can also send an example file and information to PFPE so that we can update shipgrav.

**The anomaly I've calculated looks really weird:** a good first step is to compare your (lowpass filtered) FAA to satellite data (e.g., `Sandwell et al. 2014 <https://doi.org/10.1126/science.1258213>`_). If that looks very different, you can start checking whether the data is being read properly; whether the sample rate of the data is consistent with your expectations; whether there are anomalous spikes or dropouts in the data that need to be cleaned out; and whether the corrections used to calculate the FAA seem to have reasonable magnitudes.

**I want to use shipgrav, but my data is not from a UNOLS vessel:** the functions and workflows in shipgrav are entirely adaptable to use with data from other sources. You will need to determine the data format for your gravimeter files, and write or adapt a function to read that data. There are examples in the ``io`` module. If you have raw data files, you will also need to know the calibration constants and apply those. Once the data have been read (and calibrated), you should be able to apply all of the other shipgrav functions for processing. 

**I'm going to sea and want to be able to access this documentation offline:** this is all auto-generated from text included in the shipgrav source files! So one option is just to go read those (the main part of the documentation is in ``shipgrav/__init__.py``). To view it as a nice webpage, you can build the documentation locally using ``sphinx``. Install ``sphinx`` in your conda environment, run the command ``make html`` in the ``docs/`` directory, and then open ``docs/_build/html/index.html`` in your browser to view the documentation.

**If you have some other question that's not answered here:** you can try contacting PFPE at pfpe-internal(at)whoi.edu for specific assistance with processing UNOLS gravimeter data.

Testing
-------

shipgrav comes with a set of unit tests. To run them for yourself, navigate to the ``tests/`` directory and run ``__main__.py`` (in an environment with dependencies installed, naturally).


Contributing
------------

Do you have ideas for making this software better? Go ahead and `raise an issue <https://github.com/PFPE/shipgrav/issues>`_ on the github page or, if you're a savvy Python programmer, submit a pull request. You can also email PFPE at pfpe-interal(at)whoi.edu.

"""
