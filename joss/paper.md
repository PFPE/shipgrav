---
title: 'shipgrav: A Python package for marine gravimetry'
tags:
  - Python
  - gravity
  - marine geophysics
  - UNOLS
authors:
  - name: Hannah F. Mark
    orcid: 0000-0002-1722-3759
    corresponding: true
    affiliation: 1
  - name: Jasmine Zhu
    affiliation: 1
  - name: Masako Tominaga
    orcid: 0000-0002-1169-4146
    affiliation: 1
  - name: Daniel Aliod
    affiliation: 2
  - name: Maurice Tivey
    affiliation: 1
affiliations:
 - name: Woods Hole Oceanographic Institution, USA
   index: 1
 - name: Dynamic Gravity Systems, USA
   index: 2
bibliography: paper.bib

---

# Summary

The acceleration due to gravity is not uniform across Earth's surface because of local variations in topography and in the density distribution within the Earth. Measurements of local gravity anomalies -- the difference between theoretically predicted gravity and observed gravity -- can therefore be used to infer density variations in Earth's interior which influence tectonic and geodynamic processes [e.g. @escartin_tectonic_1998;@shinevar_causes_2019;@watts_seismic_2021].

Reducing gravimeter data to gravity anomalies requires two main stages of data processing: first, converting or calibrating a sensor measurement to gravity; and second, subtracting theoretical contributions to the gravity signal to obtain the local anomaly. For marine gravimetry, the first stage involves reading a variety of operator-specific data formats, applying meter calibration factors, and correcting for meter drift; and the second stage involves subtracting out corrections for theoretical gravity on a uniform ellipsoid, tidal effects [@longman_formulas_1959], and acceleration of the measurement platform [@eotvos_experimenteller_1919;@lacoste_measurement_1967]. Once the gravity anomaly is obtained, further processing draws on independent bathymetry data and estimated thermal structure to connect gravity anomalies to subsurface structure [@parker_rapid_1972].

Here we present a Python package implementing the calibrations and corrections commonly used for marine gravity processing including tools for meter calibration, anomaly calculation, and the estimation of variations in the thickness and/or density of subsurface layers. The package includes specific tools for working with data from instruments managed by the Potential Field Pool Equipment (PFPE) facility on University National Oceanographic Laboratory System (UNOLS) vessels, and can easily be extended for use with other vessels and instruments.

# Statement of need

`shipgrav` is a Python package for marine gravimetry, designed as a tool for the marine geophysical research community. Marine gravity data is routinely collected on UNOLS vessels and archived for public access (after an embargo period) through the [Rolling Deck to Repository](https://www.rvdata.us) program, but researchers who work with this data have traditionally used their own personal processing tools for data analysis. There are some public software tools that handle parts of the processing workflow, such as @jrleeman's [Longman tide correction](https://github.com/jrleeman/LongmanTide) package and a [MATLAB toolkit](https://github.com/MAG-tominaga/DgSGravCode) that does some AT1M data processing, but the authors are not aware of any complete processing workflows for this type of data that are publicly available and entirely open-source. One major goal of `shipgrav` is to make marine gravity data more accessible to the broader community, including researchers who want to incorporate gravity into other geophysical analyses without re-writing implementations of the standard workflow or needing a MATLAB license. The package includes a suite of example scripts demonstrating all the steps for marine gravity processing using publicly available data. These examples can be used as teaching tools, and as a starting framework for research applications.

`shipgrav` includes a set of vessel-specific input functions which take into account the differences in file formats between different instruments and operators. While shipboard gravimeters are commonly configured to output the same or similar sets of data fields in ASCII files, there is not a standard data string format. Known formats for gravimeters on UNOLS vessels are built into `shipgrav` so that users can easily read in data. The framework is designed so that users can plug in their own functions to read and parse files for vessels or formats that are not yet included. There are also functions for reading separately recorded navigation data and synchronizing geographic coordinates with gravimeter records based on timestamps, which is necessary for integrating gravity data with other geophysical analyses.

An important component of `shipgrav` is the inclusion of routines for calculating and applying cross-coupling corrections. Cross-coupling refers to situations where horizontal accelerations are mapped to the vertical, and vice versa, due to some amount of uncompensated movement of the sensor platform [@wall_cross-coupling_1966;@bower_determination_1966;@lacoste_crosscorrelation_1973]. This is important because the gravimeters in use on UNOLS vessels have recently begun to transition from BGM-3 to DgS-AT1M instruments as the National Science Foundation has supported PFPE to update the instrument pool. The AT1M sensors use a beam and spring balance which is subject to cross-coupling errors, while the BGM-3 sensor used a different design which is not affected by cross-coupling [@bell_evaluation_1986]. Users accustomed to BGM-3 data therefore will need to add cross-coupling corrections into their processing workflows. While the cross-coupling correction is typically very small for data collected in calm seas, it can become important in rough sea states.

Satellite-derived global gravity measurements are widely available and are often used as a comparison point for ship-based measurements [e.g. @sandwell_new_2014]. Therefore, `shipgrav` includes functions that can be used for basic processing of 2D gridded gravity measurements as well as 1D trackline measurements.

<!-- # Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

The development of shipgrav was supported by the Potential Field Pool Equipment (PFPE) facility, which is funded by the National Science Foundation under grants OCE-2234277 and OCE-1824508; and by the Office of Naval Research under DURIP grant N00014-23-1-2475. Functions for calculating thermal structure were adapted from scripts provided by Mark Behn. Nigel Brady at DGS helpfully answered many questions about the AT1M system.

# References
