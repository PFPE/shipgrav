#!/bin/bash

########################################################################
# This script downloads, unpacks, and organizes all of the data files
# used by the example python scripts in this directory. 
########################################################################

# check if a data/ dir already exists, warn for overwrite
if [ -d "data/" ]; then
echo "data dir already exists"
while true; do
read -n 1 -p "Enter y to continue/overwrite, or q to quit -> " iq
if [[ $iq == 'q' ]]; then
exit 0
elif [[ $iq == 'y' ]]; then
break
fi
done
fi

# make the directory tree to put things in
mkdir -p data/Ride/SR2312/NAV
mkdir -p data/Ride/SR2312/gravimeter/DGS/serial
mkdir -p data/Ride/SR2312/openrvdas/data/
mkdir -p data/Ride/SR2312/openrvdas/doc/devices
mkdir -p data/Thompson/TN400/NAV
mkdir -p data/Thompson/TN400/gravimeter/DGS/serial
mkdir -p data/Thompson/TN400/gravimeter/BGM3/serial

# download zenodo repo, unpack
# since archive file structure follows what we're using, can just unpack and it will populate
curl -X GET https://zenodo.org/api/records/12733929/files/data.zip/content --output zenodo.zip
unzip zenodo.zip
# clean up
rm zenodo.zip

# set up temp dir for downloading other things and unpacking, move over there
mkdir -p temp
cd temp

# TN400 data: BGM3 and nav
# https://doi.org/10.7284/151470 (grav)
# https://doi.org/10.7284/151457 (nav)
curl -X GET https://service.rvdata.us/data/cruise/TN400/fileset/151470 --output TN400-grav.tar.gz
tar -xvzf TN400-grav.tar.gz
curl -X GET https://service.rvdata.us/data/cruise/TN400/fileset/151457 --output TN400-nav.tar.gz
tar -xvzf TN400-nav.tar.gz
for e in 20220313 20220314; do
mv TN400/151470/data/*$e* ../data/Thompson/TN400/gravimeter/BGM3/serial
mv TN400/151457/data/*$e* ../data/Thompson/TN400/NAV
done
# clean up
rm -rf TN400
rm TN400-grav.tar.gz
rm TN400-nav.tar.gz

# SR2312 data: DGS, nav, and mrus
# https://doi.org/10.7284/157179 (grav)
# https://doi.org/10.7284/157188 (nav)
# https://doi.org/10.7284/157177 (mru)
curl -X GET https://service.rvdata.us/data/cruise/SR2312/fileset/157179 --output SR2312-grav.tar.gz
tar -xvzf SR2312-grav.tar.gz
curl -X GET https://service.rvdata.us/data/cruise/SR2312/fileset/157188 --output SR2312-nav.tar.gz
tar -xvzf SR2312-nav.tar.gz
curl -X GET https://service.rvdata.us/data/cruise/SR2312/fileset/157177 --output SR2312-mru.tar.gz
tar -xvzf SR2312-mru.tar.gz
for e in 16 17 18 19 20; do
mv SR2312/157179/data/*_202306$e*.dat ../data/Ride/SR2312/gravimeter/DGS
mv SR2312/157188/data/*2023-06-$e.txt  ../data/Ride/SR2312/NAV
mv SR2312/157177/data/*2023-06-$e.txt  ../data/Ride/SR2312/openrvdas/data
done
# clean up
rm -rf SR2312
rm SR2312-grav.tar.gz
rm SR2312-nav.tar.gz
rm SR2312-mru.tar.gz

cd ..
rmdir temp
