#!/bin/bash

AUX_URL=https://www.cs.rice.edu/~vo9/fuwen/text2scene/aux_data.zip

echo "Downloading auxiliary data for composite image generation ..."
wget $AUX_URL -O aux_data.zip
echo "Unzipping..."
unzip -q aux_data.zip -d data/

rm aux_data.zip
