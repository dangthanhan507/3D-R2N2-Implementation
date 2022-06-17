#!bin/bash

#activate conda
source ~/.bashrc 

#create environment
conda env create -f environment.yaml
source ~/.bashrc

conda activate 3dr

# install the CUDA version of torch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

#install good version of pillow so that import problem goes away
python3 -m pip install pillow==6.1

conda install vim -y