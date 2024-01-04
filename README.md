# STGIC-project
The directory "STGIC" contains all the scripts which are organized as a Python package. The AGC algorithm is implemented by the script AGC.py, the dilated convolution framework is implemented by the script DCF.py, and data preprocessing is implemented by preprocess.py.
The directory "test" contains all the scripts to test the clustering performance of STGIC. DLPFC.py is for testing with the bechmark of 10x Visium human DLPFC dataset consisting of 12 samples. human_bc.py is for testing the 10x Visium breast cancer dataset. mouse_postbrain.py is for testing the 10x Visium mouse posterior brain dataset. mouse_olfactory.py is for testing the Stereo-seq mouse olfactory bulb dataset. 
mybin.py is for binning the data downloaded from https://github.com/JinmiaoChenLab/SEDR_analyses.
sparkx.r is for detecting spatially variable genes with SPARK-X.
All experiments are executed on a Centos7.9.2009 server equipped with an NVIDIA A100 GPU (NVIDIA-SMI 530.30.02).
