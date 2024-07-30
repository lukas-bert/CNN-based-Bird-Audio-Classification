# CNN-based-Bird-Audio-Classification
 A machine learning project for the ML Seminar 2024 at the TU Dortmund (faculty of physics). 
 
 A convolutional neural network is employed to classify audio recordings of calls from different bird species.
 Dataset based on data uploaded on https://xeno-canto.org/ 

 Installation:
 To install all necessary packages run:
    conda env create -f environment.yaml
 for CPU installation or
    conda env create -f env_gpu.yaml
 for GPU installation (also requires a manual installation of CUDA 12.1 and cuDNN 8.9.7).
 
 Some notebooks are outdated and will not run in this environment.

 The "core" of the project can be executed by running snakemake all --cores <n_cores> in the workflow folder.
 The notebooks "knn.ipynb" and "Results.ipynb" can be used to reproduce the plots created in this analysis, without running the snakemake command.

 Remarks:
 When training the model, the training can crash due to out of memory (oom) errors. 
 I had to install tcmalloc instead of the standard linux memory allocator (malloc) in order to prevent crashes.

 The downloading of the dataset can take up to a few hours.
 The computation of the spectrograms is highly inefficent and not multi-core friendly and so takes several hours.
 The traing of the CNN took 15 hours on a single GPU.
 
 Authors: Lukas Bertsch
