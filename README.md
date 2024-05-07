# RNN Generalization Project (P1)

This repository contains code for the following paper:

[Exploration-exploitation mechanisms in recurrent neural networks and human learners in restless bandit problems](https://doi.org/10.1101/2023.04.27.538570)

## Install Requirements

### Python

Install [Anaconda](https://www.anaconda.com/download), then open Anaconda Prompt and navigate to this repository. 

Then recreate the environment of this project by typing the following in Anaconda Prompt:

```bash
conda env create -f environment.yml
```

You should now have a virtual environment called ``RNNExplore`` with all the necessary dependancies regarding Python. 

### R

Install [R](https://www.r-project.org/) (This work is based on version 4.1.1) and [R Studio](https://posit.co/download/rstudio-desktop/) (This work is based on version 2021.09.1)

Then double click on ``p1_generalization.Rproj``, this will open R Studio, then in the console type the following to install all R dependancies: 

```r
renv::restore()
```
Alternatively, you can load the necessary packages from an .RData file and install them manually. 

```R
load("r_packages.RData")
install.packages(r_pckgs)
```


## Directory Structure

```
p1_generalization
│   README.md
│   main.py
│   helpers.py
|   .Rprofile
|   .gitignore
|   environment.yaml
|   helpers.py
|   p1_generalization.RData
|   r_packages.RData
|   renv.lock  
│
└───classes
│   │
│   └───bandits
│   │   
│   └───neural_networks
|       |
|       └───rnns
│   
└───cognitive_models
│
└───data
│   │   
│   └───intermediate_data
│   │   │   
│   │   └───fixed_bandits
│   │   │ 
│   │   └───modeling
│   │   │   │ 
│   │   │   └───modeling_fits
│   │   │   │    
│   │   │   └───preprocessed_data_for_modeling
│   │   │
│   │   └───jasp_analysis
│   │      
│   └───rnn_raw_data
│   │      
│   └───human_raw_data
│
└───doc
│
└───notebooks
│
└───plots
│
└───renv
│
└───saved_models
│
└───scripts
│
└───tensorboard
```

## Content
The repository contains:

* ``main.py``: This is the main script, which can be used for training, testing and simulating the RNNs.
* ``helpers.py``: contains classes/functions to handle .zip, .feather files and other helper functions
* ``classes``: contains classes for creating bandit tasks and RNNs
* ``cognitive_models``: contains bayesian cognitive models written in STAN
* ``data``: contains RNN and human behavioural data, pregenerated bandits to test the RNNs, data to conduct bayesian analysis with JASP, preprocessed data for cognitive modeling and posterior model fits.
* ``saved_models``: contains trained RNN weights and biases
* ``scripts``: contains scripts used in this project (see [scripts_explanations.md](../main/doc/scripts_explanations.md))
* ``notebooks``: contains jupyter notebooks used in this project (see [notebooks_explanations.md](../main/doc/notebooks_explanations.md))
* ``tensorboard``: contains saved files to plot RNN training logs in tensorboard
* ``doc``: contains documentation and how-to guides for this project


