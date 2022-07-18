# CIA
Code for ECCV2022 paper 'Hierarchical Feature Embedding for Visual Tracking', based on **PyTorch**.

### [Tracking Libraries](pytracking)

Libraries for implementing and evaluating visual trackers. It includes

* All common **tracking** and **video object segmentation** datasets.  
* Scripts to **analyse** tracker performance and obtain standard performance scores.
* General building blocks, including **deep networks**, **optimization**, **feature extraction** and utilities for **correlation filter** tracking.  

### [Training Framework: LTR](ltr)
 
**LTR** (Learning Tracking Representations) is a general framework for training your visual tracking networks. It is equipped with

* All common **training datasets** for visual object tracking and segmentation.  
* Functions for data **sampling**, **processing** etc.  
* Network **modules** for visual tracking.
* And much more...


## Trained models


## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/zxgravity/CIA.git
```
   
#### Clone the submodules.  
In the repository directory, run the commands:  
```bash
git submodule update --init  
```  
#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path pytracking
```  
This script will also download the default networks and set-up the environment.  

**Note:** The install script has been tested on an Ubuntu 16.04 system. In case of issues, check the [detailed installation instructions](INSTALL.md). 

#### Let's test it!
Activate the conda environment and run the script pytracking/run_tracker.py to run CIA18.  
```bash
conda activate pytracking
cd pytracking
python run_tracker.py CIA CIA18   
```  

## Acknowledgments
* Thanks for the project [pytracking](https://github.com/visionml/pytracking)
* Thanks for the great [PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling) module.  
