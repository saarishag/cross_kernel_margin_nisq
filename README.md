# Cross-Kernel Margins: A Robustness Measure for Quantum Kernel Methods under Local Depolarising Noise
This repository holds the code for the three independent but connected numerical experiments presented in the paper entitled "Cross-Kernel Margins: A Robustness Measure for Quantum Kernel Methods under Local Depolarising Noise", which can be found at [arXiv.org](https://arxiv.org/pdf/2601.23084).
## Description - Folder Structure
### 1. _experiments_ Folder
`margin_generalisation_link.py` - This script contains code which explores the relationship between margins and generalisation by corrupting the labels of the training data in an ideal (noiseless) setting. 
*(This script can be used to reproduce Figures 4,5 and 6 of the paper.)*

`local_vs_global.py` - This script explores the difference in test accuracy achieved on a dataset when using the local depolarising noise model compared to the global noise model. This comparison is made possible by first matching the survival probabilities of both models. 
*(This script can be used to reproduce Figure 10 of the paper.)*

`margin_bounds.py` - This script verifies the upper and lower margin bounds derived in the paper. It compares the theoretical bound value of the margin affected by (local depolarising) noise with the actual margin value. 
*(This script can be used to reproduce Figures 11 and 12 of the paper.)* 

`C_region_test.py` - This script determines a feasible range for the acceptable values for the regularisation parameter that must be used in `margin_bounds.py` for valid upper bounds.

`ibm_margin_bounds.py` - This script verifies the margin bounds on real quantum hardware.
*(This script can be used to reproduce Figure 13 of the paper.)*

### 2. _src_ Folder
`dataset_config.py`- This script contains code to fetch and preprocess the datasets used in the paper, and prepare them for use for the binary classification problem.

`kernel_definitions.py`- This script contains all functions necessary to obtain the kernel matrix using a quantum circuit with PennyLane when using the noiseless (ideal), local and global depolarising noise settings.

`bounds_definitions.py`- This script entails the functions necessary to compute the upper and lower bounds of the respective noisy margin value.

`gen_margin_definitions.py`-This script defines the necessary functions in order to run `margin_generalisation_link.py`. 

`ibm_bounds_definitions.py`-This script details the functions required to create and run the quantum circuits on the real quantum hardware using Qiskit. 

`plotting_fns.py` - This script contains plotting functions which can be used in conjunction with `results.py` to reproduce the plots from the paper.

### 3. _results_ Folder
`results.py` - This script contains the actual numeric results which can be used with `plotting_fns.py` to reproduce the exact plots from the paper.

### 4. _data_ Folder
This folder contains the `.csv` files necessary to reproduce the figure from the paper depicting boxplots of the median geometric margin with increasingly higher fractions of corrupted training labels.

### 5. _hardware\_metadata_ Folder
`ibm_fez_calibrations_2026-01-30T12_25_55Z.csv` - This file contains the calibration data for all qubits associated with `ibm_fez` at the date and time of access.

`ibm_job_metadata.json` - This file includes metadata, such as the job_id, associated with the experiment conducted on real hardware for reproducibility.

## Dependencies 
The following dependencies must be installed to run these scripts:
- Python 3.11+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - qiskit
  - PennyLane
  - scipy
- An IBM Quantum Account must be created to run the `ibm_*` scripts

## Running the Code
The three independent simulations with their related code scripts include:
- exploring the link between margins and generalisation using corrupted labels
  - `gen_margin_definitions.py`
  - `margin_generalisation_link.py` 
- investigating the difference between local and global depolarising noise using test accuracy
  - `kernel_definitions.py`
  - `local_vs_global.py`
- verifying the upper and lower margin bounds
  - `kernel_definitions.py`, `bounds_definitions.py`, `ibm_bounds_definitions.py`
  - `C_region_test.py`, `margin_bounds.py`, `ibm_margin_bounds.py`
### Steps:
1. Clone this repository
2. Install the dependencies
3. From the _root directory of the repository_, run the script for the desired numerical experiment from the _experiments_ folder as a Python module
   - e.g. python -m experiments.local_vs_global
   - Note: The scripts in the _src_ and _results_ folders are supporting scripts and must not be run alone
  
## Acknowledgements
This research was supported by the National Research Foundation (NRF) of South Africa. The authors would like to acknowledge the National Institute for Theoretical and Computational Sciences (NITheCS). We thank the University of KwaZulu-Natal (UKZN) for the use of the HEP1 machine. We acknowledge the South African Quantum Technology Initiative (SA QuTI) and IBM Quantum services for this work. The views expressed are those of the authors, and do not reflect the official policy or position of IBM or the IBM Quantum team.

