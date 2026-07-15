# The Cross-Kernel Margin: A Robustness Measure for Quantum Kernel Methods 
This repository holds the code for the three connected numerical experiments presented in the paper entitled "The Cross-Kernel Margin: A Robustness Measure for Quantum Kernel Methods", which can be found at [arXiv.org](https://arxiv.org/pdf/2601.23084)

(previously released on arXiv as
"Margin-Based Generalisation Bounds for Quantum Kernel Methods under Local Depolarising Noise").

## Description - Folder Structure
### 1. _experiments_ Folder
`margin_generalisation_link.py` - This script contains code which explores the relationship between margins and generalisation by corrupting the labels of the training data in an ideal (noiseless) setting. 
*(This script can be used to reproduce Figures 4,5 and 6 of the paper.)*

`local_vs_global.py` - This script explores the test accuracy achieved on a dataset when using the local depolarising noise model compared to the global noise model. This comparison is made possible by first matching the survival probabilities of both models. 
*(This script can be used to reproduce Figure 10 of the paper.)*

`margin_bounds.py` - This script verifies the stability bounds derived in the paper. It compares the theoretical bound value of the deviation between the inverse squared cross-kernel margin and the corresponding inverse squared ideal margin quantity with the empirical value affected by (local depolarising) noise across multiple datasets.
*(This script can be used to reproduce Figure 11 of the paper.)* 

`ibm_margin_bounds.py` - This script evaluates the stability-bound quantities using kernel matrices obtained from the IBM quantum device and the FakeFez simulator and reproduces the corresponding numerical simulations.
*(This script can be used to reproduce Figure 12 and Table 1 of the paper.)*

### 2. _src_ Folder
`dataset_config.py`- This script contains code to fetch and preprocess the datasets used in the paper, and prepare them for use for the binary classification problem.

`kernel_definitions.py`- This script contains all functions necessary to obtain the kernel matrix using a quantum circuit with PennyLane when using the noiseless (ideal), local and global depolarising noise settings.

`bounds_definitions.py`- This script entails the functions necessary to compute the stability bounds using the ideal and cross-kernel inverse squared-margins.

`gen_margin_definitions.py`-This script defines the necessary functions in order to run `margin_generalisation_link.py`. 

`ibm_bounds_definitions.py`-This script details the functions required to create and run the quantum circuits on the real quantum hardware using Qiskit. 

`plotting_fns.py` - This script contains plotting functions which can be used in conjunction with `results.py` to reproduce the plots from the paper.

### 3. _results_ Folder
`results.py` - This script contains the actual numeric results which can be used with `plotting_fns.py` to reproduce the exact plots from the paper.

### 4. _data_ Folder
This folder contains the `.csv` files necessary to reproduce the figure from the paper depicting boxplots of the cross-label margin distribution for different fractions of corrupted training labels.

### 5. _hardware\_metadata_ Folder
`ibm_fez_calibrations_2026-01-30T12_25_55Z.csv` - This file contains the calibration data for all qubits associated with `ibm_fez` at the date and time of access for the hardware experiment associated with the 20 sample subset of the breast cancer dataset.

`ibm_fez_calibrations_2026-05-12T12_45_44Z.csv` - This file contains the calibration data for all qubits associated with `ibm_fez` at the date and time of access for the hardware experiments associated with the second subset of the breast cancer dataset and the subset of the gaussian dataset.

`ibm_BC220_job_metadata.json` - This file includes metadata, such as the job_id, associated with the experiment conducted with the 20 sample subset of the breast cancer dataset on real hardware for reproducibility.

`ibm_BC45_job_metadata.json` - This file includes metadata, such as the job_id, associated with the experiment conducted with the 45 sample subset of the breast cancer dataset on real hardware for reproducibility.

`ibm_Gaus_job_metadata.json` - This file includes metadata, such as the job_id, associated with the experiment conducted with the subset of the gaussian dataset on real hardware for reproducibility.

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
  - qiskit-aer
  - qiskit-ibm-runtime
  - PennyLane
  - scipy
- An IBM Quantum Account must be created to run the `ibm_*` scripts

## Running the Code
The three connected simulations with their related code scripts include:
- exploring the link between margins and generalisation using corrupted labels
  - `gen_margin_definitions.py`
  - `margin_generalisation_link.py` 
- comparing local and global depolarising noise using test accuracy
  - `kernel_definitions.py`
  - `local_vs_global.py`
- verifying the stability bounds
  - `kernel_definitions.py`, `bounds_definitions.py`, `ibm_bounds_definitions.py`
  - `margin_bounds.py`, `ibm_margin_bounds.py`
### Steps:
1. Clone this repository
2. Install the dependencies
3. From the _root directory of the repository_, run the script for the desired numerical experiment from the _experiments_ folder as a Python module
   - e.g. python -m experiments.local_vs_global
   - Note: The scripts in the _src_ and _results_ folders are supporting scripts and must not be run alone
  
## Acknowledgements
This research was supported by the National Research Foundation (NRF) of South Africa. The authors would like to acknowledge the National Institute for Theoretical and Computational Sciences (NITheCS). We thank the University of KwaZulu-Natal (UKZN) for the use of the HEP1 machine. We acknowledge the South African Quantum Technology Initiative (SA QuTI) and IBM Quantum services for this work. The views expressed are those of the authors, and do not reflect the official policy or position of IBM or the IBM Quantum team.

