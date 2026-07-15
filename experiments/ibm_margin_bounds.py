import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pennylane.templates import IQPEmbedding 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler

#Uncomment to run FakeFez simulations
#from qiskit_ibm_runtime.fake_provider import FakeFez
#from qiskit_aer import AerSimulator
#from qiskit_aer.noise import NoiseModel
#from qiskit_aer.primitives import SamplerV2

from ibm_bounds_definitions import create_iqp_feature_map, create_training_overlap_circuit_list, compute_overlap_matrix
from src.dataset_config import define_BC_dataset, define_gaussian_dataset
from src.kernel_definitions import clean_rho_fn, get_clean_matrix, local_rho_fn, get_local_matrix
from src.bounds_definitions import get_full_alpha, VI_bounds
from results.results import ideal_kernels_IBM, IBM_bound_results
from src.plotting_fns import ibm_plots

np.random.seed(42)

#get a subset of 20 samples from BC dataset
X_subset, y_subset, n, n_layers, _, _ = define_BC_dataset(start = 200, stop=220) #Example usage

#Split first then do preprocessing on train sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_subset, 
        y_subset, 
        test_size = 0.2, 
        random_state=42)  

X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, 
        y_train_val, 
        test_size = 0.2,   
        random_state=42) #altogether a 12/4/4 test/train/val sample split

#Initialise pipeline
preprocessor = make_pipeline(
        StandardScaler(),
        PCA(n_components=n, random_state = 42),
        MinMaxScaler(feature_range=(0,np.pi)) 
    )

#Fit on training data
preprocessor.fit(X_train)

#Transform X_train set
X_train = preprocessor.transform(X_train)

#Scale y_labels
y_train_scaled = 2*(y_train-0.5) #convert from [0,1] to [-1,1]

train_size = len(X_train)

y_train = np.array(y_train_scaled).ravel()
num_samples = len(X_train)

#Define initial properties
num_features=2
num_shots = 10000

#Create IQP-style feature map
fm = create_iqp_feature_map(n, num_features, reps=n_layers)

# Get a specific backend or the least busy
service = QiskitRuntimeService()
backend = service.backend("ibm_fez") 

# Running on a real hardware
sampler = Sampler(mode=backend)

"""
#Comment the above 3 lines of code and 
#Uncomment below code to run FakeFez simulations
backend = FakeFez()

noise_model = NoiseModel.from_backend(backend)

simulator = AerSimulator(noise_model = noise_model)
sampler = SamplerV2.from_backend(simulator)
"""

# Create the circuits for training and testing overlaps
training_overlap_circ_list = create_training_overlap_circuit_list(train_size, X_train, fm)

#Generate pass managers with optimisation level = 3 for this backend
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

#ISA = Instruction Set Architecture
isa_circuit_list = [pm.run(circuit) for circuit in training_overlap_circ_list]
job_training = sampler.run(isa_circuit_list, shots=num_shots) 

# Compute training matrix
results_training = job_training.result() #sampler returns meas outcome distributions
kernel_matrix = compute_overlap_matrix(num_shots, results_training, train_size, train_size, is_symmetric=True)
print("Training matrix done")

#symmetrise
kernel_hardware = np.asarray(kernel_matrix)
kernel_hardware = 0.5*(kernel_matrix + kernel_matrix.T) 

#the same clean kernel will be used for the fake and real experiments
#Compute clean kernel for this training set
clean_rho = clean_rho_fn(n=n, n_layers=n_layers, embedding=IQPEmbedding)
clean_K = get_clean_matrix(A = X_train, B = X_train, fn_clean_rho = clean_rho)
clean_K = 0.5 * (clean_K + clean_K.T) #symmetrise

#apply stability bound code

C = 1
tau = 0.1
m = len(y_train)

p_local_list = [0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.375]

#Regularise and Symmetrise the clean kernel
clean_K_reg = clean_K + tau*np.eye(m)
clean_K_reg = 0.5 * (clean_K_reg + clean_K_reg.T)
    

#Compute clean margin with regularised kernel
svm_clean_ideal = SVC(kernel = "precomputed", C=C).fit(clean_K_reg,y_train)

"""Get IBM hardware result"""

#regularise hardware kernel
hardware_K_reg = kernel_hardware + tau*np.eye(m)

svm_hardware = SVC(kernel='precomputed',C=C).fit(hardware_K_reg, y_train)

hardware_result = VI_bounds(K_clean = clean_K, K_noisy=kernel_hardware, svm_clean=svm_clean_ideal, svm_noisy = svm_hardware, tau=tau, y = y_train) 
#Use unregularised kernels in bound calc since regularisation terms cancel in derivation

print(hardware_result)
    
"""Simulated values for comparison"""
results = [] 

#Computing bounds for all p in list
for p_local in p_local_list:
            
    local_rho = local_rho_fn(p_local, n, n_layers, IQPEmbedding)
    noisy_K = get_local_matrix(A=X_train, B=X_train, fn_local_rho = local_rho)
    
    #Symmetrise and regularise noisy kernel
    noisy_K = 0.5 * (noisy_K + noisy_K.T)
    noisy_K_reg = noisy_K + tau*np.eye(m)
    noisy_K_reg = 0.5 * (noisy_K_reg + noisy_K_reg.T)
        
    svm_noisy = SVC(kernel = "precomputed", C=C).fit(noisy_K_reg, y_train)
        
    result = VI_bounds(clean_K, noisy_K, svm_clean_ideal, svm_noisy, tau, y_train)
    #train with reg kernel -> bound just uses K

    #Save result dict to list
    result["p"] = p_local
    results.append(result)
    print(result)


#Save results to text file
filename = "IBM_BC_Bounds.txt"

with open(filename, 'a') as file:
    file.write(f"Breast Cancer Dataset\n")
    file.write(f"Stability Bound Results: \n")
    file.write(f"m = {num_samples}\n")
    for i in range(len(results)):
        for k in results[i].keys():
            file.write(f"{k}: {results[i][k]}\n")

with open(filename, 'r') as file: #read
    content = file.read()
    print(content)

        
    

"""
#Uncomment the following code to get the pre-computed clean kernel matrices 

clean_K_bc220, clean_K_bc45, clean_K_gaus = ideal_kernels_IBM()
"""
########################################################################
"""
#Uncomment the following code to reproduce the figures from the manuscript

p_local_list,q_diffs_bc220,B_VIs_bc220,q_diffs_gaus, B_VIs_gaus, q_diffs_bc45, B_VIs_bc45 = IBM_bound_results()
ibm_plots(p_local_list,q_diffs_bc220,B_VIs_bc220,q_diffs_gaus, B_VIs_gaus, q_diffs_bc45, B_VIs_bc45)

"""
