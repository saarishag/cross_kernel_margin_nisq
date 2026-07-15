import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pennylane.templates import IQPEmbedding 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.kernel_definitions import clean_rho_fn, get_clean_matrix, local_rho_fn, get_local_matrix
from src.dataset_config import define_wine_dataset, define_heart_dataset, define_BC_dataset, define_gaussian_dataset
from src.bounds_definitions import get_full_alpha, VI_bounds 
from results.results import perturbation_results
from src.plotting_fns import plot_perturbation_bound
np.random.seed(42)

X_subset, y_subset, n, n_layers, embedding, p_local_list = define_heart_dataset() #Example usage - Heart Disease Dataset

#Split first then do preprocessing on train sets
X_train, X_test, y_train, y_test = train_test_split(
        X_subset, 
        y_subset, 
        test_size = 0.25, 
        random_state=42) 

#Initialise pipeline
preprocessor = make_pipeline(
        StandardScaler(),
        PCA(n_components=n, random_state = 42),
        MinMaxScaler(feature_range=(0,np.pi)) 
    )

#Fit on training data
preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)
y_train_scaled = 2*(y_train-0.5)

num_samples = len(X_train)

y_train = np.array(y_train_scaled).ravel()

#Get the ideal kernel matrix
clean_rho = clean_rho_fn(n=n, n_layers=n_layers, embedding=embedding)
clean_K = get_clean_matrix(A = X_train, B = X_train, fn_clean_rho = clean_rho)
clean_K = 0.5 * (clean_K + clean_K.T) #symmetrise
    
"""
Perturbation Bounds
"""

C = 1
tau = 0.1
m = len(y_train)

#Regularise and Symmetrise Clean Kernel
clean_K_reg = clean_K + tau*np.eye(m)
clean_K_reg = 0.5 * (clean_K_reg + clean_K_reg.T)
        
#Compute clean margin 
svm_clean_ideal = SVC(kernel = "precomputed", C=C).fit(clean_K_reg,y_train)
#train with reg kernel

results = [] 

#Computing bounds for all p in list
for p_local in p_local_list:
            
    local_rho = local_rho_fn(p_local, n, n_layers, IQPEmbedding)
    noisy_K = get_local_matrix(A=X_train, B=X_train, fn_local_rho = local_rho)
    
    #Regularise and symmetrise noisy kernel    
    noisy_K = 0.5 * (noisy_K + noisy_K.T)
    noisy_K_reg = noisy_K + tau*np.eye(m)
    noisy_K_reg = 0.5 * (noisy_K_reg + noisy_K_reg.T)
        
    svm_noisy = SVC(kernel = "precomputed", C=C).fit(noisy_K_reg, y_train)
        
    result = VI_bounds(clean_K, noisy_K, svm_clean_ideal, svm_noisy, tau, y_train)
    #train with reg kernel -> bound just uses K

    #Save result dict to list
    result["p"] = p_local
    results.append(result)

#Save results to text file
filename = "Heart_PertBounds.txt"

with open(filename, 'a') as file:
    file.write(f"Perturbation Bound Results: \n")
    file.write(f"m = {num_samples}\n")
    for i in range(len(results)):
        for k in results[i].keys():
            file.write(f"{k}: {results[i][k]}\n")


with open(filename, 'r') as file: #read
    content = file.read()
    print(content)

"""
#Uncomment to duplicate plots from the paper 
 
p_local_list,q_diffs_heart,B_VIs_heart,q_diffs_gaus, B_VIs_gaus, q_diffs_bc, B_VIs_bc, q_diffs_wineL1, B_VIs_wineL1, q_diffs_wineL2, B_VIs_wineL2 = perturbation_results()
plot_perturbation_bound(p_local_list,q_diffs_heart,B_VIs_heart,q_diffs_gaus, B_VIs_gaus, q_diffs_bc, B_VIs_bc, q_diffs_wineL1, B_VIs_wineL1, q_diffs_wineL2, B_VIs_wineL2)

"""
