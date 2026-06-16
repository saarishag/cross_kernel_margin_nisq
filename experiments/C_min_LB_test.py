import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline

from src.kernel_definitions import clean_rho_fn, get_clean_matrix, local_rho_fn, get_local_matrix
from src.dataset_config import define_wine_dataset, define_heart_dataset, define_BC_dataset, define_gaussian_dataset
from src.bounds_definitions import calc_margin, calc_C_min_LB 

np.random.seed(42)

X_subset, y_subset, n, n_layers, embedding, p_local_list = define_heart_dataset() #example usage - Heart Disease Dataset

#Split first then do preprocessing on train sets
X_train, X_val, y_train, y_val = train_test_split(
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

#Transform both x sets
X_train = preprocessor.transform(X_train)
X_val = preprocessor.transform(X_val)
  

#Scale train and test y_labels
y_train_scaled = 2*(y_train-0.5)
y_val_scaled = 2*(y_val-0.5)

num_samples = len(X_train)

y_train = np.array(y_train_scaled).ravel()
y_val = np.array(y_val_scaled).ravel()
   
#Computing the best C for the noiseless (ideal) margin 
C_list = [0.1, 1, 10, 100, 1000]
best_score = 0 #default value
best_C = 1 #default value

clean_rho = clean_rho_fn(n=n, n_layers=n_layers, embedding=embedding)
clean_K = get_clean_matrix(A = X_train, B = X_train, fn_clean_rho = clean_rho)
clean_K_val = get_clean_matrix(A = X_val, B = X_val, fn_clean_rho = clean_rho)

for C_ideal in C_list:
    svm_clean = SVC(kernel = "precomputed", C=C_ideal)
        
    #Use 5-fold CV score to get the best C
    scores = cross_val_score(svm_clean, clean_K, y_train, cv = 5)
    avg_score = np.mean(scores)

    if best_score < avg_score:
        best_score = avg_score
        best_C = C_ideal
            
print(f"Best C from CV: C = {best_C}")

#Compute clean margin using best_C
svm_clean_ideal = SVC(kernel = "precomputed", C=best_C).fit(clean_K,y_train)
clean_margin = calc_margin(svm_clean_ideal, clean_K, y_train)

C_min_estimate = []
margin_estimate = []
noisy_margin_valid = []

#Computing the minimum C value allowed for all p_local on list
for p_local in p_local_list:
        
    #Compute noisy margin estimate using validation set
    local_rho = local_rho_fn(p_local, n, n_layers, embedding)

    noisy_K_val = get_local_matrix(A=X_val, B=X_val, fn_local_rho = local_rho)
    svm_noisy_val = SVC(kernel = "precomputed", C=best_C).fit(noisy_K_val, y_val)
    noisy_margin_val = calc_margin(svm_noisy_val,clean_K_val, y_val)

    #Compute actual noisy margin for comparison
    noisy_K_train = get_local_matrix(A=X_train, B=X_train, fn_local_rho = local_rho)
    svm_noisy = SVC(kernel = "precomputed", C=best_C).fit(noisy_K_train, y_train)
    noisy_margin = calc_margin(svm_noisy,clean_K, y_train) # Margin evaluated using noisy dual variables in clean HS

    print(f"Noisy Estimate = {noisy_margin_val}")
    print(f"Noisy Margin = {noisy_margin}")
        
    C_min_est = calc_C_min_LB(p_local, clean_margin, noisy_margin_val, n, n_layers)
    
    #Since C >= 0, set any negative C_min_est values to 0
    if C_min_est < 0: C_min_est = 0  

    C_min_estimate.append(C_min_est)
    margin_estimate.append(noisy_margin_val)
    noisy_margin_valid.append(noisy_margin)
    
for C_min_est in C_min_estimate:
    print(f"{C_min_est}")

print(f"Minimum C = {np.max(C_min_estimate)}")

#Save results to a text file
filename = "CminTest_Heart_LB.txt"
    
with open(filename, 'a') as file:
    file.write("---------------------------------------------------------------------------\n")
    file.write(f"m={num_samples}, n={n}, n_layers={n_layers}\n")
    file.write(f"Best C from CV: C = {best_C}\n")
    file.write(f"Clean Margin: {clean_margin} \n")
    file.write(f"p_local: Cmin_estimate \n")

    for p_local, C_min_est in zip(p_local_list, C_min_estimate):
        file.write(f"{p_local}: {C_min_est}\n")
    
    file.write(f"Estimated Cmin value: {np.max(C_min_estimate)}\n")

    file.write(f"p_local, estimate, margin: \n")
    for p_local, estimate, margin in zip(p_local_list, margin_estimate,noisy_margin_valid):
        file.write(f"{p_local}: {estimate}, {margin} \n")


with open(filename, 'r') as file:
    content = file.read()
    print(content)

