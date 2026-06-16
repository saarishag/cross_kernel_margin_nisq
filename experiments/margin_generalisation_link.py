import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn import metrics
import pandas as pd

from src.kernel_definitions import clean_rho_fn, get_clean_matrix
from src.dataset_config import define_wine_dataset, define_heart_dataset, define_BC_dataset, define_gaussian_dataset
from results.results import boxplot_results, acc_margin_results
from src.plotting_fns import plot_boxplots, dual_plot, acc_margin_plot
from src.gen_margin_definitions import per_sample_cross_margin, corrupt_labels

np.random.seed(42)

X_subset, y_subset, n, n_layers, embedding, _ = define_heart_dataset() #example usage - Heart Disease Dataset

#Initialise pipeline
preprocessor = make_pipeline(
        StandardScaler(),
        PCA(n_components=n, random_state = 42),
        MinMaxScaler(feature_range=(0,np.pi)) 
    )

C0 = 1 #Use default value

corruption_levels = [0.0, 0.10, 0.25, 0.5, 0.6, 0.75, 1.0]

cross_margins = []
all_corruption_levels = []
mean_accuracies = []
std_accuracies = []

for corrupt_lvl in corruption_levels:
    print(f"Testing corruption level: {corrupt_lvl}")
    accuracies = []
    
    #Use 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_subset)):
        print(f"Fold: {fold}")
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y_subset[train_idx], y_subset[test_idx]
            
        #Fit on training data
        preprocessor.fit(X_train)

        #Transform both x sets
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

        #Scale train and test y_labels
        y_train_scaled = 2*(y_train-0.5)
        y_test_scaled = 2*(y_test-0.5) #convert from [0,1] to [-1,1]

        num_samples = len(X_train)

        y_train = np.array(y_train_scaled).ravel()
        y_test = np.array(y_test_scaled).ravel()

        #Get ideal kernels 
        clean_rho = clean_rho_fn(n=n, n_layers=n_layers, embedding=embedding)
        K_train = get_clean_matrix(A = X_train, B = X_train, fn_clean_rho = clean_rho)
        K_test = get_clean_matrix(A = X_test, B = X_train, fn_clean_rho = clean_rho)
            
        #corrupt a fraction of the training labels 
        y_corrupted = corrupt_labels(y_train, corrupt_lvl)

        #Train SVM with corrupted labels 
        svm = SVC(kernel="precomputed", C=C0).fit(K_train, y_corrupted)
            
        #Calculate and store (per-sample) cross-label margins
        cross_margins = per_sample_cross_margin(svm, K_train, y_train, y_corrupted)
            
        for cross in zip(cross_margins):
            cross_margins.append(cross)
            all_corruption_levels.append(corrupt_lvl)

        #Calculate test accuracy
        y_pred = svm.predict(K_test)
        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
   
    print(f"Accuracy: {np.mean(accuracies)} +/- {np.std(accuracies)}")
    mean_accuracies.append(np.mean(accuracies))
    std_accuracies.append(np.std(accuracies))

#build the dataframe and save margin results to csv file
df = pd.DataFrame(
    {
        'corruption_levels': all_corruption_levels,
        'cross_margin': cross_margins
    }
) 

df['corruption_levels'] = df['corruption_levels'].astype(str)
df.to_csv("CrossKernelMargins_Heart.csv", index=False)


#Save accuracy results to a text file
filename = "CrossK_CorruptedLabelsAcc_Heart.txt"
    
with open(filename, 'a') as file:
    for corr_level, mean_acc, std_acc in zip(corruption_levels, mean_accuracies, std_accuracies):
        file.write(f"Corruption Level = {corr_level}\n")
        file.write(f"Test Accuracy = {mean_acc} +/- {std_acc} \n")
  
with open(filename, 'r') as file:
    content = file.read()
    print(content)

"""
#Uncomment to duplicate plots from the paper

htru2_df, wine_df, heart_df, gaus_df = boxplot_results() #margins for 3 corruption levels only
plot_boxplots(htru2_df, wine_df, heart_df, gaus_df)

htru2_df, wine_df, heart_df, gaus_df = dual_plot_results() #margins for all corruption levels
htru2_acc, htru2_std, htru2_med_margins, wine_acc, wine_std, wine_med_margins, heart_acc, heart_std, heart_med_margins, gaus_acc, gaus_std, gaus_med_margins = acc_margin_results(htru2_df, wine_df, heart_df, gaus_df)
dual_plot(htru2_acc, htru2_std, htru2_med_margins, wine_acc, wine_std, wine_med_margins, heart_acc, heart_std, heart_med_margins, gaus_acc, gaus_std, gaus_med_margins)

acc_margin_plot(htru2_acc, htru2_med_margins, wine_acc, wine_med_margins, heart_acc, heart_med_margins, gaus_acc, gaus_med_margins)
"""