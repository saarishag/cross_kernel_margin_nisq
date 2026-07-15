import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

def plot_local_global(heart_data, wine_2N1L, wine_3N1L,n, n_layers):
    """
    Reproduces plots from the paper depicting the test accuracy 
    when using global vs local depolarising noise models for three datasets
    """
    
    fig, (ax1,ax2,ax3)= plt.subplots(1,3, figsize=(30,10), sharex=True)

    
    #Wine Dataset (2 qubits, 1 layer)
    data = wine_2N1L
    n=2
    n_layers = 1
    p_local = np.array([d[0] for d in data])
    p_global = 1 - (1-p_local)**(n*n_layers) #formula used to equate survival probability of both models

    local_acc = np.array([d[1] for d in data])
    local_std = np.array([d[2] for d in data])
    global_acc = np.array([d[3] for d in data])
    global_std = np.array([d[4] for d in data])

    p_glob_idx = [0,1,2,4,6,8,10,11] #display selective values for p_global for visual purposes
#Wine Dataset (N=2,L=1)
    ax1.axhline(y=0, color = "gray", linestyle="--", linewidth=1.5, alpha=0.3)
    
    ax1.plot(p_local, local_acc,'o-', label="Local Acc", color = 'blue', markersize = 8, linewidth = 2.5)
    
    ax1.fill_between(
        p_local, 
        local_acc-local_std,
        local_acc+local_std,
        alpha=0.05, 
        color = 'blue'
    )
    
    ax1.plot(p_local, global_acc,'o-', label="Global Acc", color = 'red', markersize = 8, linewidth = 2.5)
    ax1.fill_between(
        p_local, 
        global_acc-global_std,
        global_acc+global_std,
        alpha=0.05,
        color = 'red'
    )

    ax1.set_ylabel("Accuracy", fontsize=22)
    ax1.grid(True)
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(p_local[p_glob_idx])
    ax1_top.set_xticklabels([f"{p_global[i]:.2f}" for i in p_glob_idx])

    #Heart Disease Dataset
    data = heart_data
    p_local = np.array([d[0] for d in data])

    p_global = 1 - (1-p_local)**(n*n_layers) #to match survival prob
    local_acc = np.array([d[1] for d in data])
    local_std = np.array([d[2] for d in data])
    global_acc = np.array([d[3] for d in data])
    global_std = np.array([d[4] for d in data])

    p_glob_idx = [0,2,3,4,5,6,7,9]

#Heart Disease Dataset
    n=2
    n_layers = 1
    ax2.axhline(y=0, color = "gray", linestyle="--", linewidth=1.5, alpha=0.3)
    ax2.plot(p_local, local_acc,'o-', label="Local Acc", color = 'blue', markersize = 8, linewidth = 2.5)
    
    ax2.fill_between(
        p_local, 
        local_acc-local_std,
        local_acc+local_std,
        alpha=0.05, 
        color = 'blue'
    )
    
    ax2.plot(p_local, global_acc,'o-', label="Global Acc", color = 'red', markersize = 8, linewidth = 2.5)
    ax2.fill_between(
        p_local, 
        global_acc-global_std,
        global_acc+global_std,
        alpha=0.05,
        color = 'red'
    )
    ax2.set_xlabel('Local Depolarising Noise Probability', fontsize=22)
    ax2.grid(True)
    ax2_top = ax2.twiny() #dual axis
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(p_local[p_glob_idx])
    ax2_top.set_xticklabels([f"{p_global[i]:.2f}" for i in p_glob_idx])
    ax2_top.set_xlabel("Equivalent Global Depolarising Noise Probability", fontsize=22)

    #Wine Dataset (3 qubits, 1 layer)
    n=3
    n_layers = 1
    data = wine_3N1L
    p_local = np.array([d[0] for d in data])

    n=3
    n_layers = 1
    p_global = 1 - (1-p_local)**(n*n_layers) #to match survival prob
    local_acc = np.array([d[1] for d in data])
    local_std = np.array([d[2] for d in data])
    global_acc = np.array([d[3] for d in data])
    global_std = np.array([d[4] for d in data])

    p_glob_idx = [0,1,2,4,6,8, 10, 11] #display selective values for p_global for visual purposes

    ax3.axhline(y=0, color = "gray", linestyle="--", linewidth=1.5, alpha=0.3)
    
    ax3.plot(p_local, local_acc,'o-', label="Local Accuracy", color = 'blue', markersize = 8, linewidth = 2.5)
    
    ax3.fill_between(
        p_local, 
        local_acc-local_std,
        local_acc+local_std,
        alpha=0.05, 
        color = 'blue'
    )
    
    ax3.plot(p_local, global_acc,'o-', label="Global Accuracy", color = 'red', markersize = 8, linewidth = 2.5)
    ax3.fill_between(
        p_local, 
        global_acc-global_std,
        global_acc+global_std,
        alpha=0.05,
        color = 'red'
    )
    ax3.grid(True)
    ax3.legend(fontsize=18, loc='upper right')
    ax3_top = ax3.twiny()
    ax3_top.set_xlim(ax3.get_xlim())

    ax3_top.set_xticks(p_local[p_glob_idx])
    ax3_top.set_xticklabels([f"{p_global[i]:.2f}" for i in p_glob_idx])

    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)
    ax3.tick_params(labelsize=18)

    ax1_top.tick_params(labelsize=18)
    ax2_top.tick_params(labelsize=18)
    ax3_top.tick_params(labelsize=18)

    ax1.set_ylim((0.34,0.71))
    ax2.set_ylim((0.36,0.87))
    ax3.set_ylim((0.45,0.78))


    ax1.text(0.75, 0.97, 'Wine (N=2,L=1)', transform = ax1.transAxes,
                fontsize=18, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    ax2.text(0.75, 0.97, 'Heart Disease', transform = ax2.transAxes,
             fontsize=18, verticalalignment='top', 
             bbox=dict(boxstyle='round', alpha=0.5))
    ax3.text(0.74, 0.87, 'Wine (N=3, L=1)', transform = ax3.transAxes,
                fontsize=18, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))

    plt.tight_layout()
    plt.show()



def plot_stability_bound(p_local_list,q_diffs_heart,B_VIs_heart,q_diffs_gaus, B_VIs_gaus, q_diffs_bc, B_VIs_bc, q_diffs_wineL1, B_VIs_wineL1, q_diffs_wineL2, B_VIs_wineL2):
    """
    Reproduces plots from paper depicting the empirical difference between the cross-kernel and ideal inverse squared-margins
    with the corresponding theoretical bound values for visual comparison for various datasets
    All values are assumed to have been scaled by the corresponding ideal inverse squared-margin
    """
    fig, axes = plt.subplots(1,5, figsize=(15,4), sharex=True)
    
    axes[0].plot(p_local_list, q_diffs_heart, 's-', linewidth=2, markersize=6, label = 'Empirical', color = "blue")
    axes[0].plot(p_local_list, B_VIs_heart,'o--', linewidth=2, markersize=6, label = "Bound", color = "red")
    axes[0].set_ylabel("Relative Inverse Squared-Margin \n Deviation", fontsize=12)
    axes[0].grid(True, linestyle=":", linewidth = 0.5)
    axes[0].tick_params(labelsize=10)

    
    #plot maximum relative deviation
    rel_dev = B_VIs_heart - q_diffs_heart #already scaled
    max_dev_index = np.argmax(rel_dev)
    axes[0].vlines(p_local_list[max_dev_index], B_VIs_heart[max_dev_index], q_diffs_heart[max_dev_index] , linestyles = "dotted", label = "Max Relative Deviation")

    #Gaus
    axes[1].plot(p_local_list, q_diffs_gaus, 's-', linewidth=2, markersize=6, label = '', color = "blue")
    axes[1].plot(p_local_list, B_VIs_gaus,'o--', linewidth=2, markersize=6, label = "", color = "red")
    axes[1].grid(True, linestyle=":", linewidth = 0.5)
    axes[1].tick_params(labelsize=10)

    #plot relative deviation
    rel_dev = B_VIs_gaus - q_diffs_gaus #already scaled
    max_dev_index = np.argmax(rel_dev)
    axes[1].vlines(p_local_list[max_dev_index], B_VIs_gaus[max_dev_index], q_diffs_gaus[max_dev_index] , linestyles = "dotted")

    #BC
    axes[2].plot(p_local_list, q_diffs_bc, 's-', linewidth=2, markersize=6, label = '', color = "blue")
    axes[2].plot(p_local_list, B_VIs_bc,'o--', linewidth=2, markersize=6, label = "", color = "red")
    axes[2].grid(True, linestyle=":", linewidth = 0.5)
    axes[2].tick_params(labelsize=10)

    #plot relative deviation
    rel_dev = B_VIs_bc - q_diffs_bc #already scaled
    max_dev_index = np.argmax(rel_dev)
    axes[2].vlines(p_local_list[max_dev_index], B_VIs_bc[max_dev_index], q_diffs_bc[max_dev_index] , linestyles = "dotted")

    #wineL1
    axes[3].plot(p_local_list, q_diffs_wineL1, 's-', linewidth=2, markersize=6, label = '', color = "blue")
    axes[3].plot(p_local_list, B_VIs_wineL1,'o--', linewidth=2, markersize=6, label = "", color = "red")
    axes[3].grid(True, linestyle=":", linewidth = 0.5)
    axes[3].tick_params(labelsize=10)

    #plot relative deviation
    rel_dev = B_VIs_wineL1 - q_diffs_wineL1 #already scaled
    max_dev_index = np.argmax(rel_dev)
    axes[3].vlines(p_local_list[max_dev_index], B_VIs_wineL1[max_dev_index], q_diffs_wineL1[max_dev_index] , linestyles = "dotted", label = "Max Relative Deviation")

    #WineL2

    axes[4].plot(p_local_list, q_diffs_wineL2, 's-', linewidth=2, markersize=6, label = '', color = "blue")
    axes[4].plot(p_local_list, B_VIs_wineL2,'o--', linewidth=2, markersize=6, label = "", color = "red")
    axes[4].grid(True, linestyle=":", linewidth = 0.5)
    axes[4].tick_params(labelsize=10)

    #plot relative deviation
    rel_dev = B_VIs_wineL2 - q_diffs_wineL2 #already scaled
    max_dev_index = np.argmax(rel_dev)
    axes[4].vlines(p_local_list[max_dev_index], B_VIs_wineL2[max_dev_index], q_diffs_wineL2[max_dev_index] , linestyles = "dotted")


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor = (1, 0.1),
            ncol = 2, fontsize = 11, frameon=True)
    fig.text(0.5, 0.02, 'Local Depolarising Noise Probability', ha='center', fontsize = 13)

    axes[0].text(0.5, 0.05, 'Heart Disease', transform = axes[0].transAxes,
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[1].text(0.65, 0.05, 'Gaussian', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[2].text(0.5, 0.05, 'Breast Cancer', transform = axes[2].transAxes,
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[3].text(0.43, 0.05, 'Wine (N=2,L=1)', transform = axes[3].transAxes,
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[4].text(0.43, 0.05, 'Wine (N=2, L=2)', transform = axes[4].transAxes,
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    #Set vertical log scale
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')
    axes[3].set_yscale('log')
    axes[4].set_yscale('log')
     
    plt.show()

def plot_boxplots(htru2_df, wine_df, heart_df, gaus_df):
    """
    Reproduces plots from paper comparing boxplots highlighting
    the median (per-sample) cross kernel margin obtained for 
    three corruption levels for four datasets
    """
    fig,axes = plt.subplots(1,4, figsize=(14,6), sharex=True)

    #HTRU2 Dataset
    sns.boxplot(x = "corruption_levels", y ="cross_margin", data = htru2_df, ax = axes[0], palette= "viridis",linewidth = 1.5, showfliers=False )
    axes[0].set_xlabel("", fontsize = 14)
    axes[0].set_ylabel("Cross-Label Margin", fontsize = 14)
    axes[0].set_title("(a)", loc="left")
    axes[0].grid(True, linewidth=0.5, linestyle = ":")

    #Wine Dataset
    sns.boxplot(x = "corruption_levels", y ="cross_margin", data = wine_df, ax = axes[1], palette= "viridis",linewidth = 1.5, showfliers=False )
    axes[1].set_xlabel("", fontsize = 14, loc="right")
    axes[1].set_ylabel("", fontsize = 14)
    axes[1].set_title("(b)", loc="left")
    axes[1].grid(True, linewidth=0.5, linestyle = ":")

    #Heart Disease Dataset
    sns.boxplot(x = "corruption_levels", y ="cross_margin", data = heart_df, ax = axes[2], palette= "viridis",linewidth = 1.5, showfliers=False )
    axes[2].set_xlabel("", fontsize = 14)
    axes[2].set_ylabel("", fontsize = 14)
    axes[2].set_title("(c)", loc="left")
    axes[2].grid(True, linewidth=0.5, linestyle = ":")

    #Gaussian Dataset
    sns.boxplot(x = "corruption_levels", y ="cross_margin", data = gaus_df, ax = axes[3], palette= "viridis",linewidth = 1.5, showfliers=False )
    axes[3].set_xlabel("", fontsize = 14)
    axes[3].set_ylabel("", fontsize = 14)
    axes[3].set_title("(d)", loc="left")
    axes[3].grid(True, linewidth=0.5, linestyle = ":")

    axes[0].text(0.76, 0.986, 'HTRU2', transform = axes[0].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[1].text(0.65, 0.99, 'Wine Quality', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[2].text(0.8, 0.983, 'Heart', transform = axes[2].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[3].text(0.7, 0.983, 'Gaussian', transform = axes[3].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))

    fig.text(0.53, 0.005, 'Corruption Fraction', ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.show()

def dual_plot(htru2_acc, htru2_std, htru2_med_margins, wine_acc, wine_std, wine_med_margins, heart_acc, heart_std, heart_med_margins, gaus_acc, gaus_std, gaus_med_margins):
    """
    Reproduces plots from paper highlighting
    the decay in test accuracy and median cross kernel margin
    with increasing corruption levels for four datasets
    """
    fig,axes = plt.subplots(1,4, figsize=(14,6))
    corr_levels = [0,0.1,0.25,0.5,0.6,0.75,1.0]

    #Combined legend with twin axes
    ax2 = axes[0].twinx()

    line1 = axes[0].errorbar(corr_levels, htru2_acc, yerr=htru2_std, marker = 'o',markersize = 6, capsize = 3,color = 'blue', label="Accuracy")
    line2 = ax2.plot(corr_levels, htru2_med_margins, 'ro--', linewidth = 2, markersize = 6, label="Cross-Label Margin")

    axes[0].set_ylabel("Test Accuracy", fontsize = 14)
    axes[0].set_title("(a)", loc="left")

    #Combine legends
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    #Combined legend with twin axes
    ax2 = axes[1].twinx()

    line1 = axes[1].errorbar(corr_levels, wine_acc, yerr=wine_std, marker = 'o',markersize = 6, capsize = 3,color = 'blue', label="Accuracy")
    line2 = ax2.plot(corr_levels, wine_med_margins, 'ro--', linewidth = 2, markersize = 6, label="Cross-Label Margin")

    axes[1].set_title("(b)", loc="left")
    #Combine legends
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    #Combined legend with twin axes
    ax2 = axes[2].twinx()

    line1 = axes[2].errorbar(corr_levels, heart_acc, yerr=heart_std, marker = 'o',markersize = 6, capsize = 3,color = 'blue', label="Accuracy")
    line2 = ax2.plot(corr_levels, heart_med_margins, 'ro--', linewidth = 2, markersize = 6, label="Cross-Label Margin")

    axes[2].set_title("(c)", loc="left")

    #Combine legends
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    #Combined legend with twin axes
    ax2 = axes[3].twinx()

    line1 = axes[3].errorbar(corr_levels, gaus_acc, yerr=gaus_std, marker = 'o',markersize = 6, capsize = 3,color = 'blue', label="Accuracy")
    line2 = ax2.plot(corr_levels, gaus_med_margins, 'ro--', label="Margin")

    ax2.set_ylabel("Median Cross-Label Margin", fontsize = 14)
    axes[3].set_title("(d)", loc="left")

    #Combine legends
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()


    fig.text(0.5, 0.001, 'Corruption Fraction', ha='center', fontsize = 14)

    axes[0].text(0.08, 0.05, 'HTRU2', transform = axes[0].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[1].text(0.08, 0.05, 'Wine Quality', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[2].text(0.08, 0.05, 'Heart Disease', transform = axes[2].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))
    axes[3].text(0.08, 0.05, 'Gaussian', transform = axes[3].transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', alpha=0.5))

    axes[3].legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

def acc_margin_plot(htru2_acc, htru2_med_margins, wine_acc, wine_med_margins, heart_acc, heart_med_margins, gaus_acc, gaus_med_margins):
    """
    Reproduces plots from paper highlighting
    the linear correlation between test accuracy and 
    median cross-kernel margins using four datasets
    """
    fig,axes = plt.subplots(1,4, figsize=(14,6))
    axes[0].scatter(htru2_med_margins, htru2_acc ,color='steelblue', edgecolors = 'black', s=50, alpha=0.7, label="HTRU2 Dataset")

    r,p = pearsonr(htru2_med_margins, htru2_acc) #calculate the Pearson correlation coefficient

    #trend line
    z = np.polyfit(htru2_med_margins, htru2_acc, 1)
    trend_line  = np.poly1d(z)
    print(trend_line)

    axes[0].plot(htru2_med_margins, trend_line(htru2_med_margins), "r--", alpha=0.8, linewidth=2)
    axes[0].text(0.65, 0.06, f'r={r:.4f}', transform = axes[0].transAxes,
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor="wheat", alpha=0.5))

    axes[0].set_ylabel("Test Accuracy", fontsize = 14)
    axes[0].set_title("(a)", loc="left")
    axes[0].legend()


    axes[1].scatter(wine_med_margins, wine_acc, color='steelblue', edgecolors = 'black', s=50, alpha=0.7, label="Wine Dataset")

    r,p = pearsonr(wine_med_margins, wine_acc)

    #trend line
    z = np.polyfit(wine_med_margins, wine_acc, 1)
    trend_line  = np.poly1d(z)
    print(trend_line)
    axes[1].plot(wine_med_margins, trend_line(wine_med_margins), "r--", alpha=0.8, linewidth=2)
    axes[1].text(0.65, 0.06, f'r={r:.4f}', transform = axes[1].transAxes,
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor="wheat", alpha=0.5))

    axes[1].set_title("(b)", loc="left")
    axes[1].legend()

    axes[2].scatter(heart_med_margins, heart_acc, color='steelblue', edgecolors = 'black', s=50, alpha=0.7, label="Heart Disease Dataset")

    r,p = pearsonr(heart_med_margins, heart_acc)

    #trend line
    z = np.polyfit(heart_med_margins, heart_acc, 1)
    trend_line  = np.poly1d(z)
    print(trend_line)
    axes[2].plot(heart_med_margins, trend_line(heart_med_margins), "r--", alpha=0.8, linewidth=2)
    axes[2].text(0.65, 0.06, f'r={r:.4f}', transform = axes[2].transAxes,
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor="wheat", alpha=0.5))
    axes[2].set_title("(c)", loc="left")
    axes[2].legend()


    axes[3].scatter(gaus_med_margins, gaus_acc,color='steelblue', edgecolors = 'black', s=50, alpha=0.7, label="Gaussian Dataset")

    r,p = pearsonr(gaus_med_margins, gaus_acc)

    #trend line
    z = np.polyfit(gaus_med_margins, gaus_acc, 1)
    trend_line  = np.poly1d(z)
    print(trend_line)
    axes[3].plot(gaus_med_margins, trend_line(gaus_med_margins), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    axes[3].text(0.65, 0.06, f'r={r:.4f}', transform = axes[3].transAxes,
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor="wheat", alpha=0.5))

    fig.text(0.5, 0.001, 'Median Cross-Label Margin', ha='center', fontsize = 13)

    axes[3].set_title("(d)", loc="left")
    axes[3].legend()

    plt.tight_layout()
    plt.show()


def ibm_plots(p_local_list,q_diffs_bc220,B_VIs_bc220,q_diffs_gaus, B_VIs_gaus, q_diffs_bc45, B_VIs_bc45):
    """    
    Reproduces plots from paper depicting the empirical difference between the cross-kernel and ideal inverse squared-margins
    with the corresponding theoretical bound values for visual comparison for the datasets used for the IBM hardware experiments

    All values are assumed to have been scaled by the corresponding inverse squared ideal margin

    The first two values are excluded as the first and second entries are assumed to be the results from ibm_fez and FakeFez, respectively
    (These values are reported separately in a table)
    """

    fig, axes = plt.subplots(1,3, figsize=(12,4), sharex=False)
    print(axes.shape)

    #BC (200-220)
    axes[0].plot(p_local_list, q_diffs_bc220[2:], 's-', linewidth=2, markersize=6, color = "blue")
    axes[0].plot(p_local_list, B_VIs_bc220[2:],'o--', linewidth=2, markersize=6,  color = "red")
    
    axes[0].set_ylabel("Relative Inverse Squared-Margin \n Deviation", fontsize=12)
    axes[0].grid(True, linestyle=":", linewidth = 0.5)
    axes[0].tick_params(labelsize=10)

    axes[0].text(0.43, 0.03, 'Breast Cancer (200-220)', transform = axes[0].transAxes,
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', alpha=0.5))

    #plot maximum relative deviation
    rel_dev = B_VIs_bc220[2:] - q_diffs_bc220[2:] #already scaled
    max_dev_index = np.argmax(rel_dev)
    axes[0].vlines(p_local_list[max_dev_index], B_VIs_bc220[2:][max_dev_index], q_diffs_bc220[2:][max_dev_index] , linestyles = "dotted")


    #BC (0-45)
    axes[1].plot(p_local_list, q_diffs_bc45[2:], 's-', linewidth=2, markersize=6, label = 'Empirical', color = "blue")
    axes[1].plot(p_local_list, B_VIs_bc45[2:],'o--', linewidth=2, markersize=6, label = "Bound", color = "red")
    

    axes[1].grid(True, linestyle=":", linewidth = 0.5)
    axes[1].tick_params(labelsize=10)

    axes[1].text(0.5, 0.03, 'Breast Cancer (0-45)', transform = axes[1].transAxes,
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', alpha=0.5))

    rel_dev = B_VIs_bc45[2:] - q_diffs_bc45[2:] #already scaled
    max_dev_index = np.argmax(rel_dev)
    axes[1].vlines(p_local_list[max_dev_index], B_VIs_bc45[2:][max_dev_index], q_diffs_bc45[2:][max_dev_index] , linestyles = "dotted", label = "Max Relative Deviation")


    #Gaus  
    axes[2].plot(p_local_list, q_diffs_gaus[2:], 's-', linewidth=2, markersize=6, color = "blue")
    axes[2].plot(p_local_list, B_VIs_gaus[2:],'o--', linewidth=2, markersize=6,  color = "red")
    
    axes[2].grid(True, linestyle=":", linewidth = 0.5)
    axes[2].tick_params(labelsize=10)

    axes[2].text(0.77, 0.03, 'Gaussian', transform = axes[2].transAxes,
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', alpha=0.5))
    

    rel_dev = B_VIs_gaus[2:] - q_diffs_gaus[2:] #already scaled
    max_dev_index = np.argmax(rel_dev)
    axes[2].vlines(p_local_list[max_dev_index], B_VIs_gaus[2:][max_dev_index], q_diffs_gaus[2:][max_dev_index] , linestyles = "dotted")

    #Set vertical log scale
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')
    
    
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    handles3, labels3 = axes[2].get_legend_handles_labels()

    fig.legend(handles1+handles2+handles3, labels1+labels2+labels3, bbox_to_anchor = (0.989, 0.1),
            ncol = 2, fontsize = 11, frameon
            =True)

    fig.text(0.5, 0.02, 'Local Depolarising Noise Probability', ha='center', fontsize = 13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.2)
    plt.show()