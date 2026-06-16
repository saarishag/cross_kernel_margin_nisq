import numpy as np

def calc_margin(svm, K_train, y_train):
    """
    Compute the margin using the provided kernel and the
    dual solution from the svm
    """
    alpha_y = svm.dual_coef_[0] #shape (1, n_support_vectors)

    support_indices = svm.support_ #indices of support vectors
    y_support_vec = y_train[support_indices] #+1/-1 (labels) of the support vectors

    alpha = alpha_y/y_support_vec #to get alpha only not alpha_i*y_i
    K_sv = K_train[np.ix_(support_indices, support_indices)]
    Y = np.diag(y_support_vec)
    sq_weighted_norm = alpha.T @ Y @ K_sv @ Y @ alpha
    margin = 1/sq_weighted_norm
    return np.sqrt(margin)

def calc_C_bounds(p, clean_margin, noisy_margin_est, n, n_layers):
    """Compute the range of acceptable values for 
    C to be used in the upper bound """
    prob_term = (1-p)**(2*n*n_layers)
    Cmax = (2-prob_term)/(2*(clean_margin**2))
    Cmin_est = Cmax - (prob_term/(2*(noisy_margin_est**2)))
    return Cmin_est, Cmax

def calc_C_min_LB(p, clean_margin, noisy_margin_est, n, n_layers):
    """Compute the minimum acceptable C value to be used in the lower bound """
    prob_term = (1-p)**(2*n*n_layers)
    clean_margin_sq = clean_margin**2
    noisy_margin_est_sq = noisy_margin_est**2

    Cmin_est = prob_term/(2*noisy_margin_est_sq)
    Cmin_est -= 1/(2*clean_margin_sq)
    return Cmin_est

def get_upper_params(): 
    """
    Define parameters for the upper bound calc
    Values below obtained using C_region_test.py and 
    chosen to ensure bound validity for the Heart Dataset
    """
    C = 10 
    C_bound = 10 #Or C_bound = C/m 
    clean_margin = 0.1674201006091044
    return C, C_bound, clean_margin

def get_lower_params():
    """
    Define parameters for the lower bound calc
    Values below obtained using C_min_LB_test.py and 
    chosen to ensure bound validity for the Heart Dataset
    """
    C = 10 
    C_bound = 2270 #Or C_bound = C*m 
    clean_margin = 0.17525798616260638
    return C, C_bound, clean_margin
    

def calc_upper_bound(p_local, n, n_layers, clean_margin, C_bound):
    """
    Function to compute the upper bound for a given p_local value
    """
    prob_term = (1-p_local)**(2*n*n_layers)
    denominator = (2*(1-(C_bound*(clean_margin**2))))-prob_term
    sq_bound = (prob_term*(clean_margin**2))/denominator
    bound = np.sqrt(np.abs(sq_bound))
    return bound
         
def calc_lower_bound(p_local, n, clean_margin, C_bound, n_layers = 1):
    """
    Function to compute the lower bound for a given p_local value
    """
    clean_margin_sq = clean_margin**2
    sq_lower_bound = (clean_margin_sq)*(1-p_local)**(2*n*n_layers)
    sq_lower_bound /= ((2*C_bound*clean_margin_sq) + 1)
    lower_bound = np.sqrt(sq_lower_bound)
    return lower_bound


