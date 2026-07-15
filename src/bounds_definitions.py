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

def get_full_alpha(svm,y,m): 
    alpha_full = np.zeros(m)
    sv = svm.support_ #indices of support vectors
    alpha_y = svm.dual_coef_[0] #alpha*y
    alpha_full[sv] = alpha_y / y[sv] #alpha only
    return alpha_full

def VI_bounds(K_clean, K_noisy, svm_clean, svm_noisy, tau, y):
    m = len(y)
    y = np.asarray(y)

    alpha_clean = get_full_alpha(svm_clean, y, m)
    alpha_noisy = get_full_alpha(svm_noisy, y, m)

    Y = np.diag(y)
    P = np.eye(m) - np.ones((m, m)) / m #define centering matrix 

    Q_clean = Y @ K_clean @ Y
    deltaK = K_noisy - K_clean #perturbation -> delta K

    #Define z
    z_clean = Y @ alpha_clean
    z_noisy = Y @ alpha_noisy

    # || P delta K P z||
    e_clean = np.linalg.norm(P @ deltaK @ P @ z_clean)
    e_noisy = np.linalg.norm(P @ deltaK @ P @ z_noisy)

    d_exact = np.linalg.norm(alpha_noisy - alpha_clean) 

    d_VI_clean = e_clean / tau
    d_VI_noisy = e_noisy / tau
    d_VI = min(d_VI_clean, d_VI_noisy)

    q0 = alpha_clean @ Q_clean @ alpha_clean
    q_cross = alpha_noisy @ Q_clean @ alpha_noisy
    q_diff = abs(q_cross - q0)

    q_factor = np.linalg.norm(Q_clean @ (alpha_clean + alpha_noisy))
    B_q_VI = q_factor * d_VI

    tol = 1e-10 

    return {
        "e_clean": e_clean,
        "e_noisy": e_noisy,
        "d_exact": d_exact,
        "d_VI_clean": d_VI_clean,
        "d_VI_noisy": d_VI_noisy,
        "d_VI": d_VI,
        "q0": q0,
        "q_cross": q_cross,
        "q_diff": q_diff,
        "q_factor": q_factor,
        "B_q_VI": B_q_VI,
        "VI_holds": d_exact <= d_VI + tol,
        "q_VI_holds": q_diff <= B_q_VI + tol,
    }
