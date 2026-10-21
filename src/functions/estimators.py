import numpy as np


def EMGF(X, t):
    """
    Estimates the MGF from samples X, with parameter t,
    using a common shift to avoid overflow.

    Parameters
    ----------
    X : numpy array
        Array of samples.
    t : float
        Parameter.
    """
    M = np.max(X) if t >= 0 else np.min(X)
    EMGF = np.exp(t * M) * np.sum(np.exp(t * (X - M)))
    return EMGF


def EMGF_CV(X_T, X_S, X_S_nm, t, alpha=None):
    """
    Estimates the MGF with Approximate Control Variates from target samples X_T, source samples X_S,
    and with parameter t, using a common shift to avoid overflow.

    Parameters
    ----------
    X_T, X_S : numpy arrays
        Arrays of n coupled target and source samples.
    X_S_nm : numpy array
        Array of additional source samples.
    t : float
        Parameter.
    alpha: float, optional
        Control Variates coefficient, if not provided the default is the one minimizing the approximated variance.
    """
    n = np.shape(X_S)[0]
    nm = np.shape(X_S_nm)[0]
    X_all = np.concatenate([X_T, X_S, X_S_nm])
    M = np.max(X_all) if t >= 0 else np.min(X_all)
    Y_T = np.exp(t * (X_T - M))
    Y_S = np.exp(t * (X_S - M))
    Y_S_nm = np.exp(t * (X_S_nm - M))
    if alpha is None: #alpha is unchanged by the shift, because the same shift is applied to Y_T and Y_S
        alpha = np.cov(Y_T, Y_S)[0, 1] / np.var(Y_S)
    return np.sum(Y_T) + alpha*((n/nm) * np.sum(Y_S_nm) - np.sum(Y_S))
