import numpy as np
import openturns as ot


def create_samples(n, param_copula, param_T, EVI_T, param_S, EVI_S):
    """
    Creates n coupled target-source samples and m additional source samples.

    Parameters
    ----------
    n : integer
        Number of coupled target-source samples.
    param_copula : float
        Parameter of the copula (positive float for Gumbel, [-1, 1] for Gaussian).
    EVI_T, EVI_S : floats
        Target and source Extreme Value Indexes (negative).
    param_T, param_S : floats
        Additional parameters of target and source marginals.
    """
    # target marginal
    marginal_T = ot.GeneralizedPareto(param_T, EVI_T)

    # source marginal
    marginal_S = ot.GeneralizedPareto(param_S, EVI_S)

    # copula
    copula = ot.GumbelCopula(param_copula)
    distribution = ot.ComposedDistribution([marginal_T, marginal_S], copula)
    sample_T_S_all = ot.RandomVector(distribution).getSample(n)

    sample_T = np.array(sample_T_S_all.getMarginal(0)).ravel()
    sample_S = np.array(sample_T_S_all.getMarginal(1)).ravel()
    return sample_T, sample_S
