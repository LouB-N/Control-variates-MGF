import pandas as pd
import os


def save_lim_to_csv(mean_coef, std_coef, mean_corr, std_corr, t_list, n, param_copula, param_T, param_S, EVI, EVI_diff, endpoint_diff):
    """
    Save CV coefficient and correlation estimations results across t values, to study their limit and plot them.

    Parameters
    ----------
    mean_coef, std_coef: arrays
        Mean and standard deviations of estimated CV coefficients.
    mean_corr, std_corr: arrays
        Mean and standard deviations of estimated correlations.
    t_list: array
        Temperature parameters t.
    n : integer
        Number of coupled target-source samples. To include information in saved plot names.
    param_copula : float
        Parameter of the copula. To include information in saved plot names.
    param_T, param_S : floats
        Additional parameters of target and source marginals. To include information in saved plot names.
    EVI, diff_EVI : floats
        Extreme Value Index (EVI) value, and difference between EVIs to use.   
    endpoint_diff : float
        Difference between endpoints to use.
    """
    labels = ["x*_T > x*_S, EVI_T < EVI_S", "x*_T > x*_S, EVI_T = EVI_S", "x*_T > x*_S, EVI_T > EVI_S",
              "x*_T = x*_S, EVI_T < EVI_S", "x*_T = x*_S, EVI_T = EVI_S", "x*_T = x*_S, EVI_T > EVI_S",
              "x*_T < x*_S, EVI_T < EVI_S", "x*_T < x*_S, EVI_T = EVI_S", "x*_T < x*_S, EVI_T > EVI_S"]
    KEYS = ["t_sup_s_sup", "t_sup_same", "t_sup_t_sup",
            "same_s_sup", "same_same", "same_t_sup",
            "s_sup_s_sup", "s_sup_same", "s_sup_t_sup",]
    rows = []
    for k, label in zip(KEYS, labels):
        for j, t in enumerate(t_list):
            rows.append({"t": t, "configuration": k, "label": label, "mean_cv_coef": mean_coef[k][j], "std_cv_coef": std_coef[k][j],
                        "mean_correlation": mean_corr[k][j], "std_correlation": std_corr[k][j]})
    results_df = pd.DataFrame(rows)
    csv_filename = (f"results_lim_t_min={t_list[0]}_t_max={t_list[-1]}_n={n}_param_copula={param_copula}_"
                    f"param_T={param_T}_param_S={param_S}_EVI={EVI}_EVI_diff={EVI_diff}_endpoint_diff={endpoint_diff}.csv")
    os.makedirs("results", exist_ok=True) # create folder results if necessary
    results_df.to_csv(os.path.join("results", csv_filename), index=False)
    print(f"Results saved to: {os.path.join('results', csv_filename)}")

