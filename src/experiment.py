import argparse
import random
from tqdm import tqdm
import numpy as np
import openturns as ot
from src.functions.create_samples import create_samples
from src.functions.plots import plot_lim, plot_lim_paired
from src.functions.save_to_csv import save_lim_to_csv


# Argument parser setup
parser = argparse.ArgumentParser(description="Tests limits of optimal CV coefficient and correlation.")
parser.add_argument("--t_min", type=float, default=1)
parser.add_argument("--t_max", type=float, default=150)
parser.add_argument("--t_step", type=float, default=2)
parser.add_argument("--n", type=int, default=1_000_000)
parser.add_argument("--param_copula", type=float, default=3)
parser.add_argument("--param_T", type=float, default=0.1)
parser.add_argument("--param_S", type=float, default=0.1)
parser.add_argument("--EVI", type=float, default=-2)
parser.add_argument("--EVI_diff", type=float, default=1.5)
parser.add_argument("--endpoint", type=float, default=1)
parser.add_argument("--endpoint_diff", type=float, default=0.5)
parser.add_argument("--n_exp", type=int, default=1)
args = parser.parse_args()

t_list = np.arange(args.t_min, args.t_max, args.t_step)
n_t, n_exp, n, m = len(t_list), args.n_exp, int(args.n), 1

# Define key matrix configurations
KEYS = [
    "t_sup_s_sup",
    "t_sup_same",
    "t_sup_t_sup",
    "same_s_sup",
    "same_same",
    "same_t_sup",
    "s_sup_s_sup",
    "s_sup_same",
    "s_sup_t_sup",
]

cv_coef = {k: np.zeros((n_t, n_exp)) for k in KEYS}
correlation = {k: np.zeros((n_t, n_exp)) for k in KEYS}

# Experiment scenarios setup
scenarios = [
    (args.EVI, args.EVI, ("same_same", "t_sup_same", "s_sup_same")),  # EVI_T == EVI_S
    (args.EVI + args.EVI_diff, args.EVI, ("same_t_sup", "t_sup_t_sup", "s_sup_t_sup")),  # EVI_T > EVI_S
    (args.EVI, args.EVI + args.EVI_diff, ("same_s_sup", "t_sup_s_sup", "s_sup_s_sup")),  # EVI_T < EVI_S
]

# Monte Carlo simulation loop
for i in tqdm(range(n_exp), desc="Running experiment") :
    random.seed(i); ot.RandomGenerator.SetSeed(i); np.random.seed(i) # set random seeds

    for evi_t, evi_s, (k_same, k_tsup, k_ssup) in scenarios:
        sample_T, sample_S = create_samples(n, args.param_copula, args.param_T, evi_t, args.param_S, evi_s)

        sig_T = -evi_t * args.endpoint
        sig_S = -evi_s * args.endpoint

        sample_T, sample_S = create_samples(n, args.param_copula, sig_T, evi_t, sig_S, evi_s)

        shift_t_sup = sample_T + args.endpoint_diff
        shift_s_sup = sample_T - args.endpoint_diff

        for k, t in enumerate(t_list):
            # Common shift to avoid overflow across all target/source configurations
            X_all = np.concatenate([sample_T, sample_S, shift_t_sup, shift_s_sup])
            M = np.max(X_all) if t >= 0 else np.min(X_all)

            exp_T = np.exp(t * (sample_T - M))
            exp_S = np.exp(t * (sample_S - M))
            exp_T_tsup = np.exp(t * (shift_t_sup - M))
            exp_T_ssup = np.exp(t * (shift_s_sup - M))
            var_S = np.nanvar(exp_S, ddof=1)

            # CV coef (because the shift M is the same on both variable, the coefficient is unchanged)
            cv_coef[k_same][k, i] = np.cov(exp_T, exp_S, ddof=1)[0, 1] / var_S
            cv_coef[k_tsup][k, i] = np.cov(exp_T_tsup, exp_S, ddof=1)[0, 1] / var_S
            cv_coef[k_ssup][k, i] = np.cov(exp_T_ssup, exp_S, ddof=1)[0, 1] / var_S

            # Correlation (invariant to the shift M)
            correlation[k_same][k, i] = np.corrcoef(exp_T, exp_S)[0, 1]
            correlation[k_tsup][k, i] = np.corrcoef(exp_T_tsup, exp_S)[0, 1]
            correlation[k_ssup][k, i] = np.corrcoef(exp_T_ssup, exp_S)[0, 1]           

# Compute statistics
mean_coef = {k: np.nanmean(cv_coef[k], axis=1) for k in KEYS}
std_coef = {k: np.nanstd(cv_coef[k], axis=1) for k in KEYS}
mean_corr = {k: np.nanmean(correlation[k], axis=1) for k in KEYS}
std_corr = {k: np.nanstd(correlation[k], axis=1) for k in KEYS}

# save results to csv
save_lim_to_csv(mean_coef, std_coef, mean_corr, std_corr, t_list, n, args.param_copula, args.param_T, args.param_S,
                args.EVI, args.EVI_diff, args.endpoint_diff)

# plot results
plot_lim(mean_coef, std_coef, t_list,
        filename=(f"lim_CV_coef_t_min={t_list[0]}_t_max={t_list[-1]}_n={n}_param_copula={args.param_copula}_"
        f"param_T={args.param_T}_param_S={args.param_S}_EVI={args.EVI}_EVI_diff={args.EVI_diff}_endpoint_diff={args.endpoint_diff}"),
        corr=False) # Plot 1: Mean optimal CV coefficient
plot_lim(mean_corr, std_corr, t_list,
        filename=(f"lim_corr_t_min={t_list[0]}_t_max={t_list[-1]}_n={n}_param_copula={args.param_copula}_"
        f"param_T={args.param_T}_param_S={args.param_S}_EVI={args.EVI}_EVI_diff={args.EVI_diff}_endpoint_diff={args.endpoint_diff}"),
        corr=True) # Plot 2: Mean correlation
plot_lim_paired(mean_corr, mean_coef, t_list,
                filename=(f"both_lim_t_min={t_list[0]}_t_max={t_list[-1]}_n={n}_param_copula={args.param_copula}_"
                f"param_T={args.param_T}_param_S={args.param_S}_EVI={args.EVI}_EVI_diff={args.EVI_diff}_endpoint_diff={args.endpoint_diff}"),
                ) # Plot 3: Paired plot (correlation and optimal CV coefficient)
