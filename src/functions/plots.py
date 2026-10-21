import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
plt.set_loglevel("ERROR")


def plot_lim(mean_values, std_values, t_list, filename, corr=True):
    """
    Plot mean correlation/optimal coefficient across z values, to study its limit.

    Parameters
    ----------
    mean_values, std_values: arrays
        Mean and standard deviations of estimated coefficients.
    t_list: array
        Parameters t values.
    filename: string
        Name for the saved plot.
    corr: boolean
        If True, label for correlation, else label for optimal coefficient.
    """
    labels = {"t_sup_same": r"$x*_T > x*_S, \gamma^T = \gamma^S$", "t_sup_t_sup": r"$x*_T > x*_S, \gamma^T > \gamma^S$", "t_sup_s_sup": r"$x*_T > x*_S, \gamma^T < \gamma^S$",
              "same_same": r"$x*_T = x*_S, \gamma^T = \gamma^S$", "same_t_sup": r"$x*_T = x*_S, \gamma^T > \gamma^S$", "same_s_sup": r"$x*_T = x*_S, \gamma^T < \gamma^S$",
              "s_sup_same": r"$x*_T < x*_S, \gamma^T = \gamma^S$", "s_sup_t_sup": r"$x*_T < x*_S, \gamma^T > \gamma^S$", "s_sup_s_sup": r"$x*_T < x*_S, \gamma^T < \gamma^S$"}
    linestyles = {"same_same": ":", "same_t_sup": "-", "same_s_sup": "--",
              "t_sup_same": ":", "t_sup_t_sup": "-", "t_sup_s_sup": "--",
              "s_sup_same": ":", "s_sup_t_sup": "-", "s_sup_s_sup": "--"}
    colors = {"same_same": "red", "same_t_sup": "orange", "same_s_sup": "pink",
              "t_sup_same": "blue", "t_sup_t_sup": "blue", "t_sup_s_sup": "blue",
              "s_sup_same": "green", "s_sup_t_sup": "green", "s_sup_s_sup": "green"}
    plt.figure(figsize=(10, 6))
    for key in list(labels.keys()):
        m_val = mean_values[key]
        s_val = std_values[key]
        plt.plot(t_list, m_val, color=colors[key], linestyle=linestyles[key], label=labels[key])
        plt.fill_between(t_list, m_val - s_val, m_val + s_val, color=colors[key], alpha=0.1)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.xlabel("t")
    if corr:
        plt.ylabel("mean estimated correlation")
    else:
        plt.ylabel("mean estimated optimal coefficient")
        plt.yscale("log")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True) # create folder results if necessary
    plt.savefig(os.path.join("results", f"{filename}.png"), bbox_inches="tight") # save as png for quick preview
    plt.savefig(os.path.join("results", f"{filename}.eps"), bbox_inches="tight") # save as eps for vector format



def plot_lim_paired(mean_corr, mean_coef, t_list, filename):
    """
    Paired plot of correlation and optimal coefficient limits.
    Left: correlation for equal endpoints, with equal/unequal EVIs.
    Right: optimal Control Variate coefficient.

    Parameters
    ----------
    mean_corr, mean_corr: arrays
        Mean estimated correlation and optimal CV coefficient.
    t_list: array
        Parameters t values.
    filename: string
        Name for the saved plot.
    """
    corr_cases = {"same_same": (r"$\gamma^T=\gamma^S$", "blue", ":"), "same_t_sup": (r"$\gamma^T\ne\gamma^S$", "orange", "-")}
    coef_cases = {"same_same": (r"$x_T^*=x_S^*$, $\gamma^T=\gamma^S$", "blue", ":"), "same_t_sup": (r"$x_T^*=x_S^*$, $\gamma^T>\gamma^S$", "orange", "-"),
                  "t_sup_same": (r"$x_T^*>x_S^*$, $\gamma^T=\gamma^S$", "red", "--"), "s_sup_same": (r"$x_T^*<x_S^*$, $\gamma^T=\gamma^S$", "green", "-.")}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Left panel: correlation
    for key, (label, color, linestyle) in corr_cases.items():
        m = np.asarray(mean_corr[key])
        axes[0].plot(t_list, m, color=color, linestyle=linestyle, label=label, linewidth=2)
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel("estimated correlation")
    axes[0].set_title("Correlation")
    axes[0].set_ylim(bottom=0)

    # Right panel: optimal coefficient
    for key, (label, color, linestyle) in coef_cases.items():
        m = np.asarray(mean_coef[key])
        axes[1].plot(t_list, m, color=color, linestyle=linestyle, label=label, linewidth=2)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel("estimated optimal coefficient")
    axes[1].set_title(r"Optimal coefficient $w^*(t)$")
    axes[1].set_yscale("log")

    # shared legend
    legend_handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, label=label)
        for label, color, linestyle in coef_cases.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=True
    )

    fig.tight_layout(w_pad=2.0)
    os.makedirs("results", exist_ok=True) # create folder results if necessary
    plt.savefig(os.path.join("results", f"{filename}.png"), bbox_inches="tight") # save as png for quick preview
    plt.savefig(os.path.join("results", f"{filename}.eps"), bbox_inches="tight") # save as eps for vector format
