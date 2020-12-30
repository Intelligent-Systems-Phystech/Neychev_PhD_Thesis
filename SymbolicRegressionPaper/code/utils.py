import random
import tqdm
import numpy as np
from matplotlib import pyplot as plt

from algorithms import (
    add_noise_to_matrix,
    make_random_correct_adj_matrix,
    restore_matrix,
)


def generate_arities_list(size, max_arity=2, p=0.2):
    return (np.random.binomial(max_arity - 1, p, size=size) + 1).tolist()


def do_exp(
    arities_number=200,
    arity_sizes=(5, 50),
    restor_algs=["greedy_dfs", "greedy_bfs", "prim", "kmst"],
    matrices_per_arity_number=100,
    noisy_matrices_number=100,
    noises=np.linspace(0.0, 1.0, 51),
    noise_variant="normal",
    calibration_variant="sigmoid",
    var_number=1,
    max_arity=2,
    max_cmplx=-1,
    p=0.2,
    prize_coef=0.95,
    leave_multiple=False,
):

    arities_list = []
    for i in range(0, arities_number):
        arities_list.append(
            generate_arities_list(
                random.randint(arity_sizes[0], arity_sizes[1]), max_arity, p
            )
        )

    matrices_list = []

    for arities in arities_list:
        for j in range(0, matrices_per_arity_number):
            matrices_list.append(
                (
                    arities,
                    make_random_correct_adj_matrix(
                        arities, var_number, complexity_limit=max_cmplx
                    ),
                )
            )

    recovered = []

    total = arities_number * matrices_per_arity_number * noisy_matrices_number

    for noise_level in tqdm.tqdm_notebook(noises, leave=leave_multiple):
        recovered_per_alg = [0] * len(restor_algs)
        for arities, matrix in matrices_list:
            for j in range(noisy_matrices_number):
                noisy_matrix = add_noise_to_matrix(
                    matrix, noise_level, noise_variant, calibration_variant
                )
                for i, alg in enumerate(restor_algs):
                    rest_matrix = restore_matrix(
                        arities,
                        var_number,
                        noisy_matrix,
                        alg,
                        0.0001,
                        max_cmplx,
                        prize_coef,
                    )
                    if (
                        np.round(matrix[:, :-1]) == np.round(rest_matrix[:, :-1])
                    ).all():
                        recovered_per_alg[i] += 1

        recovered.append(recovered_per_alg)
    return np.array(recovered).T / total


def make_plot(algs, recovered):
    plt.rc("text", usetex=False)

    csfont = {"fontname": "Times"}

    plt.figure(figsize=(4, 7))
    plt.grid(True)

    X = np.linspace(0.0, 1.0, 51)

    plt.plot(X, recovered[0], label="dfs", color="red")
    plt.plot(X, recovered[1], label="bfs", color="blue")
    plt.plot(X, recovered[2], label="prim", color="darkorange")
    plt.plot(X, recovered[-1], label="kmst", color="green")

    plt.ylabel("Доля правильных восстановлений", fontsize=14, **csfont)
    plt.xlabel("Шум", fontsize=14, **csfont)

    plt.title("Процент корректных восстановлений от силы шума", fontsize=18, **csfont)

    plt.legend(loc="upper right", fontsize=14)


def do_exp_multiple(repeats=10, *args, **kwargs):
    kwargs.update({"leave_multiple": False})

    recovered_total = None

    for i in tqdm.tqdm_notebook(range(repeats)):
        recovered = do_exp(*args, **kwargs)
        if recovered_total is None:
            recovered_total = np.zeros(
                (repeats, recovered.shape[0], recovered.shape[1])
            )
            recovered_total[0] = recovered
        else:
            recovered_total[i] = recovered

    return recovered_total


def make_plot_multiple(
    algs,
    recovered,
    repeats,
    alpha_rep=0.2,
    alpha_area=0.1,
    X=np.linspace(0.0, 1.0, 51),
    saveas=None,
    figsize_new=(14, 7),
    incor=False,
    _title="",
):
    plt.rc("text", usetex=False)

    csfont = {"fontname": "Times"}

    plt.figure(figsize=figsize_new)
    plt.grid(True)

    recovered_means = recovered.mean(axis=0)
    recovered_stds = recovered.std(axis=0)
    top = recovered_means + 2 * recovered_stds
    top[top > 1.0] = 1.0

    bot = recovered_means - 2 * recovered_stds
    bot[bot < 0.0] = 0.0

    styles = ["-", "-", "-", "-", "--", "--", "--"]
    colors = ["r", "g", "darkorange", "b", "black", "purple", "fuchsia"]
    if not incor:
        labels = [
            "DFS",
            "BFS",
            "Prim's",
            "k-MST",
            "k-MST-DFS",
            "k-MST-BFS",
            "k-MST-Prim's",
        ]
    else:
        labels = [
            "DFS",
            "BFS",
            "Prim's",
            "Ord-k-MST",
            "Ord-k-MST-DFS",
            "Ord-k-MST-BFS",
            "Ord-k-MST-Prim's",
        ]

    for i in range(0, len(algs)):
        for j in range(0, repeats):
            plt.plot(X, recovered[j, i], color=colors[i], alpha=alpha_rep)
        plt.plot(
            X, recovered_means[i], label=labels[i], color=colors[i], linestyle=styles[i]
        )
        if algs[i] == "prim" or algs[i] == "kmst_prim":
            plt.fill_between(X, top[i], bot[i], color=colors[i], alpha=alpha_area)

    plt.ylabel("Correct restorations ratio", fontsize=16)
    plt.xlabel("Noise", fontsize=16)

    plt.title(_title, fontsize=18)

    plt.legend(loc="upper right", fontsize=12)

    if not saveas is None:
        plt.savefig(saveas, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.show()
