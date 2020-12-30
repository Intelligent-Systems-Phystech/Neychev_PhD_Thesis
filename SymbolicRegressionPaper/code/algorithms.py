import math
import random
import heapq
import numpy as np
import tqdm
import pcst_fast


def make_random_correct_adj_matrix(arities, var_number, complexity_limit=-1):
    """
    Creates a random matrix given vertices arities (root is added)

    Parameters
    -------------------------------------------------
    arities:
        list of integers - arities of corresponding functions

    var_number:
        number of variables (x,y,...)

    complexity_limit:
        Specifies the depth of a function complexity (max depth from root)
    """
    func_num = 1 + len(arities)
    total_num = func_num + var_number

    arities = [1] + arities + [0] * var_number

    queue = [(0, 0)]

    adj_matrix = np.zeros((func_num, total_num))

    var_set = set(range(total_num)) - set(range(func_num))
    not_used = set(range(func_num)) - set([queue[0][0]])

    while not len(queue) == 0:
        func = queue[0][0]
        cmplx = queue[0][1]
        func_arity = arities[func]

        chosen = []
        if cmplx < complexity_limit or complexity_limit == -1:
            if len(not_used) >= func_arity:
                chosen = random.sample(not_used, func_arity)
            else:
                chosen = random.sample(not_used, len(not_used))

        while len(chosen) < func_arity:
            chosen.append(random.sample(var_set, 1)[0])

        for chosen_elem in chosen:
            if chosen_elem < func_num + 1:
                not_used = not_used - set([chosen_elem])
                queue.append((chosen_elem, cmplx + 1))
            adj_matrix[queue[0][0], chosen_elem] = 1
        queue[:] = queue[1:]
    return adj_matrix


def add_noise_to_matrix(
    matrix, noise_level=0, noise_variant="uniform", calibration_variant="sigmoid"
):
    """
    Adds noise to matrix

    Parameters
    -------------------------------------------------
    matrix:
        0-1 matrix with dtype=int

    noise_level:
        used as parameter for chosen variant

    noise_variant:
        Specifies the variant of noise
        noise_variant == 'uniform':
            noisy_matrix = matrix + np.random.uniform(low=-noise_level, high=noise_level)
        noise_variant == 'normal':
            noisy_matrix = matrix + np.random.normal(scale=noise_level)
    calibration_variant:
        Specifies the variant of caluibration
        calibration_variant == 'sigmoid':
            return(1 / (1 + np.exp(-noisy_matrix)))
        calibration_variant == 'linear':
            noisy_matrix -= noisy_matrix.min()
            noisy_matrix = noisy_matrix / noisy_matrix.max()
            return(noisy_matrix)

    """
    noisy_matrix = matrix.copy()
    if noise_variant == "uniform":
        noisy_matrix += np.random.uniform(
            low=-noise_level, high=noise_level, size=noisy_matrix.shape
        )
    elif noise_variant == "normal":
        noisy_matrix += np.random.normal(scale=noise_level, size=noisy_matrix.shape)
    else:
        raise ValueError("No such noise variant: {}".format(noise_variant))

    if calibration_variant == "sigmoid":
        noisy_matrix = 1 / (1 + np.exp(-noisy_matrix))
    elif calibration_variant == "linear":
        noisy_matrix -= noisy_matrix.min()
        noisy_matrix = noisy_matrix / noisy_matrix.max()
    else:
        raise ValueError("No such calibration variant: {}".format(calibration_variant))

    return noisy_matrix


def restore_matrix(
    arities,
    var_number,
    prob_matrix,
    method="greedy_dfs",
    eps=1e-03,
    max_cmplx=-1,
    prize_coef=1.0,
    pruning="gw",
):
    """
    Creates a MatrixFunc object given func_list and
    probabilities matrix prob_matrix using specified method


    Parameters
    -------------------------------------------------
    arities:
        List of integers - arities of corresponding functions

    prob_matrix:
        Probabilities matrix to restore a matrix defining a
        superposition

    method:
        A restoration method

    eps:
        used to drop zero-like elements
    """
    prob_matrix = prob_matrix.copy()
    adj_matrix = np.zeros(prob_matrix.shape, dtype=int)

    if method == "greedy_bfs":

        func_num = 1 + len(arities)
        total_num = func_num + var_number

        arities = [1] + arities + [0] * var_number

        queue = [(0, 0)]

        var_set = set(range(total_num)) - set(range(func_num))
        used = set([queue[0][0]])

        while not len(queue) == 0:
            func = queue[0][0]
            cmplx = queue[0][1]
            func_arity = arities[func]

            matrix_row = prob_matrix[func].copy()
            matrix_row[list(used)] = 0

            if cmplx >= max_cmplx and not max_cmplx == -1:
                matrix_row[list(range(func_num))] = 0

            best_variable_number = matrix_row[func_num:].argmax() + func_num

            chosen = []
            new_level_func = None

            for i in range(0, func_arity):
                new_argmax = np.argmax(matrix_row)
                if matrix_row[new_argmax] < eps:
                    new_argmax = best_variable_number

                matrix_row[new_argmax] = 0
                adj_matrix[func, new_argmax] = 1
                chosen.append(new_argmax)

            for chosen_elem in chosen:
                if chosen_elem < func_num:
                    used.add(chosen_elem)
                    queue.append((chosen_elem, cmplx + 1))
                adj_matrix[queue[0][0], chosen_elem] = 1
            queue[:] = queue[1:]
        return adj_matrix
    elif method == "greedy_dfs":
        arities = [1] + arities

        def GreedyRestoreDFS(
            arities,
            var_number,
            prob_matrix,
            new_matrix,
            max_cmplx=-1,
            cmplx=0,
            level=0,
            used=set([0]),
            eps=1e-03,
        ):

            func_num = len(arities)
            total_num = func_num + var_number

            func = level
            func_arity = arities[func]

            matrix_row = prob_matrix[level].copy()
            matrix_row[list(used)] = 0

            if cmplx >= max_cmplx and not max_cmplx == -1:
                matrix_row[list(range(func_num))] = 0

            best_variable_number = matrix_row[func_num:].argmax() + func_num

            chosen = []
            new_level_func = None

            for i in range(0, func_arity):
                new_argmax = np.argmax(matrix_row)
                if matrix_row[new_argmax] < eps:
                    new_argmax = best_variable_number

                matrix_row[new_argmax] = 0
                new_matrix[func, new_argmax] = 1

                if new_argmax < func_num:
                    used.add(new_argmax)

                chosen.append(new_argmax)

            for chosen_func in chosen:
                if chosen_func < func_num:
                    GreedyRestoreDFS(
                        arities,
                        var_number,
                        prob_matrix,
                        new_matrix,
                        max_cmplx,
                        cmplx + 1,
                        chosen_func,
                        used,
                        eps,
                    )
            return

        GreedyRestoreDFS(
            arities,
            var_number,
            prob_matrix,
            adj_matrix,
            max_cmplx=max_cmplx,
            cmplx=0,
            level=0,
            used=set([0]),
            eps=eps,
        )

        return adj_matrix

    elif method == "prim_fast":
        arities = [1] + arities
        func_num = len(arities)

        in_edges = [0] * (func_num + 1)
        out_edges = [0] * (func_num + 1)

        heap = [(-1, -1, 0)]

        used = set([0])

        while not len(heap) == 0:
            weight, from_vert, to_vert = heapq.heappop(heap)

            if to_vert < func_num:
                func = to_vert

                matrix_row = prob_matrix[func].copy()
                matrix_row[list(used)] = 0

                for to in range(len(matrix_row)):
                    if matrix_row[to] > eps:
                        heapq.heappush(heap, (-matrix_row[to], func, to))

            if not from_vert == -1:
                adj_matrix[from_vert, to_vert] = 1

                arities[from_vert] -= 1

                # Cleaning
                if arities[from_vert] == 0:
                    out_edges[from_vert] = 1

            if to_vert < func_num:
                used.add(to_vert)
                in_edges[to_vert] = 1

            if not len(heap) == 0:
                while out_edges[heap[0][1]] or in_edges[heap[0][2]]:
                    heapq.heappop(heap)
                    if len(heap) == 0:
                        break
        return adj_matrix

    elif method == "prim":
        arities = [1] + arities
        func_num = len(arities)

        last_added = 0
        used = set([0])

        edges = []
        weights = []

        while not last_added is None:

            if last_added < func_num:
                func = last_added

                matrix_row = prob_matrix[func].copy()
                matrix_row[list(used)] = 0

                for to in range(len(matrix_row)):
                    if matrix_row[to] > eps:
                        edges.append((func, to))
                        weights.append(matrix_row[to])

            if len(edges) == 0:
                last_added = None
            else:
                max_edge_num = np.array(weights).argmax()
                max_edge = edges[max_edge_num]

                from_vert = max_edge[0]
                to_vert = max_edge[1]

                adj_matrix[from_vert, to_vert] = 1

                if to_vert < func_num:
                    used.add(to_vert)

                last_added = to_vert

                arities[from_vert] -= 1

                # Cleaning
                to_drop = set([max_edge_num])
                for i, edge in enumerate(edges):
                    if edge[1] in used:
                        to_drop.add(i)

                if arities[from_vert] == 0:
                    for i, edge in enumerate(edges):
                        if edge[0] == from_vert:
                            to_drop.add(i)

                edges = [e for i, e in enumerate(edges) if i not in to_drop]
                weights = [w for i, w in enumerate(weights) if i not in to_drop]

        return adj_matrix
    else:

        def kmst_base(arities, prob_matrix, correct=True, pruning="gw"):
            func_num = 1 + len(arities)
            total_num = func_num + var_number

            for i in range(0, func_num):
                prob_matrix[i, 0] = 0

            func_prob_matr = prob_matrix[:, :func_num]

            if correct:
                func_prob_matr = (func_prob_matr + func_prob_matr.T) / 2.0
            else:
                func_prob_matr = func_prob_matr

            edges = []
            costs = []
            for i in range(0, func_num):
                for j in range(0, func_num):
                    if i == j:
                        func_prob_matr[i, j] = 0
                    if func_prob_matr[i, j] > eps:
                        edges.append([i, j])
                        costs.append(1 - func_prob_matr[i, j] + eps)

            edges = np.array(edges)
            prizes = np.ones(func_num) * prize_coef
            root = 0
            num_clusters = 1
            verbosity_level = 0

            return (
                *pcst_fast.pcst_fast(
                    edges, prizes, costs, root, num_clusters, pruning, verbosity_level
                ),
                func_num,
                edges,
            )

    if method == "kmst_pure":
        vertices, chosen_edges, func_num, edges = kmst_base(
            arities, prob_matrix, True, pruning
        )

        prob_matrix[:, :func_num] = 0
        for edge_number in chosen_edges:
            edge = edges[edge_number]
            prob_matrix[edge[0], edge[1]] = 1
            prob_matrix[edge[1], edge[0]] = 1

        return restore_matrix(
            arities, var_number, prob_matrix, "greedy_dfs", eps, max_cmplx
        )
    elif method == "kmst_dfs":
        vertices, chosen_edges, func_num, edges = kmst_base(
            arities, prob_matrix, True, pruning
        )

        for edge_number in chosen_edges:
            edge = edges[edge_number]
            prob_matrix[edge[0], edge[1]] += 1
            prob_matrix[edge[0], edge[1]] /= 2
            prob_matrix[edge[1], edge[0]] += 1
            prob_matrix[edge[1], edge[0]] /= 2

        return restore_matrix(
            arities, var_number, prob_matrix, "greedy_dfs", eps, max_cmplx
        )
    elif method == "kmst_bfs":
        vertices, chosen_edges, func_num, edges = kmst_base(
            arities, prob_matrix, True, pruning
        )

        for edge_number in chosen_edges:
            edge = edges[edge_number]
            prob_matrix[edge[0], edge[1]] += 1
            prob_matrix[edge[0], edge[1]] /= 2
            prob_matrix[edge[1], edge[0]] += 1
            prob_matrix[edge[1], edge[0]] /= 2

        return restore_matrix(
            arities, var_number, prob_matrix, "greedy_bfs", eps, max_cmplx
        )

    elif method == "kmst_prim":
        vertices, chosen_edges, func_num, edges = kmst_base(
            arities, prob_matrix, True, pruning
        )

        for edge_number in chosen_edges:
            edge = edges[edge_number]
            prob_matrix[edge[0], edge[1]] += 1
            prob_matrix[edge[0], edge[1]] /= 2
            prob_matrix[edge[1], edge[0]] += 1
            prob_matrix[edge[1], edge[0]] /= 2

        return restore_matrix(
            arities, var_number, prob_matrix, "prim_fast", eps, max_cmplx
        )
    elif method == "kmst_pure_incor":
        vertices, chosen_edges, func_num, edges = kmst_base(
            arities, prob_matrix, False, pruning
        )

        prob_matrix[:, :func_num] = 0
        for edge_number in chosen_edges:
            edge = edges[edge_number]
            prob_matrix[edge[0], edge[1]] = 1

        return restore_matrix(
            arities, var_number, prob_matrix, "greedy_bfs", eps, max_cmplx
        )
    elif method == "kmst_dfs_incor":
        vertices, chosen_edges, func_num, edges = kmst_base(
            arities, prob_matrix, False, pruning
        )

        for edge_number in chosen_edges:
            edge = edges[edge_number]
            prob_matrix[edge[0], edge[1]] += 2
            prob_matrix[edge[0], edge[1]] /= 3

        return restore_matrix(
            arities, var_number, prob_matrix, "greedy_dfs", eps, max_cmplx
        )

    elif method == "kmst_bfs_incor":
        vertices, chosen_edges, func_num, edges = kmst_base(
            arities, prob_matrix, False, pruning
        )

        for edge_number in chosen_edges:
            edge = edges[edge_number]
            prob_matrix[edge[0], edge[1]] += 2
            prob_matrix[edge[0], edge[1]] /= 3

        return restore_matrix(
            arities, var_number, prob_matrix, "greedy_bfs", eps, max_cmplx
        )

    elif method == "kmst_prim_incor":
        vertices, chosen_edges, func_num, edges = kmst_base(
            arities, prob_matrix, False, pruning
        )

        for edge_number in chosen_edges:
            edge = edges[edge_number]
            prob_matrix[edge[0], edge[1]] += 2
            prob_matrix[edge[0], edge[1]] /= 3

        return restore_matrix(
            arities, var_number, prob_matrix, "prim_fast", eps, max_cmplx
        )
