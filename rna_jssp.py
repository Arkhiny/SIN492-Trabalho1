import random

from rna_support import (
    evaluate_sequence,
    evaluate_sequence_full,
    find_critical_path,
    find_critical_blocks,
    generate_critical_neighbors,
)


def random_non_ascending(initial_seq, n_jobs, n_machines, p_times, m_sequence,
                         iter_max=500, seed=None):
    """
    Método Não Ascendente Randômico (RNA).

    Differs from VND by accepting any neighbor whose C_max is ≤ the current
    solution (i.e., equal-cost lateral moves are allowed).  A random eligible
    neighbor is chosen each iteration, enabling the search to traverse
    plateaus and escape local optima that trap strict-improvement methods.

    The search terminates after *iter_max* consecutive iterations with no
    improvement to the **global best** solution found so far.

    Parameters
    ----------
    initial_seq : list[int]
        Starting permutation-with-repetition sequence.
    n_jobs, n_machines : int
        Problem dimensions.
    p_times : list[list[int]]
        Processing times matrix.
    m_sequence : list[list[int]]
        Machine routing matrix.
    iter_max : int, optional
        Maximum consecutive iterations without global improvement before
        stopping (default 500).
    seed : int or None, optional
        Random seed for reproducibility (default None).

    Returns
    -------
    best_seq : list[int]
        The best sequence found during the entire search.
    best_cmax : int
        Makespan of the best sequence.
    """
    if seed is not None:
        random.seed(seed)

    # Current solution (the one that "walks")
    current_seq = list(initial_seq)
    current_cmax, st, et, mo = evaluate_sequence_full(
        current_seq, n_jobs, n_machines, p_times, m_sequence
    )

    # Global best (may differ from current after lateral moves)
    best_seq = list(current_seq)
    best_cmax = current_cmax

    no_improve_count = 0  # Consecutive iterations without beating best_cmax

    while no_improve_count < iter_max:
        # 1. Extract critical path and blocks from the CURRENT schedule
        critical_path = find_critical_path(
            n_jobs, n_machines, current_cmax, st, et, mo, m_sequence
        )
        critical_blocks = find_critical_blocks(critical_path, m_sequence)
        neighbors = generate_critical_neighbors(current_seq, critical_blocks)

        if not neighbors:
            break  # No neighbors to explore (degenerate case)

        # 2. Filter: keep only non-ascending neighbors (cmax <= current)
        eligible = []
        for neighbor in neighbors:
            cmax = evaluate_sequence(neighbor, n_jobs, n_machines, p_times, m_sequence)
            if cmax <= current_cmax:
                eligible.append((neighbor, cmax))

        if not eligible:
            # All neighbors are worse → stuck in a strict local minimum
            break

        # 3. Pick one eligible neighbor at random
        chosen_seq, chosen_cmax = random.choice(eligible)

        # 4. Move to the chosen neighbor (lateral or improving)
        current_seq = chosen_seq
        current_cmax = chosen_cmax
        # Recompute full schedule info for the new current solution
        current_cmax, st, et, mo = evaluate_sequence_full(
            current_seq, n_jobs, n_machines, p_times, m_sequence
        )

        # 5. Update global best if this is a strict improvement
        if current_cmax < best_cmax:
            best_seq = list(current_seq)
            best_cmax = current_cmax
            no_improve_count = 0  # Reset counter
        else:
            no_improve_count += 1

    return best_seq, best_cmax
