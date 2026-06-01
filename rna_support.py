def evaluate_sequence(sequence, n_jobs, n_machines, processing_times, machine_sequence):
    """
    Permutation with Repetition Decoder.
    Calculates the exact Makespan (C_max) of a valid operation sequence.
    """
    job_next_op = [0] * n_jobs
    machine_free_time = [0] * n_machines
    job_ready_time = [0] * n_jobs

    for job_id in sequence:
        op_idx = job_next_op[job_id]
        if op_idx >= n_machines:
            continue
        
        machine_id = machine_sequence[job_id][op_idx]
        duration = processing_times[job_id][op_idx]

        # Operation activation rule
        start_time = max(machine_free_time[machine_id], job_ready_time[job_id])
        end_time = start_time + duration

        # Update machine and job states
        machine_free_time[machine_id] = end_time
        job_ready_time[job_id] = end_time
        job_next_op[job_id] += 1

    return max(machine_free_time)


def evaluate_sequence_full(sequence, n_jobs, n_machines, processing_times, machine_sequence):
    """
    Extended Permutation with Repetition Decoder.
    Returns C_max along with full timing matrices and machine execution order,
    which are required for critical path extraction.

    Returns
    -------
    cmax : int
        Makespan.
    start_times : list[list[int]]
        start_times[job_id][op_idx] = start time of that operation.
    end_times : list[list[int]]
        end_times[job_id][op_idx] = end time of that operation.
    machine_order : list[list[tuple]]
        machine_order[machine_id] = list of (job_id, op_idx) in execution order.
    """
    job_next_op = [0] * n_jobs
    machine_free_time = [0] * n_machines
    job_ready_time = [0] * n_jobs

    start_times = [[0] * n_machines for _ in range(n_jobs)]
    end_times = [[0] * n_machines for _ in range(n_jobs)]
    machine_order = [[] for _ in range(n_machines)]

    for job_id in sequence:
        op_idx = job_next_op[job_id]
        if op_idx >= n_machines:
            continue

        machine_id = machine_sequence[job_id][op_idx]
        duration = processing_times[job_id][op_idx]

        # Operation activation rule
        start_time = max(machine_free_time[machine_id], job_ready_time[job_id])
        end_time = start_time + duration

        # Record timing
        start_times[job_id][op_idx] = start_time
        end_times[job_id][op_idx] = end_time

        # Record machine execution order
        machine_order[machine_id].append((job_id, op_idx))

        # Update machine and job states
        machine_free_time[machine_id] = end_time
        job_ready_time[job_id] = end_time
        job_next_op[job_id] += 1

    cmax = max(machine_free_time)
    return cmax, start_times, end_times, machine_order


def generate_gt_sequence(n_jobs, n_machines, processing_times, machine_sequence):
    """
    Generates an initial solution using the Giffler & Thompson (G&T) algorithm
    with the MWKR (Most Work Remaining) dispatching rule.

    Runs the full active-schedule construction and records each scheduling
    decision into a permutation-with-repetition sequence — the exact format
    expected by evaluate_sequence and the RNA optimizer.

    This produces a high-quality starting point because the sequence directly
    encodes a feasible active schedule respecting machine conflicts.
    """
    machine_free_time  = [0] * n_machines
    job_available_time = [0] * n_jobs
    job_current_op     = [0] * n_jobs
    job_remaining_work = [sum(pt) for pt in processing_times]

    total_ops     = n_jobs * n_machines
    ops_scheduled = 0
    sequence      = []   # The permutation-with-repetition sequence being built

    while ops_scheduled < total_ops:
        available_ops = []

        for job_id in range(n_jobs):
            op_idx = job_current_op[job_id]
            if op_idx >= n_machines:
                continue

            machine_id = machine_sequence[job_id][op_idx]
            p_time     = processing_times[job_id][op_idx]

            est = max(machine_free_time[machine_id], job_available_time[job_id])
            ect = est + p_time

            available_ops.append({
                'job_id':    job_id,
                'machine_id': machine_id,
                'est':        est,
                'ect':        ect,
                'p_time':     p_time,
            })

        if not available_ops:
            break

        # G&T rule: find the operation with the earliest completion time
        min_ect_op     = min(available_ops, key=lambda x: x['ect'])
        min_ect        = min_ect_op['ect']
        target_machine = min_ect_op['machine_id']

        # Conflict set: all operations on the same machine that could
        # start before the critical operation finishes
        conflict_set = [
            op for op in available_ops
            if op['machine_id'] == target_machine and op['est'] < min_ect
        ]

        # MWKR dispatching rule: among conflicts, choose the job with
        # the most remaining work
        best_op = max(conflict_set, key=lambda x: job_remaining_work[x['job_id']])

        j_id   = best_op['job_id']
        m_id   = best_op['machine_id']
        p_time = best_op['p_time']
        est    = best_op['est']

        finish_time                = est + p_time
        machine_free_time[m_id]    = finish_time
        job_available_time[j_id]   = finish_time
        job_current_op[j_id]      += 1
        ops_scheduled             += 1
        job_remaining_work[j_id]  -= p_time

        # Record this scheduling decision into the permutation sequence
        sequence.append(j_id)

    return sequence


def find_critical_path(n_jobs, n_machines, cmax, start_times, end_times,
                       machine_order, machine_sequence):
    """
    Traces the critical path backwards from C_max to time zero.

    Starting from the operation whose end_time equals C_max, walks backwards
    by finding the constraining predecessor (the one whose end_time equals the
    current operation's start_time).  The predecessor can be:
      - The previous operation of the same job (job arc).
      - The previous operation on the same machine (machine/disjunctive arc).

    Returns
    -------
    list[tuple[int, int]]
        Ordered list [(job_id, op_idx), ...] from the first critical operation
        to the last one (whose end_time == C_max).
    """
    # Reverse lookup: (job_id, op_idx) -> (machine_id, position_in_machine_order)
    machine_pos = {}
    for m in range(n_machines):
        for pos, (j, o) in enumerate(machine_order[m]):
            machine_pos[(j, o)] = (m, pos)

    # Find an operation whose end_time == cmax (start of backtracking)
    current = None
    for j in range(n_jobs):
        for o in range(n_machines):
            if end_times[j][o] == cmax:
                current = (j, o)
                break
        if current is not None:
            break

    path = []
    while current is not None:
        path.append(current)
        j, o = current
        st = start_times[j][o]

        if st == 0:
            break  # Reached the beginning of the schedule

        # Check machine predecessor
        machine_pred = None
        m_id, pos = machine_pos[(j, o)]
        if pos > 0:
            prev_j, prev_o = machine_order[m_id][pos - 1]
            if end_times[prev_j][prev_o] == st:
                machine_pred = (prev_j, prev_o)

        # Check job predecessor
        job_pred = None
        if o > 0 and end_times[j][o - 1] == st:
            job_pred = (j, o - 1)

        # Prefer machine predecessor (keeps us on the same machine → bigger
        # critical blocks → more swap opportunities).
        if machine_pred is not None:
            current = machine_pred
        elif job_pred is not None:
            current = job_pred
        else:
            break  # No constraining predecessor found

    path.reverse()
    return path


def find_seq_index(sequence, job_id, occurrence):
    """
    Index Mapper for the permutation-with-repetition representation.

    Returns the position (index) in *sequence* where *job_id* appears for
    the (*occurrence*+1)-th time (0-indexed occurrence).

    Example
    -------
    >>> find_seq_index([0, 1, 0, 2, 0], job_id=0, occurrence=2)
    4
    """
    count = 0
    for i, val in enumerate(sequence):
        if val == job_id:
            if count == occurrence:
                return i
            count += 1
    return -1  # Should never happen with valid input


def find_critical_blocks(critical_path, machine_sequence):
    """
    Slices the critical path into Critical Blocks.

    A critical block is a maximal run of two or more consecutive operations
    on the critical path that share the **same machine**.  Single-operation
    "blocks" are discarded because there is nothing to swap inside them.

    Returns
    -------
    list[list[tuple[int, int]]]
        Each inner list is a block of (job_id, op_idx) tuples processed on the
        same machine, in the order they appear on the critical path.
    """
    if not critical_path:
        return []

    blocks = []
    j0, o0 = critical_path[0]
    current_machine = machine_sequence[j0][o0]
    current_block = [critical_path[0]]

    for i in range(1, len(critical_path)):
        j, o = critical_path[i]
        m = machine_sequence[j][o]

        if m == current_machine:
            current_block.append((j, o))
        else:
            if len(current_block) >= 2:
                blocks.append(current_block)
            current_block = [(j, o)]
            current_machine = m

    # Flush the last block
    if len(current_block) >= 2:
        blocks.append(current_block)

    return blocks


def generate_critical_neighbors(sequence, critical_blocks):
    """
    Surgical Neighborhood Generator based on critical blocks.

    For each critical block:
      - Swap the first two operations  → one neighbor.
      - Swap the last  two operations  → another neighbor (only if block > 2).

    Uses :func:`find_seq_index` to locate the exact positions of each
    operation inside the permutation-with-repetition sequence.

    Returns
    -------
    list[list[int]]
        List of neighbor sequences ready to be evaluated.
    """
    neighbors = []

    for block in critical_blocks:
        # --- Swap first two operations of the block ---
        first_op = block[0]   # (job_id, op_idx)
        second_op = block[1]

        idx1 = find_seq_index(sequence, first_op[0], first_op[1])
        idx2 = find_seq_index(sequence, second_op[0], second_op[1])

        neighbor = list(sequence)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        neighbors.append(neighbor)

        # --- Swap last two operations of the block (if block has > 2 ops) ---
        if len(block) > 2:
            second_last_op = block[-2]
            last_op = block[-1]

            idx1 = find_seq_index(sequence, second_last_op[0], second_last_op[1])
            idx2 = find_seq_index(sequence, last_op[0], last_op[1])

            neighbor = list(sequence)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            neighbors.append(neighbor)

    return neighbors
