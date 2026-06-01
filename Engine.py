from Taillard import validate_instance
import time

def calculate_operation_start_time(machine_free_time, job_ready_time):
    """
    Operation start rule:
    Start = max(time the machine becomes free, time the job finishes the previous operation)
    """
    return max(machine_free_time, job_ready_time)


def calculate_makespan_gt(num_jobs, num_machines, processing_times, machine_sequence):
    """
    Constructive Heuristic using the Giffler & Thompson Algorithm (Active Schedule).
    Dispatching Rule: MWKR (Most Work Remaining).
    """
    validate_instance(num_jobs, num_machines, processing_times, machine_sequence)

    machine_free_time = [0] * num_machines
    job_available_time = [0] * num_jobs
    job_current_op = [0] * num_jobs
    
    job_remaining_work = [sum(pt) for pt in processing_times]

    total_ops = num_jobs * num_machines
    ops_scheduled = 0

    while ops_scheduled < total_ops:
        available_ops = []

        for job_id in range(num_jobs):
            op_idx = job_current_op[job_id]
            if op_idx >= num_machines:
                continue

            machine_id = machine_sequence[job_id][op_idx]
            p_time = processing_times[job_id][op_idx]
            
            est = calculate_operation_start_time(
                machine_free_time[machine_id],
                job_available_time[job_id]
            )
            ect = est + p_time
            
            available_ops.append({
                'job_id': job_id,
                'machine_id': machine_id,
                'est': est,
                'ect': ect,
                'p_time': p_time
            })

        if not available_ops:
            raise RuntimeError("No operations available for scheduling.")
        
        min_ect_op = min(available_ops, key=lambda x: x['ect'])
        min_ect = min_ect_op['ect']
        target_machine = min_ect_op['machine_id']

        conflict_set = [op for op in available_ops 
                        if op['machine_id'] == target_machine and op['est'] < min_ect]

        best_op = max(conflict_set, key=lambda x: job_remaining_work[x['job_id']])

        j_id = best_op['job_id']
        m_id = best_op['machine_id']
        p_time = best_op['p_time']
        est = best_op['est']

        finish_time = est + p_time
        machine_free_time[m_id] = finish_time
        job_available_time[j_id] = finish_time
        
        job_current_op[j_id] += 1
        ops_scheduled += 1
        
        job_remaining_work[j_id] -= p_time

    return max(machine_free_time)


def solve_instance(instance_path):
    """Executes reading + optimization for a single instance."""
    # Delayed imports to avoid circular dependencies
    from Taillard import read_taillard_instance
    from rna_support import generate_gt_sequence
    from rna_jssp import random_non_ascending

    # 1. Load the data
    n_jobs, n_machines, p_times, m_sequence = read_taillard_instance(instance_path)

    # 2. Generate the initial solution using G&T + MWKR (active schedule).
    initial_seq = generate_gt_sequence(n_jobs, n_machines, p_times, m_sequence)

    # 3. Run RNA optimization
    best_seq, best_cmax = random_non_ascending(
        initial_seq, n_jobs, n_machines, p_times, m_sequence
    )

    return n_jobs, n_machines, best_cmax


def solve_instance_safe(instance_path):
    """Safe wrapper for execution that measures CPU time with precision."""
    from pathlib import Path
    path = Path(instance_path)
    try:
        # Start timer
        start_time = time.perf_counter()

        # Run algorithm
        _, _, cmax = solve_instance(path)

        # Stop timer
        end_time = time.perf_counter()

        # Calculate time elapsed
        execution_time = end_time - start_time

        # Return name, cmax, execution time, and None (for error)
        return path.name, cmax, execution_time, None
    except Exception as error:
        # In case of error, return 0.0 for execution time
        return path.name, None, 0.0, str(error)