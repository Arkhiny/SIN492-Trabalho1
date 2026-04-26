from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def read_taillard_instance(filepath):
    """Reads a Taillard Job Shop instance and returns normalized data."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    try:
        n_jobs, n_machines = map(int, lines[0].split())
    except ValueError as exc:
        raise ValueError(
            "Invalid header. Expected: '<num_jobs> <num_machines>'"
        ) from exc

    if n_jobs <= 0 or n_machines <= 0:
        raise ValueError("Number of jobs and machines must be greater than zero.")

    if len(lines) - 1 < n_jobs:
        raise ValueError(
            f"Invalid instance: expected {n_jobs} job lines, found {len(lines) - 1}."
        )

    processing_times = []
    machine_sequence = []

    for i in range(1, n_jobs + 1):
        tokens = lines[i].split()

        expected_cols = 2 * n_machines
        if len(tokens) != expected_cols:
            raise ValueError(
                f"Invalid job {i} line: expected {expected_cols} values, found {len(tokens)}."
            )

        try:
            row_data = list(map(int, tokens))
        except ValueError as exc:
            raise ValueError(f"Job {i} line contains non-numeric values.") from exc

        job_machines = []
        job_times = []
        seen_machines = set()

        for j in range(0, len(row_data), 2):
            machine_id = row_data[j]
            proc_time = row_data[j + 1]

            if machine_id < 0 or machine_id >= n_machines:
                raise ValueError(
                    f"Job {i} has machine_id out of bounds [0, {n_machines - 1}]: {machine_id}."
                )
            if proc_time < 0:
                raise ValueError(f"Job {i} has a negative processing time: {proc_time}.")

            seen_machines.add(machine_id)
            job_machines.append(machine_id)
            job_times.append(proc_time)

        if len(seen_machines) != n_machines:
            raise ValueError(
                f"Job {i} does not cover exactly {n_machines} distinct machines."
            )

        machine_sequence.append(job_machines)
        processing_times.append(job_times)

    return n_jobs, n_machines, processing_times, machine_sequence


def validate_instance(num_jobs, num_machines, processing_times, machine_sequence):
    """Validates dimensions to avoid silent failures during scheduling."""
    if len(processing_times) != num_jobs:
        raise ValueError("Number of rows in processing_times differs from num_jobs.")
    if len(machine_sequence) != num_jobs:
        raise ValueError("Number of rows in machine_sequence differs from num_jobs.")

    for j in range(num_jobs):
        if len(processing_times[j]) != num_machines:
            raise ValueError(f"Job {j} in processing_times does not have {num_machines} operations.")
        if len(machine_sequence[j]) != num_machines:
            raise ValueError(f"Job {j} in machine_sequence does not have {num_machines} operations.")


def calculate_operation_start_time(machine_free_time, job_ready_time):
    """
    Operation start rule:
    Start = max(time the machine becomes free, time the job finishes the previous operation)
    """
    return max(machine_free_time, job_ready_time)


def calculate_makespan_gt(num_jobs, num_machines, processing_times, machine_sequence, rule="MWKR"):
    """
    Constructive Heuristic using the Giffler & Thompson Algorithm (Active Schedule).
    Allows choosing between Dispatching Rules: 'SPT' or 'MWKR'.
    """
    validate_instance(num_jobs, num_machines, processing_times, machine_sequence)

    machine_free_time = [0] * num_machines
    job_available_time = [0] * num_jobs
    job_current_op = [0] * num_jobs
    
    # Pre-calculate the total remaining work for each Job (Required for the MWKR rule)
    job_remaining_work = [sum(pt) for pt in processing_times]

    total_ops = num_jobs * num_machines
    ops_scheduled = 0

    while ops_scheduled < total_ops:
        available_ops = []

        # 1. Identify the next pending operation for each Job and calculate its times
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
            ect = est + p_time # Earliest Completion Time (When this task would finish)
            
            available_ops.append({
                'job_id': job_id,
                'machine_id': machine_id,
                'est': est,
                'ect': ect,
                'p_time': p_time
            })

        if not available_ops:
            raise RuntimeError("No operations available for scheduling.")
        
        # 2. Giffler & Thompson: Find the operation that FINISHES earliest
        min_ect_op = min(available_ops, key=lambda x: x['ect'])
        min_ect = min_ect_op['ect']
        target_machine = min_ect_op['machine_id']

        # 3. Form the Conflict Set
        # Gets all tasks that want to use the same machine and that could 
        # start before the fastest task finishes.
        conflict_set = [op for op in available_ops 
                        if op['machine_id'] == target_machine and op['est'] < min_ect]

        # 4. APPLY DISPATCHING RULE
        if rule == "MWKR":
            # MWKR: Prioritizes the job with the MOST total remaining factory work
            best_op = max(conflict_set, key=lambda x: job_remaining_work[x['job_id']])
        else:
            # SPT: Prioritizes the job with the SHORTEST immediate processing time
            best_op = min(conflict_set, key=lambda x: x['p_time'])

        # 5. Schedule the winning operation and update the simulation
        j_id = best_op['job_id']
        m_id = best_op['machine_id']
        p_time = best_op['p_time']
        est = best_op['est']

        finish_time = est + p_time
        machine_free_time[m_id] = finish_time
        job_available_time[j_id] = finish_time
        
        job_current_op[j_id] += 1
        ops_scheduled += 1
        
        # Subtract the completed work from the job's total remaining work
        job_remaining_work[j_id] -= p_time

    return max(machine_free_time)


def solve_instance(instance_path):
    """Executes reading + makespan calculation for a single instance."""
    n_jobs, n_machines, p_times, m_sequence = read_taillard_instance(instance_path)
    
    # Now you pass the rule you want to test ("MWKR" or "SPT")
    cmax = calculate_makespan_gt(n_jobs, n_machines, p_times, m_sequence, rule="MWKR")
    
    return n_jobs, n_machines, cmax


def run_single_instance(instance_path):
    """Runs a single instance and prints the result to the console."""
    path = Path(instance_path)
    n_jobs, n_machines, cmax = solve_instance(path)

    print(f"Dataset loaded: {n_jobs} jobs and {n_machines} machines.")
    print(f"Instance: {path}")
    print("Constructive Heuristic - Dispatching Rule: SPT")
    print(f"Calculated Makespan (C_max): {cmax}")


def solve_instance_safe(instance_path):
    """Safe wrapper for multiprocessing execution."""
    path = Path(instance_path)
    try:
        _, _, cmax = solve_instance(path)
        return path.name, cmax, None
    except Exception as error:
        return path.name, None, str(error)


def run_group_and_save(start_idx, end_idx, output_file):
    """Runs a group of instances in multiprocessing and saves results + average."""
    output_path = Path(output_file)
    cmax_values = []
    lines = []

    instance_paths = [Path("Data") / f"ta{idx:02d}" for idx in range(start_idx, end_idx + 1)]
    results = {}
    max_workers = min(len(instance_paths), os.cpu_count() or 1)

    if len(instance_paths) > 1 and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(solve_instance_safe, str(instance_path)): instance_path
                for instance_path in instance_paths
            }

            for future in as_completed(future_map):
                instance_name, cmax, error = future.result()
                results[instance_name] = (cmax, error)
                if error is None:
                    print(f"{instance_name}: C_max = {cmax}")
                else:
                    print(f"{instance_name}: ERROR - {error}")
    else:
        for instance_path in instance_paths:
            instance_name, cmax, error = solve_instance_safe(str(instance_path))
            results[instance_name] = (cmax, error)
            if error is None:
                print(f"{instance_name}: C_max = {cmax}")
            else:
                print(f"{instance_name}: ERROR - {error}")

    for instance_path in instance_paths:
        instance_name = instance_path.name
        cmax, error = results[instance_name]
        if error is None:
            cmax_values.append(cmax)
            lines.append(f"{instance_name};{cmax}")
        else:
            lines.append(f"{instance_name};ERROR;{error}")

    avg_cmax = (sum(cmax_values) / len(cmax_values)) if cmax_values else 0.0
    lines.append(f"AVERAGE;{avg_cmax:.2f}")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"Results saved to: {output_path}")
    print(f"Group average: {avg_cmax:.2f}")


def parse_instance_input(user_text):
    """Allows entering the instance number (41) or full path (Data/ta41)."""
    clean_text = user_text.strip()
    if clean_text.isdigit():
        return Path("Data") / f"ta{int(clean_text):02d}"
    return Path(clean_text)


def run_menu():
    """Main execution menu."""
    while True:
        print("\n===== JOB SHOP MENU (SPT) =====")
        print("1 - Run single instance")
        print("2 - Run group 41 to 50")
        print("3 - Run group 71 to 80")
        print("4 - Exit")

        option = input("Choose an option: ").strip()

        if option == "1":
            user_input = input(
                "Enter the instance number (e.g., 41) or path (e.g., Data/ta41): "
            ).strip()

            try:
                instance_path = parse_instance_input(user_input)
                run_single_instance(instance_path)
            except Exception as error:
                print(f"Error processing instance: {error}")

        elif option == "2":
            run_group_and_save(41, 50, "resultado_grupo_41_50.txt")

        elif option == "3":
            run_group_and_save(71, 80, "resultado_grupo_71_80.txt")

        elif option == "4":
            print("Terminating execution.")
            break

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default action when no arguments: show menu (works with Code Runner)
        run_menu()
    elif sys.argv[1] == "--menu":
        run_menu()
    elif sys.argv[1] == "--group-41-50":
        run_group_and_save(41, 50, "resultado_grupo_41_50.txt")
    elif sys.argv[1] == "--group-71-80":
        run_group_and_save(71, 80, "resultado_grupo_71_80.txt")
    else:
        try:
            run_single_instance(sys.argv[1])
        except Exception as error:
            print(f"Error processing instance: {error}")
            sys.exit(1)