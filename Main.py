from pathlib import Path
import sys
from Engine import solve_instance, solve_instance_safe
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single_instance(instance_path):
    """Runs a single instance and prints the result to the console."""
    path = Path(instance_path)
    n_jobs, n_machines, cmax = solve_instance(path)

    print(f"Dataset loaded: {n_jobs} jobs and {n_machines} machines.")
    print(f"Instance: {path}")
    print(f"Optimization: MWKR + RNA (Random Non-Ascending)")
    print(f"Calculated Makespan (C_max): {cmax}")


def run_group_and_save(start_idx, end_idx, output_file):
    """Runs a group of instances in multiprocessing and saves results + average."""
    output_path = Path(output_file)
    cmax_values = []
    lines = []

    instance_paths = [Path("Data") / f"ta{idx:02d}" for idx in range(start_idx, end_idx + 1)]
    results = {}
    
    # Determine the maximum number of concurrent processes.
    # Uses the smaller value between the number of instances and the number of CPU cores.
    max_workers = min(len(instance_paths), os.cpu_count() or 1)

    print(f"Starting parallel processing with {max_workers} cores...")

    if len(instance_paths) > 1 and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map each future submission to the corresponding instance path
            future_map = {
                executor.submit(solve_instance_safe, str(instance_path)): instance_path
                for instance_path in instance_paths
            }

            # as_completed returns results as each process finishes
            for future in as_completed(future_map):
                # 1. Unpack the 4 values:
                instance_name, cmax, execution_time, error = future.result() 
                
                # 2. Store the important values in the dictionary:
                results[instance_name] = (cmax, execution_time, error) 
                
                if error is None:
                    print(f"{instance_name} (Completed): C_max = {cmax} | Time: {execution_time:.3f}s")
                else:
                    print(f"{instance_name} (Error): {error}")
    else:
        # Fallback to sequential execution if there is only one instance or CPU core
        for instance_path in instance_paths:
            instance_name, cmax, execution_time, error = solve_instance_safe(str(instance_path))
            results[instance_name] = (cmax, execution_time, error)
            if error is None:
                print(f"{instance_name}: C_max = {cmax}")
            else:
                print(f"{instance_name}: ERROR - {error}")

    for instance_path in instance_paths:
        instance_name = instance_path.name
        if instance_name in results:
            cmax, execution_time, error = results[instance_name]
            if error is None:
                cmax_values.append(cmax)
                # Store in format: name;Cmax;Time
                lines.append(f"{instance_name};{cmax};{execution_time:.3f}s")
            else:
                lines.append(f"{instance_name};ERROR;{error}")

    avg_cmax = (sum(cmax_values) / len(cmax_values)) if cmax_values else 0.0
    lines.append(f"AVERAGE;{avg_cmax:.2f}")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"Results recorded in: {output_path}")
    print(f"Group average Makespan: {avg_cmax:.2f}")


def parse_instance_input(user_text):
    """Allows entering the instance number (41) or full path (Data/ta41)."""
    clean_text = user_text.strip()
    if clean_text.isdigit():
        return Path("Data") / f"ta{int(clean_text):02d}"
    return Path(clean_text)


def run_menu():
    """Main execution menu."""
    while True:
        print("\n===== JOB SHOP SCHEDULER =====")
        print("1 - Run single instance")
        print("2 - Run group 41 to 50")
        print("3 - Run group 71 to 80")
        print("4 - Exit")

        option = input("Choose an option: ").strip()

        if option == "1":
            user_input = input("Enter the instance number (e.g., 41) or path (e.g., Data/ta41): ").strip()
            try:
                instance_path = parse_instance_input(user_input)
                run_single_instance(instance_path)
            except Exception as error:
                print(f"Error processing instance: {error}")
        elif option == "2":
            run_group_and_save(41, 50, "results_group_41_50.txt")
        elif option == "3":
            run_group_and_save(71, 80, "results_group_71_80.txt")
        elif option == "4":
            print("Terminating execution.")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    run_menu()