from pathlib import Path

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