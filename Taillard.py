from pathlib import Path


def read_taillard_instance(filepath):
    """Lê uma instância Job Shop no formato Taillard e retorna os dados normalizados."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError(f"Arquivo vazio: {path}")

    try:
        n_jobs, n_machines = map(int, lines[0].split())
    except ValueError as exc:
        raise ValueError(
            "Cabeçalho inválido. Esperado: '<num_jobs> <num_machines>'"
        ) from exc

    if n_jobs <= 0 or n_machines <= 0:
        raise ValueError("O número de jobs e máquinas deve ser maior que zero.")

    if len(lines) - 1 < n_jobs:
        raise ValueError(
            f"Instância inválida: esperadas {n_jobs} linhas de jobs, encontradas {len(lines) - 1}."
        )

    processing_times = []
    machine_sequence = []

    for i in range(1, n_jobs + 1):
        tokens = lines[i].split()

        expected_cols = 2 * n_machines
        if len(tokens) != expected_cols:
            raise ValueError(
                f"Linha do job {i} inválida: esperados {expected_cols} valores, encontrados {len(tokens)}."
            )

        try:
            row_data = list(map(int, tokens))
        except ValueError as exc:
            raise ValueError(f"Linha do job {i} contém valores não numéricos.") from exc

        job_machines = []
        job_times = []
        seen_machines = set()

        for j in range(0, len(row_data), 2):
            machine_id = row_data[j]
            proc_time = row_data[j + 1]

            if machine_id < 0 or machine_id >= n_machines:
                raise ValueError(
                    f"Job {i} possui machine_id fora dos limites [0, {n_machines - 1}]: {machine_id}."
                )
            if proc_time < 0:
                raise ValueError(f"Job {i} possui tempo de processamento negativo: {proc_time}.")

            seen_machines.add(machine_id)
            job_machines.append(machine_id)
            job_times.append(proc_time)

        if len(seen_machines) != n_machines:
            raise ValueError(
                f"Job {i} não cobre exatamente {n_machines} máquinas distintas."
            )

        machine_sequence.append(job_machines)
        processing_times.append(job_times)

    return n_jobs, n_machines, processing_times, machine_sequence