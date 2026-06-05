"""
teste.py — Script independente de testes estatísticos para JSSP.

Processa automaticamente os grupos ta41-50 e ta71-80.
Cada instância é executada com o algoritmo RNA exatamente 20 vezes.
Instâncias dentro de cada grupo rodam em paralelo (multiprocessing).
Gera o arquivo 'resultado_testes.txt' com os resultados formatados.

Sem interação do usuário (sem menus).
"""

import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from Taillard import read_taillard_instance
from rna_jssp import random_non_ascending
from rna_support import generate_gt_sequence

NUM_RUNS = 20
OUTPUT_FILE = "resultado_testes.txt"


def run_instance_test(instance_path, num_runs=NUM_RUNS):
    """
    Executa o RNA num_runs vezes para uma instância, coletando
    o C_max e o tempo de cada rodada individualmente.

    Retorna
    -------
    best_cmax : int
        Melhor C_max encontrado entre as num_runs execuções.
    avg_cmax : float
        Média dos C_max das num_runs execuções.
    best_time : float
        Tempo da rodada que obteve o melhor C_max.
    """
    n_jobs, n_machines, p_times, m_sequence = read_taillard_instance(instance_path)

    # Solução inicial determinística — gerada 1 vez
    initial_seq = generate_gt_sequence(n_jobs, n_machines, p_times, m_sequence)

    cmax_results = []
    time_results = []

    for _ in range(num_runs):
        start = time.perf_counter()
        _, cmax = random_non_ascending(
            initial_seq, n_jobs, n_machines, p_times, m_sequence
        )
        elapsed = time.perf_counter() - start

        cmax_results.append(cmax)
        time_results.append(elapsed)

    best_cmax = min(cmax_results)
    avg_cmax = sum(cmax_results) / len(cmax_results)

    # Tempo da rodada que obteve o melhor resultado
    best_idx = cmax_results.index(best_cmax)
    best_time = time_results[best_idx]

    return best_cmax, avg_cmax, best_time


def run_instance_test_safe(instance_path_str):
    """
    Wrapper seguro e pickle-safe para ProcessPoolExecutor.
    Recebe string, retorna (name, best_cmax, avg_cmax, best_time, error).
    """
    path = Path(instance_path_str)
    try:
        best_cmax, avg_cmax, best_time = run_instance_test(path)
        return path.name, best_cmax, avg_cmax, best_time, None
    except Exception as e:
        return path.name, None, None, None, str(e)


def process_group(group_name, start_idx, end_idx):
    """
    Processa todas as instâncias de um grupo em paralelo, retornando
    os resultados formatados como lista de linhas.
    """
    lines = []
    lines.append("=" * 55)
    lines.append(f"Grupo: ta{start_idx}-{end_idx} — {NUM_RUNS} execuções por instância")
    lines.append("=" * 55)
    lines.append(f"{'Instância':<16}{'Melhor':<10}{'Média':<12}{'Tempo_Melhor'}")
    lines.append("-" * 55)

    instance_paths = [Path("Data") / f"ta{idx:02d}" for idx in range(start_idx, end_idx + 1)]
    max_workers = min(len(instance_paths), os.cpu_count() or 1)

    print(f"  Processando {len(instance_paths)} instâncias com {max_workers} cores...")

    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_instance_test_safe, str(p)): p
            for p in instance_paths
        }

        for future in as_completed(future_map):
            name, best_cmax, avg_cmax, best_time, error = future.result()
            results[name] = (best_cmax, avg_cmax, best_time, error)

            if error is None:
                print(f"  {name} (OK): Melhor={best_cmax} | Média={avg_cmax:.2f} | Tempo={best_time:.3f}s")
            else:
                print(f"  {name} (ERRO): {error}")

    # Escrever resultados na ordem correta (ta41, ta42, ...)
    all_best = []
    all_avg = []

    for instance_path in instance_paths:
        instance_name = instance_path.name
        if instance_name in results:
            best_cmax, avg_cmax, best_time, error = results[instance_name]
            if error is None:
                lines.append(
                    f"{instance_name:<16}{best_cmax:<10}{avg_cmax:<12.2f}{best_time:.3f}s"
                )
                all_best.append(best_cmax)
                all_avg.append(avg_cmax)
            else:
                lines.append(f"{instance_name:<16}ERRO: {error}")

    # Média do grupo
    lines.append("-" * 55)
    if all_best:
        media_best = sum(all_best) / len(all_best)
        media_avg = sum(all_avg) / len(all_avg)
        lines.append(f"{'MÉDIA DO GRUPO':<16}{media_best:<10.2f}{media_avg:<12.2f}")
    else:
        lines.append("MÉDIA DO GRUPO: Sem resultados válidos")
    lines.append("=" * 55)
    lines.append("")

    return lines


def main():
    print(f"Iniciando testes estatísticos ({NUM_RUNS} execuções por instância)...")
    print()

    all_lines = []

    # Grupo 41-50
    all_lines.extend(process_group("ta41-50", 41, 50))

    # Grupo 71-80
    all_lines.extend(process_group("ta71-80", 71, 80))

    # Salvar arquivo
    output_path = Path(OUTPUT_FILE)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")

    print(f"\nResultados salvos em: {output_path.resolve()}")


if __name__ == "__main__":
    main()
