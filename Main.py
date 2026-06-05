from pathlib import Path
from rna_jssp import (solve_instance, solve_instance_safe,
                      solve_instance_gt_only, solve_instance_gt_only_safe)
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single_instance(instance_path):
    """Executa uma única instância e imprime o resultado no console."""
    path = Path(instance_path)
    n_jobs, n_machines, best_cmax = solve_instance(path)

    print(f"Dataset carregado: {n_jobs} jobs e {n_machines} máquinas.")
    print(f"Instância: {path}")
    print(f"Otimização: MWKR + RNA (Random Non-Ascending)")
    print(f"Makespan (C_max): {best_cmax}")


def run_group_and_save(start_idx, end_idx, output_file):
    """Executa um grupo de instâncias em multiprocessamento e salva resultados + média."""
    output_path = Path(output_file)
    cmax_values = []
    lines = []

    lines.append("Instance;C_max;Time")

    instance_paths = [Path("Data") / f"ta{idx:02d}" for idx in range(start_idx, end_idx + 1)]
    results = {}

    max_workers = min(len(instance_paths), os.cpu_count() or 1)

    print(f"Iniciando processamento paralelo com {max_workers} cores...")

    if len(instance_paths) > 1 and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(solve_instance_safe, str(instance_path)): instance_path
                for instance_path in instance_paths
            }

            for future in as_completed(future_map):
                instance_name, cmax, execution_time, error = future.result()
                results[instance_name] = (cmax, execution_time, error)

                if error is None:
                    print(f"{instance_name} (Concluído): C_max = {cmax} | Tempo: {execution_time:.3f}s")
                else:
                    print(f"{instance_name} (Erro): {error}")
    else:
        for instance_path in instance_paths:
            instance_name, cmax, execution_time, error = solve_instance_safe(str(instance_path))
            results[instance_name] = (cmax, execution_time, error)
            if error is None:
                print(f"{instance_name}: C_max = {cmax}")
            else:
                print(f"{instance_name}: ERRO - {error}")

    for instance_path in instance_paths:
        instance_name = instance_path.name
        if instance_name in results:
            cmax, execution_time, error = results[instance_name]
            if error is None:
                cmax_values.append(cmax)
                lines.append(f"{instance_name};{cmax};{execution_time:.3f}s")
            else:
                lines.append(f"{instance_name};ERROR;{error}")

    avg_cmax = (sum(cmax_values) / len(cmax_values)) if cmax_values else 0.0
    lines.append(f"AVERAGE;{avg_cmax:.2f}")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"Resultados registrados em: {output_path}")
    print(f"Makespan médio do grupo: {avg_cmax:.2f}")


def run_group_and_save_gt_only(start_idx, end_idx, output_file):
    """Executa um grupo de instâncias usando apenas a solução inicial (sem RNA) e salva resultados."""
    output_path = Path(output_file)
    cmax_values = []
    lines = []

    instance_paths = [Path("Data") / f"ta{idx:02d}" for idx in range(start_idx, end_idx + 1)]
    results = {}

    max_workers = min(len(instance_paths), os.cpu_count() or 1)
    print(f"Iniciando processamento paralelo (Solução inicial apenas) com {max_workers} cores...")

    if len(instance_paths) > 1 and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(solve_instance_gt_only_safe, str(p)): p
                for p in instance_paths
            }
            for future in as_completed(future_map):
                instance_name, cmax, execution_time, error = future.result()
                results[instance_name] = (cmax, execution_time, error)
                if error is None:
                    print(f"{instance_name} (Concluído): C_max = {cmax} | Tempo: {execution_time:.3f}s")
                else:
                    print(f"{instance_name} (Erro): {error}")
    else:
        for instance_path in instance_paths:
            instance_name, cmax, execution_time, error = solve_instance_gt_only_safe(str(instance_path))
            results[instance_name] = (cmax, execution_time, error)
            if error is None:
                print(f"{instance_name}: C_max = {cmax}")
            else:
                print(f"{instance_name}: ERRO - {error}")

    for instance_path in instance_paths:
        instance_name = instance_path.name
        if instance_name in results:
            cmax, execution_time, error = results[instance_name]
            if error is None:
                cmax_values.append(cmax)
                lines.append(f"{instance_name};{cmax};{execution_time:.3f}s")
            else:
                lines.append(f"{instance_name};ERROR;{error}")

    avg_cmax = (sum(cmax_values) / len(cmax_values)) if cmax_values else 0.0
    lines.append(f"AVERAGE;{avg_cmax:.2f}")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"Resultados registrados em: {output_path}")
    print(f"Makespan médio do grupo: {avg_cmax:.2f}")


def parse_instance_input(user_text):
    """Permite inserir o número da instância (41) ou o caminho completo (Data/ta41)."""
    clean_text = user_text.strip()
    if clean_text.isdigit():
        return Path("Data") / f"ta{int(clean_text):02d}"
    return Path(clean_text)


def ask_algorithm_submenu():
    """Submenu para escolher entre otimização RNA ou apenas solução inicial."""
    while True:
        print("\n  --- Selecione o algoritmo ---")
        print("  1 - Solução inicial + RNA (otimização)")
        print("  2 - Solução inicial apenas")
        print("  0 - Voltar")
        sub = input("  Escolha: ").strip()
        if sub in ("0", "1", "2"):
            return sub
        print("  Opção inválida.")


def run_menu():
    """Menu principal de execução."""
    while True:
        print("\n===== JOB SHOP SCHEDULER =====")
        print("1 - Instância única")
        print("2 - Grupo 41 a 50")
        print("3 - Grupo 71 a 80")
        print("4 - Sair")

        option = input("Escolha uma opção: ").strip()

        if option == "1":
            algo = ask_algorithm_submenu()
            if algo == "0":
                continue
            user_input = input("Insira o número da instância (ex: 41) ou caminho: ").strip()
            try:
                instance_path = parse_instance_input(user_input)
                if algo == "1":
                    run_single_instance(instance_path)
                else:
                    path = Path(instance_path)
                    n_jobs, n_machines, cmax = solve_instance_gt_only(path)
                    print(f"\n[Solução inicial] Instância: {path.name}")
                    print(f"Makespan (C_max): {cmax}")
            except Exception as error:
                print(f"Erro ao processar instância: {error}")
        elif option == "2":
            algo = ask_algorithm_submenu()
            if algo == "0":
                continue
            if algo == "1":
                run_group_and_save(41, 50, "results_group_41_50.txt")
            else:
                run_group_and_save_gt_only(41, 50, "results_group_41_50_baseline.txt")
        elif option == "3":
            algo = ask_algorithm_submenu()
            if algo == "0":
                continue
            if algo == "1":
                run_group_and_save(71, 80, "results_group_71_80.txt")
            else:
                run_group_and_save_gt_only(71, 80, "results_group_71_80_baseline.txt")
        elif option == "4":
            print("Encerrando execução.")
            break
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    run_menu()