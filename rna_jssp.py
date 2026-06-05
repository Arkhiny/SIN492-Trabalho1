import random
import time
from pathlib import Path

from Taillard import read_taillard_instance
from rna_support import (
    evaluate_sequence,
    evaluate_sequence_full,
    find_critical_path,
    find_critical_blocks,
    generate_critical_neighbors,
    generate_gt_sequence,
)


def random_non_ascending(initial_seq, n_jobs, n_machines, p_times, m_sequence,
                         iter_max=100, seed=None):
    """
    Método Não Ascendente Randômico (RNA).

    Aceita qualquer vizinho cujo C_max seja menor ou igual à solução atual (movimentos laterais de custo igual são permitidos).
    Um vizinho elegível é escolhido aleatoriamente a cada iteração, permitindo
    que a busca atravesse platôs e escape de ótimos locais que prendem métodos
    de melhoria estrita.

    A busca termina após *iter_max* iterações consecutivas sem melhoria na
    **melhor solução global** encontrada até o momento.

    Parâmetros
    ----------
    initial_seq : list[int]
        Sequência inicial de permutação com repetição.
    n_jobs, n_machines : int
        Dimensões do problema.
    p_times : list[list[int]]
        Matriz de tempos de processamento.
    m_sequence : list[list[int]]
        Matriz de roteamento de máquinas.
    iter_max : int, opcional
        Máximo de iterações consecutivas sem melhoria global antes de
        parar (padrão 500).
    seed : int ou None, opcional
        Semente aleatória para reprodutibilidade (padrão None).

    Retorna
    -------
    best_seq : list[int]
        A melhor sequência encontrada durante toda a busca.
    best_cmax : int
        Makespan da melhor sequência.
    """
    if seed is not None:
        random.seed(seed)

    # Solução corrente (a que "caminha")
    current_seq = list(initial_seq)
    current_cmax, st, et, mo = evaluate_sequence_full(
        current_seq, n_jobs, n_machines, p_times, m_sequence
    )

    # Melhor global (pode diferir da corrente após movimentos laterais)
    best_seq = list(current_seq)
    best_cmax = current_cmax

    no_improve_count = 0  # Iterações consecutivas sem superar best_cmax

    while no_improve_count < iter_max:
        # 1. Extrai caminho crítico e blocos da solução CORRENTE
        critical_path = find_critical_path(
            n_jobs, n_machines, current_cmax, st, et, mo, m_sequence
        )
        critical_blocks = find_critical_blocks(critical_path, m_sequence)
        neighbors = generate_critical_neighbors(current_seq, critical_blocks)

        if not neighbors:
            break  # Sem vizinhos para explorar (caso degenerado)

        # 2. Filtro: mantém apenas vizinhos não ascendentes (cmax <= corrente)
        eligible = []
        for neighbor in neighbors:
            cmax = evaluate_sequence(neighbor, n_jobs, n_machines, p_times, m_sequence)
            if cmax <= current_cmax:
                eligible.append((neighbor, cmax))

        if not eligible:
            # Todos os vizinhos são piores → preso em mínimo local estrito
            break

        # 3. Escolhe um vizinho elegível aleatoriamente
        chosen_seq, chosen_cmax = random.choice(eligible)

        # 4. Move para o vizinho escolhido (lateral ou de melhoria)
        current_seq = chosen_seq
        current_cmax = chosen_cmax
        # Recalcula informações completas do escalonamento para a nova solução
        current_cmax, st, et, mo = evaluate_sequence_full(
            current_seq, n_jobs, n_machines, p_times, m_sequence
        )

        # 5. Atualiza melhor global se houve melhoria estrita
        if current_cmax < best_cmax:
            best_seq = list(current_seq)
            best_cmax = current_cmax
            no_improve_count = 0  # Reseta contador
        else:
            no_improve_count += 1

    return best_seq, best_cmax


def solve_instance(instance_path):
    """
    Solver principal: lê uma instância Taillard, constrói uma solução
    inicial com G&T + MWKR e a otimiza com RNA (Random Non-Ascending).
    Executa o algoritmo 1 única vez por chamada.
    """
    n_jobs, n_machines, p_times, m_sequence = read_taillard_instance(instance_path)

    initial_seq = generate_gt_sequence(n_jobs, n_machines, p_times, m_sequence)

    best_seq, best_cmax = random_non_ascending(
        initial_seq, n_jobs, n_machines, p_times, m_sequence
    )

    return n_jobs, n_machines, best_cmax


def solve_instance_safe(instance_path):
    """Wrapper seguro para execução que mede o tempo de CPU com precisão."""
    path = Path(instance_path)
    try:
        start_time = time.perf_counter()

        _, _, cmax = solve_instance(path)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        return path.name, cmax, execution_time, None
    except Exception as error:
        return path.name, None, 0.0, str(error)


def solve_instance_gt_only(instance_path):
    """
    Lê a instância, constrói a solução usando apenas Giffler & Thompson
    e avalia o C_max final, ignorando totalmente a busca local.
    """
    n_jobs, n_machines, p_times, m_sequence = read_taillard_instance(instance_path)

    # Gera a sequência inicial estática com G&T
    initial_seq = generate_gt_sequence(n_jobs, n_machines, p_times, m_sequence)

    # Avalia o C_max dessa sequência diretamente
    cmax = evaluate_sequence(initial_seq, n_jobs, n_machines, p_times, m_sequence)

    return n_jobs, n_machines, cmax


def solve_instance_gt_only_safe(instance_path):
    """Wrapper seguro para rodar apenas o G&T medindo o tempo."""
    path = Path(instance_path)
    try:
        start_time = time.perf_counter()

        _, _, cmax = solve_instance_gt_only(path)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        return path.name, cmax, execution_time, None
    except Exception as error:
        return path.name, None, 0.0, str(error)