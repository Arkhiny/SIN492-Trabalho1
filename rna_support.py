def evaluate_sequence(sequence, n_jobs, n_machines, processing_times, machine_sequence):
    """
    Decodificador de Permutação com Repetição.
    Calcula o Makespan exato (C_max) de uma sequência válida de operações.
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

        # Regra de ativação da operação
        start_time = max(machine_free_time[machine_id], job_ready_time[job_id])
        end_time = start_time + duration

        # Atualiza estados da máquina e do job
        machine_free_time[machine_id] = end_time
        job_ready_time[job_id] = end_time
        job_next_op[job_id] += 1

    return max(machine_free_time)


def evaluate_sequence_full(sequence, n_jobs, n_machines, processing_times, machine_sequence):
    """
    Decodificador de Permutação com Repetição.
    Retorna o C_max junto com as matrizes completas de temporização
    e a ordem de execução das máquinas, necessárias para a extração
    do caminho crítico.

    Retorna
    -------
    cmax : int
        Makespan.
    start_times : list[list[int]]
        start_times[job_id][op_idx] = instante de início da operação.
    end_times : list[list[int]]
        end_times[job_id][op_idx] = instante de término da operação.
    machine_order : list[list[tuple]]
        machine_order[machine_id] = lista de (job_id, op_idx) na ordem de execução.
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

        # Regra de ativação da operação
        start_time = max(machine_free_time[machine_id], job_ready_time[job_id])
        end_time = start_time + duration

        # Registra temporização
        start_times[job_id][op_idx] = start_time
        end_times[job_id][op_idx] = end_time

        # Registra ordem de execução na máquina
        machine_order[machine_id].append((job_id, op_idx))

        # Atualiza estados da máquina e do job
        machine_free_time[machine_id] = end_time
        job_ready_time[job_id] = end_time
        job_next_op[job_id] += 1

    cmax = max(machine_free_time)
    return cmax, start_times, end_times, machine_order


def generate_gt_sequence(n_jobs, n_machines, processing_times, machine_sequence):
    """
    Gera uma solução inicial usando o algoritmo de Giffler & Thompson (G&T)
    com a regra de despacho MWKR (Most Work Remaining).

    Executa a construção completa de um escalonamento ativo e registra cada
    decisão de alocação em uma sequência de permutação com repetição — o
    formato exato esperado por evaluate_sequence e pelo otimizador RNA.

    """
    machine_free_time  = [0] * n_machines
    job_available_time = [0] * n_jobs
    job_current_op     = [0] * n_jobs
    job_remaining_work = [sum(pt) for pt in processing_times]

    total_ops     = n_jobs * n_machines
    ops_scheduled = 0
    sequence      = []   

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

        # Regra G&T: encontra a operação com o menor tempo de conclusão
        min_ect_op     = min(available_ops, key=lambda x: x['ect'])
        min_ect        = min_ect_op['ect']
        target_machine = min_ect_op['machine_id']

        # Conjunto de conflito: todas as operações na mesma máquina que
        # poderiam iniciar antes da operação crítica terminar
        conflict_set = [
            op for op in available_ops
            if op['machine_id'] == target_machine and op['est'] < min_ect
        ]

        # Regra de despacho MWKR: entre os conflitos, escolhe o job
        # com mais trabalho restante
        best_op = min(conflict_set, key=lambda x: (x['est'], -job_remaining_work[x['job_id']]))
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

        # Registra esta decisão de alocação na sequência de permutação
        sequence.append(j_id)

    return sequence


def find_critical_path(n_jobs, n_machines, cmax, start_times, end_times,
                       machine_order, machine_sequence):
    """
    Rastreia o caminho crítico de trás para frente, de C_max até o tempo zero.

    Partindo da operação cujo end_time é igual a C_max, percorre para trás
    encontrando o predecessor restritivo (aquele cujo end_time é igual ao
    start_time da operação atual). O predecessor pode ser:
      - A operação anterior do mesmo job (arco de job).
      - A operação anterior na mesma máquina (arco de máquina/disjuntivo).

    Retorna
    -------
    list[tuple[int, int]]
        Lista ordenada [(job_id, op_idx), ...] da primeira operação crítica
        até a última (cujo end_time == C_max).
    """
    # Lookup reverso: (job_id, op_idx) -> (machine_id, posição_na_ordem_da_máquina)
    machine_pos = {}
    for m in range(n_machines):
        for pos, (j, o) in enumerate(machine_order[m]):
            machine_pos[(j, o)] = (m, pos)

    # Encontra uma operação cujo end_time == cmax (início do backtracking)
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
            break  # Chegou ao início do escalonamento

        # Verifica predecessor de máquina
        machine_pred = None
        m_id, pos = machine_pos[(j, o)]
        if pos > 0:
            prev_j, prev_o = machine_order[m_id][pos - 1]
            if end_times[prev_j][prev_o] == st:
                machine_pred = (prev_j, prev_o)

        # Verifica predecessor de job
        job_pred = None
        if o > 0 and end_times[j][o - 1] == st:
            job_pred = (j, o - 1)

        # Prefere o predecessor de máquina (mantém na mesma máquina →
        # blocos críticos maiores → mais oportunidades de troca).
        if machine_pred is not None:
            current = machine_pred
        elif job_pred is not None:
            current = job_pred
        else:
            break

    path.reverse()
    return path


def find_seq_index(sequence, job_id, occurrence):
    """
    Retorna a posição (índice) em *sequence* onde *job_id* aparece pela
    (*occurrence*+1)-ésima vez (occurrence indexado em 0).
    """
    count = 0
    for i, val in enumerate(sequence):
        if val == job_id:
            if count == occurrence:
                return i
            count += 1
    return -1 


def find_critical_blocks(critical_path, machine_sequence):
    """
    Divide o caminho crítico em Blocos Críticos.

    Um bloco crítico é uma sequência máxima de duas ou mais operações
    consecutivas no caminho crítico que compartilham a **mesma máquina**.

    Retorna
    -------
    list[list[tuple[int, int]]]
        Cada lista interna é um bloco de tuplas (job_id, op_idx) processadas
        na mesma máquina
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

    # Descarrega o último bloco
    if len(current_block) >= 2:
        blocks.append(current_block)

    return blocks


def generate_critical_neighbors(sequence, critical_blocks):
    """
    Gerador de vizinhança cirúrgico baseado em blocos críticos.

    Para cada bloco crítico:
      - Troca as duas primeiras operações → um vizinho.
      - Troca as duas últimas operações  → outro vizinho (somente se bloco > 2).

    Retorna
    -------
    list[list[int]]
        Lista de sequências vizinhas prontas para avaliação.
    """
    neighbors = []

    for block in critical_blocks:
        # --- Troca as duas primeiras operações do bloco ---
        first_op = block[0]   # (job_id, op_idx)
        second_op = block[1]

        idx1 = find_seq_index(sequence, first_op[0], first_op[1])
        idx2 = find_seq_index(sequence, second_op[0], second_op[1])

        neighbor = list(sequence)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        neighbors.append(neighbor)

        # --- Troca as duas últimas operações do bloco (se bloco > 2 ops) ---
        if len(block) > 2:
            second_last_op = block[-2]
            last_op = block[-1]

            idx1 = find_seq_index(sequence, second_last_op[0], second_last_op[1])
            idx2 = find_seq_index(sequence, last_op[0], last_op[1])

            neighbor = list(sequence)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            neighbors.append(neighbor)

    return neighbors
