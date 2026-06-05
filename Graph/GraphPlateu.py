"""
GraphPlateu.py — Gráfico de convergência do RNA para artigo acadêmico.

Executa o RNA com iterações configuráveis e plota a evolução do melhor C_max
encontrado a cada iteração. Gráfico limpo e profissional, adequado
para publicação científica.
"""

import random
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# ── Configuração ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ITER_MAX = 50
INSTANCE_FILE = "ta71"  # Instância a ser analisada


def random_non_ascending_with_history(initial_seq, n_jobs, n_machines,
                                      p_times, m_sequence,
                                      iter_max=ITER_MAX, seed=None):
    """
    RNA idêntico ao original, mas retorna também o histórico do melhor
    C_max global a cada iteração — necessário para o gráfico de platô.

    Retorna
    -------
    best_seq : list[int]
        Melhor sequência encontrada.
    best_cmax : int
        Makespan da melhor sequência.
    history : list[int]
        history[i] = melhor C_max global após a iteração i.
    """
    if seed is not None:
        random.seed(seed)

    current_seq = list(initial_seq)
    current_cmax, st, et, mo = evaluate_sequence_full(
        current_seq, n_jobs, n_machines, p_times, m_sequence
    )

    best_seq = list(current_seq)
    best_cmax = current_cmax

    history = [best_cmax]  # Iteração 0 = solução inicial
    no_improve_count = 0

    while no_improve_count < iter_max:
        critical_path = find_critical_path(
            n_jobs, n_machines, current_cmax, st, et, mo, m_sequence
        )
        critical_blocks = find_critical_blocks(critical_path, m_sequence)
        neighbors = generate_critical_neighbors(current_seq, critical_blocks)

        if not neighbors:
            break

        eligible = []
        for neighbor in neighbors:
            cmax = evaluate_sequence(neighbor, n_jobs, n_machines, p_times, m_sequence)
            if cmax <= current_cmax:
                eligible.append((neighbor, cmax))

        if not eligible:
            break

        chosen_seq, chosen_cmax = random.choice(eligible)

        current_seq = chosen_seq
        current_cmax = chosen_cmax
        current_cmax, st, et, mo = evaluate_sequence_full(
            current_seq, n_jobs, n_machines, p_times, m_sequence
        )

        if current_cmax < best_cmax:
            best_seq = list(current_seq)
            best_cmax = current_cmax
            no_improve_count = 0
        else:
            no_improve_count += 1

        history.append(best_cmax)

    return best_seq, best_cmax, history


def plot_convergence(history, instance_name, output_path):
    """
    Gera um gráfico de convergência limpo para artigo acadêmico.

    Mostra apenas a curva do melhor C_max ao longo das iterações,
    sem informações extras — ideal para publicação.
    """
    # ── Estilo acadêmico ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         12,
        "axes.linewidth":    0.8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi":        300,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    iterations = list(range(len(history)))

    # ── Linha principal ───────────────────────────────────────────────────
    ax.plot(iterations, history,
            color="#1a1a1a",
            linewidth=1.2,
            zorder=3)

    # ── Eixos ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Iteração", fontsize=13, labelpad=8)
    ax.set_ylabel("$C_{\\mathrm{max}}$", fontsize=13, labelpad=8)

    # Limites do eixo Y com margem proporcional
    y_min = min(history)
    y_max = max(history)
    y_range = y_max - y_min if y_max > y_min else y_max * 0.05
    ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.08)
    ax.set_xlim(0, len(history) - 1)

    # Grade sutil
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#999999")
    ax.set_axisbelow(True)

    # Formatação dos ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

    # Remover bordas superior e direita (estilo acadêmico)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Salvar ────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Gráfico salvo em: {output_path}")
    plt.close(fig)


def main():
    instance_path = Path(BASE_DIR) / "Data" / INSTANCE_FILE
    print(f"Carregando instância: {instance_path}")

    n_jobs, n_machines, p_times, m_sequence = read_taillard_instance(instance_path)
    print(f"  {n_jobs} jobs × {n_machines} máquinas")

    # Solução inicial com G&T
    initial_seq = generate_gt_sequence(n_jobs, n_machines, p_times, m_sequence)
    initial_cmax = evaluate_sequence(initial_seq, n_jobs, n_machines, p_times, m_sequence)
    print(f"  C_max inicial (G&T): {initial_cmax}")

    # Executar RNA com histórico
    print(f"  Executando RNA com iter_max={ITER_MAX}...")
    best_seq, best_cmax, history = random_non_ascending_with_history(
        initial_seq, n_jobs, n_machines, p_times, m_sequence,
        iter_max=ITER_MAX
    )
    print(f"  C_max final: {best_cmax}")
    print(f"  Total de iterações: {len(history)}")

    # Gerar gráfico
    output_file = os.path.join(BASE_DIR, f"convergencia_{INSTANCE_FILE}.png")
    plot_convergence(history, INSTANCE_FILE, output_file)


if __name__ == "__main__":
    main()
