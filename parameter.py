"""
GraphIterMax.py — Gráfico de sensibilidade do parâmetro iter_max para artigo acadêmico.

Executa o algoritmo RNA variando o critério de parada (iter_max) de 1 a 1000
(com saltos configuráveis) e plota o C_max final encontrado. O objetivo é
justificar visualmente o ponto de equilíbrio ótimo entre esforço computacional
e qualidade da solução para a instância escolhida.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Importando as funções do seu projeto original
from Taillard import read_taillard_instance
from rna_support import generate_gt_sequence
from rna_jssp import random_non_ascending

# ── Configuração ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_FILE = "ta71"  # Instância a ser analisada

# Configuração do eixo X (iter_max)
START_ITER = 10
END_ITER = 100
STEP = 1  # Salto entre os testes (ex: 10, 20, 30... 1000). Mude para 1 se quiser todos.

# Semente fixa para garantir que o gráfico seja consistente e sem ruído aleatório
SEED = 42 


def plot_itermax_sensitivity(x_values, y_values, instance_name, output_path):
    """
    Gera um gráfico de sensibilidade limpo para artigo acadêmico.
    Eixo X: Valores de iter_max testados.
    Eixo Y: C_max final obtido.
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

    # ── Linha principal e marcadores ──────────────────────────────────────
    ax.plot(x_values, y_values,
            color="#1a1a1a",
            linewidth=1.2,
            marker="o",
            markersize=3,
            markerfacecolor="white",
            markeredgewidth=0.8,
            zorder=3)

    # ── Eixos ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Critério de Parada (iter_max)", fontsize=13, labelpad=8)
    ax.set_ylabel("$C_{\\mathrm{max}}$ Encontrado", fontsize=13, labelpad=8)

    # Limites dos eixos com margem proporcional
    y_min = min(y_values)
    y_max = max(y_values)
    y_range = y_max - y_min if y_max > y_min else y_max * 0.05
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    
    x_min = min(x_values)
    x_max = max(x_values)
    ax.set_xlim(x_min, x_max)

    # Grade sutil
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#999999")
    ax.set_axisbelow(True)

    # Formatação dos ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

    # Remover bordas superior e direita (estilo acadêmico)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Salvar ────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\nGráfico salvo em: {output_path}")
    plt.close(fig)


def main():
    instance_path = Path(BASE_DIR) / "Data" / INSTANCE_FILE
    print(f"Carregando instância: {instance_path}")

    # Lê os dados da instância
    n_jobs, n_machines, p_times, m_sequence = read_taillard_instance(instance_path)
    print(f"  {n_jobs} jobs × {n_machines} máquinas")

    # Gera a solução inicial (G&T) uma única vez
    initial_seq = generate_gt_sequence(n_jobs, n_machines, p_times, m_sequence)
    
    # Cria a lista de valores do iter_max que serão testados
    iter_max_values = list(range(START_ITER, END_ITER + 1, STEP))
    final_cmax_values = []

    print(f"\nIniciando testes de sensibilidade (iter_max de {START_ITER} a {END_ITER}, passo {STEP})...")
    print("Isso pode levar alguns minutos.\n")

    # Avalia o RNA para cada valor de iter_max
    for imax in iter_max_values:
        # Passa a mesma seed para evitar que ruído atrapalhe a análise de tendência do iter_max
        _, best_cmax = random_non_ascending(
            initial_seq, n_jobs, n_machines, p_times, m_sequence,
            iter_max=imax, seed=SEED
        )
        final_cmax_values.append(best_cmax)
        
        # Print simples para acompanhamento no terminal
        if imax % 100 == 0 or imax == START_ITER:
            print(f"  Testado iter_max = {imax:4d} | C_max resultante = {best_cmax}")

    # Gerar e salvar gráfico
    output_file = os.path.join(BASE_DIR, f"sensibilidade_itermax_{INSTANCE_FILE}.png")
    plot_itermax_sensitivity(iter_max_values, final_cmax_values, INSTANCE_FILE, output_file)


if __name__ == "__main__":
    main()