import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

# ── Caminhos dos arquivos ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_VND = os.path.join(BASE_DIR, "resultado_grupo_41_50Raw.txt")
FILE_RNA = os.path.join(BASE_DIR, "resultado_testes.txt")

CPU_LABEL = "CPU: AMD Ryzen 7 2700X"

# ── Parsers ──────────────────────────────────────────────────────────────────
def parse_vnd_file(filepath: str):
    """Lê o arquivo de resultados VND no formato CSV (ponto-e-vírgula)."""
    instances = []
    cmaxes = []
    times = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("AVERAGE"):
                continue
            parts = line.split(";")
            if len(parts) >= 3:
                instances.append(parts[0])
                cmaxes.append(int(parts[1]))
                times.append(float(parts[2].rstrip("s")))
    return instances, cmaxes, times


def parse_rna_file(filepath: str):
    """Lê o arquivo de resultados RNA no formato tabular do teste estatístico."""
    instances = []
    cmaxes = []
    times = []
    in_target_group = False
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if "Grupo: ta41-50" in line:
                in_target_group = True
                continue
            if in_target_group:
                # Se encontra o início de outro grupo, para de ler
                if "Grupo:" in line and "ta41-50" not in line:
                    in_target_group = False
                    continue
                if line.startswith("ta") and not (line.startswith("ta7") or line.startswith("ta8")):
                    parts = line.split()
                    if len(parts) >= 4:
                        instances.append(parts[0])
                        cmaxes.append(int(parts[1]))
                        times.append(float(parts[3].rstrip("s")))
    return instances, cmaxes, times


# ── Carregamento dos dados ────────────────────────────────────────────────────
inst_vnd, cmax_vnd, time_vnd = parse_vnd_file(FILE_VND)
inst_rna, cmax_rna, time_rna = parse_rna_file(FILE_RNA)

# Garante que as instâncias coincidem
assert inst_vnd == inst_rna, f"Instâncias não coincidem: {inst_vnd} vs {inst_rna}"

# ── Cores ─────────────────────────────────────────────────────────────────────
COLOR_VND    = "#4F81E8"  # Azul elegante
COLOR_RNA    = "#F5C542"  # Dourado vibrante
BG_COLOR     = "#0F1117"
GRID_COLOR   = "#2A2D3A"
TEXT_COLOR   = "#E8ECF4"
SUBTITLE_CLR = "#8B90A8"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "legend.facecolor": "#1C1F2E",
    "legend.edgecolor": GRID_COLOR,
})


def generate_bar_chart(instances, values_vnd, values_rna, title, y_label, out_filename, val_fmt, chart_type=None):
    """Renderiza e salva um gráfico de barras agrupadas comparando VND e RNA."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG_COLOR)

    # ── Bloco de título ──────────────────────────────────────────────────────
    fig.text(0.5, 0.97, title,
             ha="center", va="top",
             fontsize=15, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.91, CPU_LABEL,
             ha="center", va="top",
             fontsize=10, color=SUBTITLE_CLR)

    # ── Barras ───────────────────────────────────────────────────────────────
    n     = len(instances)
    x     = np.arange(n)
    bar_w = 0.38
    gap   = 0.04

    ax.bar(x - bar_w / 2 - gap / 2, values_vnd,
           width=bar_w, color=COLOR_VND, alpha=0.88, linewidth=0, zorder=3)
    ax.bar(x + bar_w / 2 + gap / 2, values_rna,
           width=bar_w, color=COLOR_RNA, alpha=0.88, linewidth=0, zorder=3)

    # ── Rótulos de valor ─────────────────────────────────────────────────────
    all_max = max(max(values_vnd), max(values_rna))
    offset  = all_max * 0.012

    vnd_xs = x - bar_w / 2 - gap / 2
    rna_xs  = x + bar_w / 2 + gap / 2

    # Rótulos para VND
    for i, (xi, val) in enumerate(zip(vnd_xs, values_vnd)):
        label_text = val_fmt(val)
        if chart_type == "cmax" and val < values_rna[i]:
            diff = val - values_rna[i]
            diff_pct = (diff / values_rna[i]) * 100
            label_text = f"{val}\n{diff} ({diff_pct:.1f}%)"
            ax.text(xi, val + offset, label_text,
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=COLOR_VND)
        else:
            ax.text(xi, val + offset, label_text,
                    ha="center", va="bottom",
                    fontsize=8, color=TEXT_COLOR)

    # Rótulos para RNA
    for i, (xi, val) in enumerate(zip(rna_xs, values_rna)):
        if chart_type == "time":
            speedup = values_vnd[i] / val
            label_text = f"{val_fmt(val)}\n{speedup:,.0f}x"
            ax.text(xi, val + offset, label_text,
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=COLOR_RNA)
        elif chart_type == "cmax" and val < values_vnd[i]:
            diff = val - values_vnd[i]
            diff_pct = (diff / values_vnd[i]) * 100
            label_text = f"{val}\n{diff} ({diff_pct:.1f}%)"
            ax.text(xi, val + offset, label_text,
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=COLOR_RNA)
        else:
            ax.text(xi, val + offset, val_fmt(val),
                    ha="center", va="bottom",
                    fontsize=8, color=TEXT_COLOR)

    # ── Eixos ────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=30, ha="right", fontsize=9)

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Instância", labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    
    # Limites do eixo Y com margem para os rótulos
    ax.set_ylim(0, all_max * 1.15)

    # ── Legenda ──────────────────────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=COLOR_VND, alpha=0.88, label="VND"),
        Patch(facecolor=COLOR_RNA, alpha=0.88, label="RNA (Melhor Execução)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=10, framealpha=0.75)

    # ── Salvar ───────────────────────────────────────────────────────────────
    plt.subplots_adjust(top=0.83)
    out_path = os.path.join(BASE_DIR, out_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("Salvo ->", out_path)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Gráfico 1 — Tempo de Execução
# ════════════════════════════════════════════════════════════════════════════
generate_bar_chart(
    instances    = inst_vnd,
    values_vnd   = time_vnd,
    values_rna   = time_rna,
    title        = "Instâncias ta41-50 — Comparação de Tempo de Execução",
    y_label      = "Tempo de Execução (s)",
    out_filename = "comparacao_tempo_ta41_50.png",
    val_fmt      = lambda t: f"{t:.2f}s" if t >= 0.1 else f"{t:.3f}s",
    chart_type   = "time"
)

# ════════════════════════════════════════════════════════════════════════════
# Gráfico 2 — Resultado / Makespan (C_max)
# ════════════════════════════════════════════════════════════════════════════
generate_bar_chart(
    instances    = inst_vnd,
    values_vnd   = cmax_vnd,
    values_rna   = cmax_rna,
    title        = "Instâncias ta41-50 — Comparação de Makespan (C_max)",
    y_label      = "Makespan (C_max)",
    out_filename = "comparacao_resultado_ta41_50.png",
    val_fmt      = lambda c: f"{c}",
    chart_type   = "cmax"
)
