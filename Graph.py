import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from matplotlib.patches import Patch

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
FILE_41_CPP     = os.path.join(BASE_DIR, "resultado_grupo_41_50.txt")
FILE_41_PYTHON  = os.path.join(BASE_DIR, "resultado_grupo_41_50Raw.txt")
FILE_71_80      = os.path.join(BASE_DIR, "resultado_grupo_71_80.txt")

CPU_LABEL = "CPU: AMD Ryzen 7 2700X"


def parse_result_file(filepath: str):
    instances: list[str] = []
    times:     list[float] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if parts[0].upper() == "AVERAGE":
                continue
            instances.append(parts[0])
            times.append(float(parts[2].rstrip("s")))
    return instances, times


# ── Parse files ───────────────────────────────────────────────────────────────
inst_41,  time_cpp    = parse_result_file(FILE_41_CPP)
_,        time_python = parse_result_file(FILE_41_PYTHON)
inst_71,  time_71     = parse_result_file(FILE_71_80)

# ── Slowdown factor (Python / C++) from ta41-50 ───────────────────────────────
slowdown_factors      = [py / cpp for py, cpp in zip(time_python, time_cpp)]
avg_factor            = sum(slowdown_factors) / len(slowdown_factors)
time_71_python_est    = [t * avg_factor for t in time_71]

# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_CPP    = "#4F81E8"
COLOR_PYTHON = "#F5C542"
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


def _make_graph(instances, times_cpp, times_py,
                title, py_label, py_hatch, out_filename,
                y_fmt_py_as_hours=False):
    """Render and save a single grouped-bar chart."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG_COLOR)

    # ── Title block ──────────────────────────────────────────────────────────
    fig.text(0.5, 0.97, title,
             ha="center", va="top",
             fontsize=15, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.91, CPU_LABEL,
             ha="center", va="top",
             fontsize=10, color=SUBTITLE_CLR)

    # ── Bars ─────────────────────────────────────────────────────────────────
    n     = len(instances)
    x     = np.arange(n)
    bar_w = 0.38
    gap   = 0.04

    ax.bar(x - bar_w / 2 - gap / 2, times_cpp,
           width=bar_w, color=COLOR_CPP, alpha=0.88, linewidth=0, zorder=3)
    ax.bar(x + bar_w / 2 + gap / 2, times_py,
           width=bar_w, color=COLOR_PYTHON, alpha=0.88,
           linewidth=1.2 if py_hatch else 0,
           edgecolor=TEXT_COLOR if py_hatch else "none",
           hatch=py_hatch, zorder=3)

    # ── Value labels ─────────────────────────────────────────────────────────
    all_max = max(max(times_cpp), max(times_py))
    offset  = all_max * 0.012

    def _label(bars_x, times, fmt):
        for xi, t in zip(bars_x, times):
            ax.text(xi, t + offset, fmt(t),
                    ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_COLOR)

    cpp_xs = x - bar_w / 2 - gap / 2
    py_xs  = x + bar_w / 2 + gap / 2
    _label(cpp_xs, times_cpp, lambda t: f"{t:.1f}s")
    if y_fmt_py_as_hours:
        _label(py_xs, times_py, lambda t: f"{t/3600:.1f}h")
    else:
        _label(py_xs, times_py, lambda t: f"{t:.1f}s")

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=30, ha="right", fontsize=9)

    if y_fmt_py_as_hours:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v/3600:.1f}h" if v >= 3600 else f"{v:.0f}s"
        ))
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f s"))

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Instance", labelpad=8)
    ax.set_ylabel("Execution Time", labelpad=8)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=COLOR_CPP,    alpha=0.88, label="C++"),
        Patch(facecolor=COLOR_PYTHON, alpha=0.88, label=py_label,
              hatch=py_hatch,
              linewidth=1.2 if py_hatch else 0,
              edgecolor=TEXT_COLOR if py_hatch else "none"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=10, framealpha=0.75)

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.subplots_adjust(top=0.83)
    out_path = os.path.join(BASE_DIR, out_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("Saved ->", out_path)
    plt.show()
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Graph 1 — ta41-50  (real C++ and real Python)
# ════════════════════════════════════════════════════════════════════════════
_make_graph(
    instances       = inst_41,
    times_cpp       = time_cpp,
    times_py        = time_python,
    title           = "Instances ta41-50  —  Execution Time",
    py_label        = "Python",
    py_hatch        = "",
    out_filename    = "graph_ta41_50.png",
    y_fmt_py_as_hours = False,
)

# ════════════════════════════════════════════════════════════════════════════
# Graph 2 — ta71-80  (real C++, Python estimated ~167x)
# ════════════════════════════════════════════════════════════════════════════
_make_graph(
    instances       = inst_71,
    times_cpp       = time_71,
    times_py        = time_71_python_est,
    title           = f"Instances ta71-80  —  Execution Time",
    py_label        = f"Python (~{avg_factor:.0f}x)",
    py_hatch        = "//",
    out_filename    = "graph_ta71_80.png",
    y_fmt_py_as_hours = True,
)
