"""
Figura 1 — Pirámide del mercado wealth management español.
Sección 6 (Análisis de Negocio) — TFG Kairos.

Genera:
    docs/figures/business/fig1_piramide_mercado.png  (300 DPI)
    docs/figures/business/fig1_piramide_mercado.pdf  (vectorial)

Estilo: Okabe-Ito (colorblind-safe), Tufte, sans-serif, español.
Autor: Mateo Madrigal Arteaga, UFV
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ── Paleta Okabe-Ito ──────────────────────────────────────────────
OKABE = {
    "azul":     "#0072B2",
    "naranja":  "#E69F00",
    "verde":    "#009E73",
    "amarillo": "#F0E442",
    "rojo":     "#D55E00",
    "rosa":     "#CC79A7",
    "celeste":  "#56B4E9",
    "gris":     "#999999",
}

HIGHLIGHT = OKABE["naranja"]
TEXTO = "#222222"
ANOT = "#555555"

OUTDIR = "docs/figures/business"
os.makedirs(OUTDIR, exist_ok=True)


def trapecio(ax, y_bot, y_top, w_bot, w_top, color, alpha=0.95, edge="white", lw=2.0):
    """Dibuja un trapecio horizontal centrado en x=0."""
    pts = [
        (-w_bot/2, y_bot),
        ( w_bot/2, y_bot),
        ( w_top/2, y_top),
        (-w_top/2, y_top),
    ]
    poly = patches.Polygon(pts, closed=True, facecolor=color,
                           edgecolor=edge, linewidth=lw, alpha=alpha)
    ax.add_patch(poly)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "text.color": TEXTO,
    })

    # Lienzo 16:9 panorámico
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="white")

    # ── Título y subtítulo (figtext, separación generosa) ──────────
    fig.text(0.5, 0.955,
             "Mercado español de wealth management",
             ha="center", va="center", fontsize=20, fontweight="bold", color=TEXTO)
    fig.text(0.5, 0.910,
             "Segmentación por AUM y stack tecnológico",
             ha="center", va="center", fontsize=14, color=ANOT)
    fig.text(0.5, 0.870,
             "Kairos se posiciona en el segmento medio (EAF / MFO 100–2.000 M€), "
             "no atendido por las plataformas institucionales",
             ha="center", va="center", fontsize=11, fontstyle="italic", color=ANOT)

    # ── Eje principal — sin aspect='equal', llena el lienzo ────────
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.78])
    # xlim asimétrico: más espacio a la derecha para el callout
    ax.set_xlim(-8.5, 13.5)
    ax.set_ylim(0, 4.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Geometría de la pirámide ───────────────────────────────────
    # Pirámide centrada en x=0; base 13, altura total 4
    levels = [
        # (y_bot, y_top, w_bot, w_top)
        (3.0, 4.0, 5.0, 3.0),    # Nivel 1: cúspide
        (2.0, 3.0, 7.5, 5.0),    # Nivel 2
        (1.0, 2.0, 10.0, 7.5),   # Nivel 3 (HIGHLIGHT)
        (0.0, 1.0, 13.0, 10.0),  # Nivel 4: base
    ]
    colores = [
        OKABE["azul"],     # 1
        OKABE["celeste"],  # 2
        HIGHLIGHT,         # 3 — Kairos target
        OKABE["gris"],     # 4
    ]

    for (yb, yt, wb, wt), c in zip(levels, colores):
        trapecio(ax, yb, yt, wb, wt, c)

    # ── Etiquetas dentro de cada nivel ─────────────────────────────
    # Nivel 1 — Banca privada de marca (cúspide, dos líneas para que entre)
    ax.text(0, 3.66, "Banca privada de marca",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(0, 3.45, "AUM 2024: 922.354 M€",
            ha="center", va="center", fontsize=10, color="white")
    ax.text(0, 3.25, "Stack: Aladdin · Bloomberg · FactSet",
            ha="center", va="center", fontsize=9, fontstyle="italic", color="white")

    # Nivel 2 — SGIIC
    ax.text(0, 2.62, "SGIIC y agencias de valores",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(0, 2.38, "Stack: research interno · FactSet · Morningstar Direct",
            ha="center", va="center", fontsize=10, fontstyle="italic", color="white")

    # Nivel 3 — EAF / MFO mid-size (HIGHLIGHT, pieza clave)
    ax.text(0, 1.72, "EAF y multi-family offices mid-size",
            ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    ax.text(0, 1.46, "AUM 2024: 17.122 M€   ·   140 entidades   ·   +8,65 % YoY",
            ha="center", va="center", fontsize=11.5, color="white")
    ax.text(0, 1.22, "Stack actual: Excel + research papers + screeners",
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color="white")

    # Nivel 4 — DIY
    ax.text(0, 0.62, "Inversores DIY (robo-advisors, broker online)",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(0, 0.38, "Stack: app móvil",
            ha="center", va="center", fontsize=10, fontstyle="italic", color="white")

    # ── Callout Kairos (a la derecha del nivel 3) ─────────────────
    # Borde derecho del nivel 3 a y=1.5: ((10+7.5)/2)/2 = 4.375
    ax.annotate(
        "Posicionamiento de Kairos:\n"
        "segmento desatendido por\n"
        "Bloomberg / FactSet / Aladdin\n"
        "por motivos económicos\n"
        "estructurales",
        xy=(4.4, 1.5),
        xytext=(6.5, 1.5),
        ha="left", va="center",
        fontsize=11, color=TEXTO, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white",
                  edgecolor=HIGHLIGHT, linewidth=2.0),
        arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT, lw=2.2,
                        connectionstyle="arc3,rad=0.0",
                        mutation_scale=20),
    )

    # ── Fuente al pie ──────────────────────────────────────────────
    fig.text(0.5, 0.022,
             "Fuente: FundsPeople XI Ranking Banca Privada (2025); CNMV Informe Anual 2024; "
             "EAF-CGE Informa nº 30 (jul-2025). Elaboración propia.",
             ha="center", va="bottom", fontsize=9.5, color=ANOT, fontstyle="italic")

    out_png = os.path.join(OUTDIR, "fig1_piramide_mercado.png")
    out_pdf = os.path.join(OUTDIR, "fig1_piramide_mercado.pdf")
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    print(f"✓ Figura 1 guardada en:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
