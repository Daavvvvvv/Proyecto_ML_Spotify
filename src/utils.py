"""Helpers de EDA y visualización para el proyecto de clasificación de género musical."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def save_fig(name: str, dpi: int = 150, tight: bool = True) -> Path:
    """Guarda la figura activa en figures/<name>.png y devuelve la ruta."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def quick_overview(df: pd.DataFrame) -> None:
    """Imprime un resumen compacto: shape, duplicados, dtypes, missing y primeras filas."""
    print(f"Shape: {df.shape}")
    print(f"Duplicados exactos: {df.duplicated().sum()}")

    print("\nDtypes:")
    print(df.dtypes.value_counts())

    na = df.isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    if not na.empty:
        print("\nMissing por columna:")
        print(na)
    else:
        print("\nSin valores faltantes.")

    print("\nPrimeras filas:")
    print(df.head())


def plot_target_distribution(
    s: pd.Series,
    title: str = "Distribución del target",
    top_n: int | None = None,
):
    """Barplot horizontal ordenado de una variable categórica."""
    counts = s.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.3)))
    counts.sort_values().plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Número de tracks")
    return fig, ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    title: str = "Matriz de correlaciones (Pearson)",
):
    """Heatmap de correlaciones entre columnas numéricas."""
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title)
    return fig, ax
