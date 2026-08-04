"""Matplotlib helpers for benchmark visualisation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numdiff as nd


def bar_chart_compare(
    data: dict[str, list[float]],
    labels: list[str],
    title: str = "",
    ylabel: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart comparing multiple series.

    Parameters
    ----------
    data : dict[str, list[float]]
        Mapping from group label to list of bar values.
    labels : list[str]
        Labels for the x-axis categories.
    title : str
        Chart title.
    ylabel : str
        Y-axis label.

    Returns
    -------
    tuple[Figure, Axes]
    """
    n_groups = len(labels)
    n_series = len(data)
    width = 0.8 / n_series

    x = nd.arange(n_groups)
    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.5), 4))

    for i, (name, values) in enumerate(data.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar([float(v + offset) for v in x], values, width, label=name)

    ax.set_xticks([float(v) for v in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.show()
    return fig, ax


def summary_table(
    data: dict[str, list[Any]],
    rows: list[str],
    cols: list[str],
) -> tuple[plt.Figure, plt.Axes]:
    """Render a summary table via matplotlib.

    Parameters
    ----------
    data : dict[str, list[Any]]
        Column-oriented data. Each key is a column header, each value a
        list of cell values.
    rows : list[str]
        Row labels.
    cols : list[str]
        Column headers (must match ``data`` keys).

    Returns
    -------
    tuple[Figure, Axes]
    """
    n_rows = len(rows)
    n_cols = len(cols)

    fig, ax = plt.subplots(figsize=(n_cols * 2.5, n_rows * 0.5 + 1))
    ax.axis("off")

    table_data = []
    table_data.append(cols)
    for i in range(n_rows):
        table_data.append([data[c][i] for c in cols])

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        rowLabels=rows,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    plt.show()
    return fig, ax


def heatmap(
    data: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Method x backend heatmap.

    Parameters
    ----------
    data : list[list[float]]
        2-D array of values, shape ``(n_rows, n_cols)``.
    row_labels : list[str]
        Labels for rows (e.g. method names).
    col_labels : list[str]
        Labels for columns (e.g. backend names).
    title : str
        Chart title.

    Returns
    -------
    tuple[Figure, Axes]
    """
    arr = nd.array(data)
    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.5, len(row_labels) * 1.2))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(
                j, i, f"{float(arr[i, j]):.3g}", ha="center", va="center", fontsize=8
            )

    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.show()
    return fig, ax
