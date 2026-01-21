from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_phase_2d(y: np.ndarray, i: int, j: int, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.set_dpi(150)
    ax.plot(y[i, :], y[j, :], linewidth=0.07)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.3)
    ax.set_aspect("equal", adjustable="box")
    return fig


def plot_phase_3d(y: np.ndarray, i: int, j: int, k: int, title: str, labels: Tuple[str, str, str]):
    fig = plt.figure(figsize=(3.2, 3.2))
    fig.set_dpi(150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y[i, :], y[j, :], y[k, :], linewidth=0.07)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(labels[0], fontsize=9)
    ax.set_ylabel(labels[1], fontsize=9)
    ax.set_zlabel(labels[2], fontsize=9)
    ax.tick_params(labelsize=8)
    return fig


def plot_time_seiries_functional(t: np.ndarray, y: np.ndarray, indices: List[int], var_names: List[str], title: str):
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    fig.set_dpi(140)
    for i in indices:
        ax.plot(t, y[i, :], linewidth=0.9, label=var_names[i])
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    return fig


def plot_time_series(t: np.ndarray, y: np.ndarray, indices: List[int], var_names: List[str], title: str):
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    fig.set_dpi(140)
    for i in indices:
        ax.plot(t, y[i, :], linewidth=0.9, label=var_names[i])
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    return fig
