# utils/viz.py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_order_parameter(orders: list, avg_order: float, path: Path, title: str):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(orders, lw=1.2, alpha=0.8)
    plt.axhline(avg_order, color='red', ls='--', label=f"Stationär: {avg_order:.4f}")
    plt.xlabel("Zeitschritt")
    plt.ylabel("Polarer Ordnungsparameter |⟨e^{iθ}⟩|")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()