# utils/io.py
from pathlib import Path
import pandas as pd

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_metrics_csv(metrics: dict, path: Path):
    df = pd.DataFrame({
        'timestep': range(len(metrics['orders'])),
        'order_parameter': metrics['orders']
    })
    df['avg_stationary'] = metrics['avg_order']
    df.to_csv(path, index=False)

def save_summary_csv(summaries: list, path: Path):
    pd.DataFrame(summaries).to_csv(path, index=False)