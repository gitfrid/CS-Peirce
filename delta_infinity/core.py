# delta_infinity/core.py
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

from .models.base import HypothesisModel
from .models.vicsek import NonInteractingModel, VicsekBasicModel
from .detectors.structural import detect_structural_failure
from .utils.io import ensure_dir, save_metrics_csv, save_summary_csv
from .utils.viz import plot_order_parameter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS = {
    'NonInteracting': NonInteractingModel,
    'VicsekBasic': VicsekBasicModel,
    # Hier erweitern: 'VicsekAttraction', 'Kuramoto', etc.
}

class DeltaInfinity:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.hypotheses: List[HypothesisModel] = [
            MODELS[name]() for name in self.config['cycles']['hypotheses']
        ]
        self.sim_kwargs = self.config['simulation']
        self.out_dir = Path(self.config['output']['base_dir'])
        ensure_dir(self.out_dir)
        
        self.cycles_results: List[Dict] = []
    
    def run(self):
        logger.info("Δ∞ Zyklen starten – Hypothesenraum-Erweiterung")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, model in enumerate(self.hypotheses, 1):
            if idx > self.config['cycles']['max_cycles']:
                break
            
            desc = model.describe()
            logger.info(f"Zyklus {idx}: {desc}")
            
            metrics = model.simulate(cycle_num=idx, **self.sim_kwargs)
            
            failure, reasons = detect_structural_failure(metrics)
            
            summary = {
                'cycle': idx,
                'model': desc,
                'avg_order': metrics['avg_order'],
                'failure': failure,
                'reasons': '; '.join(reasons),
                'timestamp': timestamp
            }
            self.cycles_results.append(summary)
            
            logger.info(f"  Avg. Order: {metrics['avg_order']:.4f}")
            if failure:
                logger.warning(f"  Failure: {'; '.join(reasons)} → Hypothesenraum erweitern")
            else:
                logger.info("  Strukturell geschlossen – Hypothese plausibel")
            
            # Speichern
            prefix = self.config['output']['prefix']
            ts_str = f"_{timestamp}" if self.config['output']['timestamp'] else ""
            
            plot_path = self.out_dir / f"{prefix}order_cycle_{idx}{ts_str}.png"
            plot_order_parameter(metrics['orders'], metrics['avg_order'], plot_path, title=desc)
            
            save_metrics_csv(metrics, self.out_dir / f"{prefix}metrics_cycle_{idx}{ts_str}.csv")
            save_summary_csv([summary], self.out_dir / f"{prefix}summary_cycle_{idx}{ts_str}.csv")
            
            if not failure:
                logger.info("→ Erfolgreicher Abschluss – passende Hypothese gefunden.")
                break
        
        # Gesamt-Summary
        pd.DataFrame(self.cycles_results).to_csv(
            self.out_dir / f"{prefix}all_cycles_{timestamp}.csv", index=False
        )
        logger.info("Alle Zyklen abgeschlossen. Ergebnisse in: %s", self.out_dir)