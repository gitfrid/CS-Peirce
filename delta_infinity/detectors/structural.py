# delta_infinity/detectors/structural.py
import numpy as np
from typing import Tuple, List

def detect_structural_failure(metrics: dict) -> Tuple[bool, List[str]]:
    orders = np.array(metrics['orders'])
    avg_order = metrics['avg_order']
    reasons = []
    
    failure = False
    
    # 1. Emergenz fehlt (Firstness nicht stabil)
    if avg_order < 0.3:
        failure = True
        reasons.append("Keine stabile kollektive Ordnung (ψ < 0.3) – fehlende Firstness-Patterns")
    
    # 2. Fluktuation zu hoch (Secondness dominiert)
    late_orders = orders[-200:]
    if np.std(late_orders) / np.mean(late_orders) > 0.4:
        failure = True
        reasons.append("Hohe Fluktuation im stationären Zustand – Secondness (Resistance) überwiegt")
    
    # 3. Keine Thirdness (keine Mediation)
    if avg_order < 0.6 and np.max(orders) < 0.7:
        failure = True
        reasons.append("Keine Mediation/Habit – Emergenz bricht nicht durch (Identity-Gap in Richtungskontinuität)")
    
    return failure, reasons