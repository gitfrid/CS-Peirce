# delta_infinity/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class HypothesisModel(ABC):
    """
    Abstract base class for all hypothesis models in the Δ∞ framework.

    Provides a shared method for consistent parameter extraction.
    Subclasses should implement simulate() and describe().
    """

    def _extract_simulation_params(self, kwargs: Dict[str, Any]) -> Tuple:
        """
        Extrahiert die Simulationsparameter aus kwargs.

        Returns:
            Tuple: (N, L, v0, eta, steps, seed, r0)
                   r0 hat Default-Wert 1.0, wenn nicht angegeben

        Raises:
            KeyError: wenn ein erforderlicher Parameter fehlt
        """
        required_keys = ['N', 'L', 'v0', 'eta', 'steps', 'seed']

        params = []
        for key in required_keys:
            if key not in kwargs:
                raise KeyError(f"Fehlender Simulations-Parameter: '{key}'")
            params.append(kwargs[key])

        # Optionaler Parameter r0 (wird in Vicsek benötigt, in anderen ignoriert)
        r0 = kwargs.get('r0', 1.0)
        params.append(r0)

        return tuple(params)

    @abstractmethod
    def simulate(self, cycle_num: int, **kwargs) -> Dict[str, Any]:
        """
        Führt die Simulation für diesen Hypothesenraum durch.

        Args:
            cycle_num: Aktuelle Zyklus-Nummer (für Reproduzierbarkeit / Logging)
            **kwargs: Simulationsparameter aus config

        Returns:
            Dict mit mindestens:
                'orders': list[float]          # Zeitreihe des Ordnungsparameters
                'avg_order': float             # Mittelwert im stationären Bereich
                Optional: 'last_pos', 'last_orient', etc.
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """
        Kurze textuelle Beschreibung des Hypothesenraums / Modells.
        Wird im Logging und in CSV-Dateien verwendet.
        """
        pass