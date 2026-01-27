# delta_infinity/models/vicsek.py
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sparse
from typing import Dict, Any

from .base import HypothesisModel


class NonInteractingModel(HypothesisModel):
    """
    Hypothese ohne jegliche Interaktion:
    Partikel ändern ihre Orientierung nur durch unkorreliertes Rauschen (diffusive rotation).
    Keine Ausrichtung, kein Vicsek-Mechanismus → erwartet kein kollektives Verhalten.
    """
    def simulate(self, cycle_num: int, **kwargs) -> Dict[str, Any]:
        N, L, v0, eta, steps, seed, r0 = self._extract_simulation_params(kwargs)
        # r0 wird hier nicht verwendet

        np.random.seed(seed + cycle_num)

        pos = np.random.uniform(0, L, (N, 2))
        orient = np.random.uniform(-np.pi, np.pi, N)
        orders = []

        for _ in range(steps):
            cos = np.cos(orient)
            sin = np.sin(orient)

            # Bewegung
            pos += np.column_stack([cos, sin]) * v0
            pos %= L

            # Reine diffusive Änderung der Orientierung
            orient += eta * np.random.uniform(-np.pi, np.pi, N)

            # Polarer Ordnungsparameter
            mean_complex = np.mean(np.exp(1j * orient))
            psi = np.abs(mean_complex)
            orders.append(psi)

        avg_order = np.mean(orders[-200:]) if len(orders) >= 200 else np.mean(orders)

        return {
            'orders': orders,
            'avg_order': avg_order,
            # Optional – falls du später speichern möchtest
            'last_pos': pos.copy(),
            'last_orient': orient.copy()
        }

    def describe(self) -> str:
        return "Keine Interaktion: Reine diffusive Orientierungsänderung (keine Thirdness/Mediation)"


class VicsekBasicModel(HypothesisModel):
    """
    Klassisches Vicsek-Modell (2005-Version):
    Lokale Mittelung der Nachbar-Orientierungen + Rauschen.
    Erwartet bei niedrigem eta und ausreichender Dichte kollektive Ausrichtung.
    """
    def simulate(self, cycle_num: int, **kwargs) -> Dict[str, Any]:
        N, L, v0, eta, steps, seed, r0 = self._extract_simulation_params(kwargs)
        # r0 = Interaktionsradius

        np.random.seed(seed + cycle_num)

        pos = np.random.uniform(0, L, (N, 2))
        orient = np.random.uniform(-np.pi, np.pi, N)
        orders = []

        for t in range(steps):
            # Nachbarn finden (periodische Randbedingungen)
            tree = cKDTree(pos, boxsize=[L, L])
            dist_mat = tree.sparse_distance_matrix(tree, r0, output_type='coo_matrix')

            # Komplexe Summe der Orientierungen der Nachbarn
            data = np.exp(1j * orient[dist_mat.col])
            neigh = sparse.coo_matrix((data, (dist_mat.row, dist_mat.col)), shape=dist_mat.shape)
            S = np.squeeze(np.asarray(neigh.sum(axis=1)))

            # Neue Orientierung = Mittelwert + Rauschen
            orient = np.angle(S) + eta * np.random.uniform(-np.pi, np.pi, N)

            cos = np.cos(orient)
            sin = np.sin(orient)

            # Bewegung
            pos += np.column_stack([cos, sin]) * v0
            pos %= L

            # Ordnungsparameter
            mean_complex = np.mean(np.exp(1j * orient))
            psi = np.abs(mean_complex)
            orders.append(psi)

        avg_order = np.mean(orders[-200:]) if len(orders) >= 200 else np.mean(orders)

        return {
            'orders': orders,
            'avg_order': avg_order,
            'last_pos': pos.copy(),
            'last_orient': orient.copy()
        }

    def describe(self) -> str:
        return "Vicsek-Basis: Lokale Mittelung der Nachbar-Orientierungen → emergente Alignment-Law (Thirdness)"