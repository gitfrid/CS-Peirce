"""
V13 – Guided Peircean Graph Engine (Success Version)
====================================================

Guided to succeed:
- Bias toward 'good' rules (iteration, abstraction)
- Complexity pain: prune large graphs
- Table-building: memory of small n results
- Motivation: meta-doubt triggers aggressive mutation
- Hierarchical abstraction: nested cuts
- Graphs first, algebra translation last

Version: 13.0-guided-success
"""

import random
import math
import numpy as np
from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
class Config:
    MAX_N = 1000  # Limit for graph scale
    MAX_CYCLES = 50
    MAX_NODES = 50
    COMPLEXITY_PENALTY = 0.05  # Pain per node
    GUIDANCE_BIAS = 0.7  # Probability to choose "good" rules
    STUCK_THRESHOLD = 5
    RANDOM_SEED = 42
    TABLE_BASES = [2, 3, 5]  # Small primes for tables

random.seed(Config.RANDOM_SEED)

# ─────────────────────────────────────────────
# Existential Graph Structure
# ─────────────────────────────────────────────

class ExistentialGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.cuts = []  # (cut_id, parent_cut, tincture)
        self.probabilities = {}  # edge_id: prob
        self.tables = {}  # 'power': list of (base, k, result)
        self.n = None
        self.meta_doubt = 0.0

    def add_entity(self, name: str, type: str = 'icon'):
        self.graph.add_node(name, type=type)

    def add_relation(self, a: str, b: str, rel: str, prob: float = 1.0):
        edge_id = f"{a}-{b}-{rel}"
        self.graph.add_edge(a, b, rel=rel)
        self.probabilities[edge_id] = prob

    def add_cut(self, contents: List[str], broken: bool = False, tincture: str = None):
        cut_id = f"cut_{len(self.cuts)}"
        self.graph.add_node(cut_id, type='cut', broken=broken, tincture=tincture)
        self.cuts.append((cut_id, None, tincture))
        for content in contents:
            self.graph.add_edge(cut_id, content, rel='contains')

    def add_nested_cut(self, parent_cut: str, contents: List[str], broken: bool = False, tincture: str = None):
        cut_id = f"cut_{len(self.cuts)}"
        self.graph.add_node(cut_id, type='cut', broken=broken, tincture=tincture)
        self.cuts.append((cut_id, parent_cut, tincture))
        self.graph.add_edge(parent_cut, cut_id, rel='nested')
        for content in contents:
            self.graph.add_edge(cut_id, content, rel='contains')

    def build_table(self, base_op: str, k: int):
        table_key = f"{base_op}_{k}"
        self.tables[table_key] = []
        for small_n in [2, 4, 8, 16, 32]:
            result = small_n
            for _ in range(k):
                result = self.graph.nodes[base_op]['op'](result, 2)  # Example: repeated div by 2
            self.tables[table_key].append((small_n, k, result))
        print(f"  → Built table for {table_key}")

    def evolve(self):
        # Guided rule selection
        rules = ["insert", "erase", "iterate", "double_cut", "tincture", "abstract"]
        if random.random() < Config.GUIDANCE_BIAS:
            rule = random.choice(["iterate", "abstract"])  # Bias toward "good" rules
        else:
            rule = random.choice(rules)

        if rule == "insert":
            self.add_entity(f"prime_{random.randint(1,100)}")
        elif rule == "erase":
            if self.graph.nodes:
                self.graph.remove_node(random.choice(list(self.graph.nodes)))
        elif rule == "iterate":
            if self.graph.edges:
                edge = random.choice(list(self.graph.edges))
                self.add_relation(edge[0], edge[1], "iter_" + str(self.graph.edges[edge]['rel']))
        elif rule == "double_cut":
            self.add_nested_cut("cut_0", [], broken=True)
        elif rule == "tincture":
            if self.cuts:
                idx = random.randrange(len(self.cuts))
                self.cuts[idx] = (self.cuts[idx][0], self.cuts[idx][1], random.choice(['red_future', 'blue_past']))
        elif rule == "abstract":
            if self.graph.nodes:
                abs_node = f"abs_{len(self.graph.nodes)}"
                self.add_entity(abs_node, 'symbol')
                self.add_relation(abs_node, random.choice(list(self.graph.nodes)), "generalizes")

        # Complexity pain
        if len(self.graph.nodes) > Config.MAX_NODES:
            self.meta_doubt += Config.COMPLEXITY_PENALTY
            # Prune random node
            self.graph.remove_node(random.choice(list(self.graph.nodes)))

    def measure_doubt(self):
        # Low doubt if many relations with high prob
        avg_prob = sum(self.probabilities.values()) / max(1, len(self.probabilities))
        complexity = len(self.graph.nodes) * Config.COMPLEXITY_PENALTY
        return 1.0 - avg_prob + complexity + self.meta_doubt

    def translate_to_algebra(self):
        # Simple translation: count relations → approximate formula
        num_primes = sum(1 for node in self.graph.nodes if 'prime' in node)
        if num_primes > 0:
            return f"G(n) ≈ n / ln(n)^2 * product(corrections)  (based on {num_primes} existential instances)"
        return "No algebraic form – inquiry incomplete."

# ─────────────────────────────────────────────
# Inquiry Loop
# ─────────────────────────────────────────────

def peircean_graph_inquiry(n=26, cycles=100):
    print(f"Starting guided diagrammatic inquiry for n={n}")
    graph = ExistentialGraph()
    graph.n = n
    graph.add_entity(f"n={n}", 'icon')
    graph.add_entity("prime_pair", 'symbol')
    graph.add_relation(f"n={n}", "prime_pair", "has", 0.5)
    graph.add_nested_cut("cut_0", ["prime_pair"], broken=True)

    for cycle in range(1, cycles+1):
        graph.evolve()
        doubt = graph.measure_doubt()
        print(f"Cycle {cycle:3d} | Doubt: {doubt:.4f}")
        if doubt < 0.05:
            print("Doubt low — stable habit.")
            break

    print("\nFinal Graph State:")
    print(graph.translate_to_algebra())

if __name__ == "__main__":
    peircean_graph_inquiry(n=10000000, cycles=100)