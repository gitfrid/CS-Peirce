#!/usr/bin/env python3
"""
Guided Peircean Graph Engine (Version 13.0-guided-success)
Ready-to-run script with:
- Safe node/edge removal and probability cleanup
- Deterministic RNG per graph instance
- build_table accepts a callable
- Reinforcement and normalization of edge probabilities
- Stagnation detector and dynamic doubt target
- Nested/contains edges carry probabilities
- Visualization using explicit Axes to avoid layout warnings

Requirements:
    pip install networkx matplotlib
Run:
    python peircean_graph_engine.py
"""

import random
import logging
from typing import List, Callable, Dict, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------
# Config (tuned for convergence)
# -------------------------
class Config:
    MAX_N = 1000
    MAX_CYCLES = 200
    MAX_NODES = 80
    COMPLEXITY_PENALTY = 0.02   # reduced from 0.05
    GUIDANCE_BIAS = 0.7
    STUCK_THRESHOLD = 6
    RANDOM_SEED = 42
    TABLE_BASES = [2, 3, 5]
    DOUBT_TARGET_BASE = 0.5     # base for dynamic target
    PROB_REINFORCE_STEP = 0.05  # how much to increase prob on reinforcement
    PROB_NORMALIZE_EVERY = 10   # normalize after this many probability entries

# -------------------------
# ExistentialGraph
# -------------------------
class ExistentialGraph:
    def __init__(self, seed: Optional[int] = None):
        self.graph = nx.DiGraph()
        self.cuts: List[Tuple[str, Optional[str], Optional[str]]] = []  # (cut_id, parent_cut, tincture)
        self.probabilities: Dict[str, float] = {}  # edge_id -> prob
        self.tables: Dict[str, List[Tuple[int, int, float]]] = {}
        self.n: Optional[int] = None
        self.meta_doubt: float = 0.0
        self.rng = random.Random(seed if seed is not None else Config.RANDOM_SEED)
        self.history = {"doubt": [], "nodes": [], "edges": []}
        # ensure initial cut_0 exists
        self._ensure_cut0()

    # -------------------------
    # Helpers
    # -------------------------
    def _edge_id(self, a: str, b: str, rel: str) -> str:
        return f"{a}--{rel}--{b}"

    def _safe_new_cut_id(self) -> str:
        return f"cut_{len(self.cuts)}"

    def _ensure_cut0(self) -> str:
        if "cut_0" not in self.graph:
            cut_id = self._safe_new_cut_id()
            self.graph.add_node(cut_id, type="cut", broken=False, tincture=None)
            self.cuts.append((cut_id, None, None))
            logging.debug("Created initial cut: %s", cut_id)
            return cut_id
        return "cut_0"

    # -------------------------
    # Node / edge operations
    # -------------------------
    def add_entity(self, name: str, type: str = "icon", attrs: Optional[dict] = None):
        attrs = dict(attrs or {})
        attrs.setdefault("type", type)
        self.graph.add_node(name, **attrs)
        logging.debug("Added entity: %s type=%s", name, type)
        return name

    def add_relation(self, a: str, b: str, rel: str, prob: float = 0.5):
        if a not in self.graph or b not in self.graph:
            logging.debug("Skipping add_relation: missing node %s or %s", a, b)
            return None
        eid = self._edge_id(a, b, rel)
        # If same relation exists, reinforce its probability
        if eid in self.probabilities:
            old = self.probabilities[eid]
            self.probabilities[eid] = min(1.0, old + Config.PROB_REINFORCE_STEP)
            logging.debug("Reinforced relation %s -> prob=%.4f", eid, self.probabilities[eid])
        else:
            self.graph.add_edge(a, b, rel=rel)
            self.probabilities[eid] = prob
            logging.debug("Added relation: %s prob=%.4f", eid, prob)
        # occasional normalization
        if len(self.probabilities) > 0 and len(self.probabilities) % Config.PROB_NORMALIZE_EVERY == 0:
            self._normalize_probabilities()
        return eid

    def _normalize_probabilities(self):
        if not self.probabilities:
            return
        total = sum(self.probabilities.values())
        if total <= 0:
            return
        for k in list(self.probabilities.keys()):
            self.probabilities[k] = self.probabilities[k] / total
        logging.debug("Normalized probabilities (total now 1.0)")

    # -------------------------
    # Cuts
    # -------------------------
    def add_cut(self, contents: List[str], broken: bool = False, tincture: Optional[str] = None) -> str:
        cut_id = self._safe_new_cut_id()
        self.graph.add_node(cut_id, type="cut", broken=broken, tincture=tincture)
        self.cuts.append((cut_id, None, tincture))
        for content in contents:
            if content in self.graph:
                self.graph.add_edge(cut_id, content, rel="contains")
                eid = self._edge_id(cut_id, content, "contains")
                # default contains probability
                self.probabilities[eid] = self.probabilities.get(eid, 0.5)
        logging.debug("Added cut: %s contents=%s", cut_id, contents)
        return cut_id

    def add_nested_cut(self, parent_cut: str, contents: List[str], broken: bool = False, tincture: Optional[str] = None) -> str:
        if parent_cut not in self.graph:
            logging.debug("Parent cut %s missing; creating top-level cut", parent_cut)
            parent_cut = self.add_cut([], broken=False, tincture=None)
        cut_id = self._safe_new_cut_id()
        self.graph.add_node(cut_id, type="cut", broken=broken, tincture=tincture)
        self.cuts.append((cut_id, parent_cut, tincture))
        self.graph.add_edge(parent_cut, cut_id, rel="nested")
        # add probability for nested relation
        eid_nested = self._edge_id(parent_cut, cut_id, "nested")
        self.probabilities[eid_nested] = self.probabilities.get(eid_nested, 0.1)
        for content in contents:
            if content in self.graph:
                self.graph.add_edge(cut_id, content, rel="contains")
                eid = self._edge_id(cut_id, content, "contains")
                self.probabilities[eid] = self.probabilities.get(eid, 0.5)
        logging.debug("Added nested cut: %s parent=%s contents=%s", cut_id, parent_cut, contents)
        return cut_id

    # -------------------------
    # Table building
    # -------------------------
    def build_table(self, op_callable: Callable[[float, float], float], k: int, table_name: Optional[str] = None):
        if not callable(op_callable):
            raise ValueError("op_callable must be callable")
        table_key = table_name or f"op_{k}"
        self.tables[table_key] = []
        for small_n in [2, 4, 8, 16, 32]:
            result = float(small_n)
            for _ in range(k):
                result = op_callable(result, 2)
            self.tables[table_key].append((small_n, k, result))
        logging.info("Built table %s with k=%d", table_key, k)
        return table_key

    # -------------------------
    # Safe removal and cleanup
    # -------------------------
    def _remove_probabilities_for_node(self, node: str):
        # remove any probability entries that reference node anywhere in the key
        keys_to_remove = [k for k in self.probabilities if node in k]
        for k in keys_to_remove:
            del self.probabilities[k]
            logging.debug("Removed probability entry: %s", k)

    def _remove_probabilities_for_edge(self, a: str, b: str, rel: str):
        eid = self._edge_id(a, b, rel)
        if eid in self.probabilities:
            del self.probabilities[eid]
            logging.debug("Removed probability for edge: %s", eid)

    # -------------------------
    # Evolution rules
    # -------------------------
    def evolve(self):
        rules = ["insert", "erase", "iterate", "double_cut", "tincture", "abstract"]
        if self.rng.random() < Config.GUIDANCE_BIAS:
            rule = self.rng.choice(["iterate", "abstract"])
        else:
            rule = self.rng.choice(rules)
        logging.debug("Selected rule: %s", rule)

        if rule == "insert":
            name = f"prime_{self.rng.randint(1, 100)}"
            self.add_entity(name)
        elif rule == "erase":
            candidates = [n for n in self.graph.nodes if not str(n).startswith("cut_0") and not str(n).startswith("n=")]
            if candidates:
                node = self.rng.choice(candidates)
                self._remove_probabilities_for_node(node)
                self.graph.remove_node(node)
                logging.info("Erased node: %s", node)
        elif rule == "iterate":
            if self.graph.edges:
                edge = self.rng.choice(list(self.graph.edges))
                rel = self.graph.edges[edge].get("rel", "rel")
                # reinforce the original edge's probability if present
                orig_eid = self._edge_id(edge[0], edge[1], rel)
                if orig_eid in self.probabilities:
                    old = self.probabilities[orig_eid]
                    self.probabilities[orig_eid] = min(1.0, old + Config.PROB_REINFORCE_STEP)
                    logging.debug("Iterate reinforced %s -> prob=%.4f", orig_eid, self.probabilities[orig_eid])
                else:
                    self.add_relation(edge[0], edge[1], rel, prob=0.5)
        elif rule == "double_cut":
            self._ensure_cut0()
            self.add_nested_cut("cut_0", [], broken=True)
        elif rule == "tincture":
            if self.cuts:
                idx = self.rng.randrange(len(self.cuts))
                cut_id, parent, _ = self.cuts[idx]
                new_tincture = self.rng.choice(["red_future", "blue_past"])
                self.cuts[idx] = (cut_id, parent, new_tincture)
                if cut_id in self.graph:
                    self.graph.nodes[cut_id]["tincture"] = new_tincture
                logging.info("Tinctured cut %s -> %s", cut_id, new_tincture)
        elif rule == "abstract":
            abs_node = f"abs_{len(self.graph.nodes)}"
            self.add_entity(abs_node, "symbol")
            targets = [n for n in self.graph.nodes if n != abs_node]
            if targets:
                target = self.rng.choice(targets)
                self.add_relation(abs_node, target, "generalizes")

        # Complexity pain and pruning
        if len(self.graph.nodes) > Config.MAX_NODES:
            self.meta_doubt += Config.COMPLEXITY_PENALTY
            candidates = [n for n in self.graph.nodes if not str(n).startswith("cut_0") and not str(n).startswith("n=")]
            if candidates:
                node = self.rng.choice(candidates)
                self._remove_probabilities_for_node(node)
                self.graph.remove_node(node)
                logging.warning("Pruned node due to complexity: %s", node)

    # -------------------------
    # Metrics and translation
    # -------------------------
    def measure_doubt(self) -> float:
        if self.probabilities:
            avg_prob = sum(self.probabilities.values()) / len(self.probabilities)
        else:
            avg_prob = 0.0
        complexity = len(self.graph.nodes) * Config.COMPLEXITY_PENALTY
        doubt = 1.0 - avg_prob + complexity + self.meta_doubt
        # record history
        self.history["doubt"].append(doubt)
        self.history["nodes"].append(len(self.graph.nodes))
        self.history["edges"].append(len(self.graph.edges))
        logging.debug("Measured doubt: %.4f (avg_prob=%.4f complexity=%.4f meta=%.4f)",
                      doubt, avg_prob, complexity, self.meta_doubt)
        return doubt

    def translate_to_algebra(self) -> str:
        num_primes = sum(1 for node in self.graph.nodes if "prime" in str(node))
        num_cuts = sum(1 for _, data in self.graph.nodes(data=True) if data.get("type") == "cut")
        return f"G(n) ≈ n / ln(n)^2 * corrections  (primes={num_primes}, cuts={num_cuts})"

    # -------------------------
    # Visualization helpers
    # -------------------------
    def draw(self, figsize: Tuple[int, int] = (10, 6), with_edge_labels: bool = True):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        pos = nx.spring_layout(self.graph, seed=Config.RANDOM_SEED)
        node_types = nx.get_node_attributes(self.graph, "type")
        colors = []
        for n in self.graph.nodes:
            t = node_types.get(n, "icon")
            if t == "cut":
                colors.append("#ffcccb")
            elif t == "symbol":
                colors.append("#add8e6")
            elif isinstance(n, str) and n.startswith("n="):
                colors.append("#90ee90")
            else:
                colors.append("#f0f0f0")
        nx.draw(self.graph, pos, ax=ax, with_labels=True, node_color=colors, node_size=800, font_size=8)
        if with_edge_labels:
            edge_labels = {(u, v): d.get("rel", "") for u, v, d in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=7, ax=ax)
        ax.set_title("Existential Graph")
        plt.show()

    def plot_history(self):
        if not self.history["doubt"]:
            print("No history to plot.")
            return
        fig = plt.figure(figsize=(9, 4), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.history["doubt"], label="Doubt")
        ax.plot(self.history["nodes"], label="Nodes")
        ax.plot(self.history["edges"], label="Edges")
        ax.set_xlabel("Cycle")
        ax.legend()
        ax.set_title("Evolution History")
        plt.show()

    # -------------------------
    # Debug helpers
    # -------------------------
    def summary(self):
        print("Nodes:", len(self.graph.nodes), "Edges:", len(self.graph.edges))
        types = {}
        for _, d in self.graph.nodes(data=True):
            types[d.get("type", "icon")] = types.get(d.get("type", "icon"), 0) + 1
        print("Node types:", types)
        print("Top probabilities (sorted):")
        for k, v in sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {k}: {v:.4f}")

    def list_edges(self, limit: int = 50):
        for i, (u, v, d) in enumerate(self.graph.edges(data=True)):
            if i >= limit:
                break
            eid = self._edge_id(u, v, d.get("rel", ""))
            print(f"{u} -[{d.get('rel','')}]-> {v}  prob={self.probabilities.get(eid, 0):.4f}")

# -------------------------
# Inquiry loop
# -------------------------
def peircean_graph_inquiry(n: int = 26, cycles: int = 100, seed: Optional[int] = None, visualize: bool = True) -> ExistentialGraph:
    logging.info("Starting guided diagrammatic inquiry for n=%d seed=%s", n, seed)
    graph = ExistentialGraph(seed=seed)
    graph.n = n
    graph.add_entity(f"n={n}", "icon")
    graph.add_entity("prime_pair", "symbol")
    graph.add_relation(f"n={n}", "prime_pair", "has", prob=0.5)
    graph._ensure_cut0()
    graph.add_nested_cut("cut_0", ["prime_pair"], broken=True)

    stuck_count = 0
    prev_doubt = None

    for cycle in range(1, cycles + 1):
        graph.evolve()
        doubt = graph.measure_doubt()
        logging.info("Cycle %3d | Doubt: %.4f | Nodes: %d | Edges: %d", cycle, doubt, len(graph.graph.nodes), len(graph.graph.edges))

        # dynamic target scales with graph size to avoid trivial early stopping
        dynamic_target = Config.DOUBT_TARGET_BASE + 0.01 * len(graph.graph.nodes)

        # stagnation detection
        if prev_doubt is not None and abs(doubt - prev_doubt) < 1e-5:
            stuck_count += 1
        else:
            stuck_count = 0
        prev_doubt = doubt

        if stuck_count >= Config.STUCK_THRESHOLD:
            logging.info("Stagnation detected at cycle %d (no meaningful change). Breaking.", cycle)
            break
        if doubt < dynamic_target:
            logging.info("Doubt low — stable habit reached at cycle %d (dynamic target %.3f).", cycle, dynamic_target)
            break

    logging.info("Final Graph State: %s", graph.translate_to_algebra())
    if visualize:
        try:
            graph.draw()
            graph.plot_history()
        except Exception as e:
            logging.warning("Visualization failed: %s", e)
    return graph

# -------------------------
# Example op callables for build_table
# -------------------------
def div_by(x: float, y: float) -> float:
    return x / y

def subtract_const(x: float, y: float) -> float:
    return x - y

# -------------------------
# Optional parameter sweep helper
# -------------------------
def parameter_sweep(seeds: List[int], penalty_values: List[float], cycles: int = 100):
    results = {}
    for penalty in penalty_values:
        Config.COMPLEXITY_PENALTY = penalty
        for seed in seeds:
            g = peircean_graph_inquiry(n=1000, cycles=cycles, seed=seed, visualize=False)
            results[(penalty, seed)] = g.history
    return results

# -------------------------
# Main guard and simple tests
# -------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Single run (visualized)
    g = peircean_graph_inquiry(n=10000000, cycles=100, seed=Config.RANDOM_SEED, visualize=True)

    # Build a small table using a safe callable
    g.build_table(div_by, k=3, table_name="div_by_3")
    print("Tables built:", list(g.tables.keys()))

    # Sanity checks
    assert "cut_0" in g.graph, "cut_0 should exist"
    assert any("n=" in n for n in g.graph.nodes), "n=... node should exist"
    print("Final algebraic sketch:", g.translate_to_algebra())

    # Optional quick inspection
    print("\n--- Summary ---")
    g.summary()
    print("\n--- Top edges ---")
    g.list_edges(limit=20)

    # Uncomment to run a small parameter sweep (no visualization)
    # seeds = [1, 2, 3]
    # penalties = [0.01, 0.02, 0.03]
    # sweep = parameter_sweep(seeds, penalties, cycles=50)
    # print("Sweep keys:", list(sweep.keys())[:5])

if __name__ == "__main__":
    main()
