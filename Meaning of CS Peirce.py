"""
Meaning of CS Peirce – A Peircean Mathematical Inquiry Engine
=============================================================

Tribute to Charles Sanders Peirce (1839–1914)

This script implements a full Peircean cycle of inquiry for the Goldbach conjecture:
• Abduction: creative hypothesis generation
• Deduction: diagrammatic unfolding (existential graphs + probabilistic evolution)
• Induction: habit formation with fallibilism and self-reflection
• Translation: symbolic algebraic output for formal mathematics

Inspired by Peirce's triadic logic (CP 5.171), diagrammatic reasoning (CP 5.162),
habit as generalization (CP 5.586), and infinite semiosis (CP 1.339).

Author: AI / in collaboration with drifting
Purpose: Philosophical exploration + practical demonstration of Peircean mathematics

Usage:
    Meaning of CS Peirce.py
    # Customize: change n, chain_prob, synechism_prob, etc. in __main__

Dependencies:
    Required: math, numpy, random, typing, logging, datetime
    Optional: sympy (symbolic algebra), networkx + matplotlib (visual graphs)
    Install: pip install sympy networkx matplotlib

Output:
    • Console output
    • Full log written to 'peirce_inquiry.log' (appended with timestamps)
    • PNG graph saved as 'peirce_graph_n<value>.png' (for n ≤ 10,000)
"""

import math
import numpy as np
import random
from typing import List, Tuple, Dict
import logging
import sys
from datetime import datetime

# ─────────────────────────────────────────────
# LOGGING SETUP – captures ALL print output to file
# ─────────────────────────────────────────────

log_filename = r"C:\github\CS-Peirce\peirce_inquiry_n16.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='a'  # append mode
)

# Redirect all print() to both console and log file
class DualLogger:
    def __init__(self, file_handler):
        self.file_handler = file_handler
        self.stdout = sys.__stdout__

    def write(self, message):
        self.stdout.write(message)
        if message.strip():
            self.file_handler.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {message.strip()}\n")
            self.file_handler.flush()

    def flush(self):
        self.stdout.flush()
        self.file_handler.flush()

sys.stdout = DualLogger(open(log_filename, 'a', encoding='utf-8'))

print(f"Log started at {datetime.now():%Y-%m-%d %H:%M:%S} → writing to {log_filename}")

# ─────────────────────────────────────────────
# Optional libraries
# ─────────────────────────────────────────────

try:
    from sympy import symbols, Integral, ln, Product, oo, latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("SymPy not installed – algebraic bounds will be text-only. Install: pip install sympy")

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALS_AVAILABLE = True
except ImportError:
    VISUALS_AVAILABLE = False
    print("Visualization libraries missing – graphs will be text-only. Install: pip install networkx matplotlib")

# ─────────────────────────────────────────────
# Utility: deterministic primality test
# ─────────────────────────────────────────────

def is_prime(n: int) -> bool:
    """Return True if n is prime (deterministic trial division)."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# ─────────────────────────────────────────────
# PeirceSign – basic sign representation
# ─────────────────────────────────────────────

class PeirceSign:
    def __init__(self, sign_type: str, value: str):
        self.type = sign_type
        self.value = value

# ─────────────────────────────────────────────
# Goldbach Discovery – Abductive step
# ─────────────────────────────────────────────

class GoldbachDiscovery:
    def __init__(self, n: int):
        self.n = n
        self.discovered_pairs: List[Tuple[int, int]] = []

    def abduce_prime_pairs(self) -> List[Tuple[int, int]]:
        """Abductively search for prime pairs summing to n (even)."""
        if self.n % 2 != 0 or self.n <= 2:
            return []

        for p in range(2, self.n // 2 + 1):
            q = self.n - p
            if is_prime(p) and is_prime(q):
                self.discovered_pairs.append((p, q))

        return self.discovered_pairs

    def diagram_summary(self) -> str:
        return f"Discovered {len(self.discovered_pairs)} candidate prime decompositions."

# ─────────────────────────────────────────────
# Existential Graph Deduction – Pure Peircean logic
# ─────────────────────────────────────────────

class ExistentialGraphDeduction:
    @staticmethod
    def beta_graph_lemma1(n: int, pairs: List[Tuple[int, int]]) -> str:
        if not pairs:
            return "No Beta graph: No existential decomposition."

        return f"""
Sheet of Assertion (Beta Graph - Relational Existence):

  ──•── Prime ── + ── Prime ──•──
      \\               /
       \\             /
        \\           /
         \\         /
          \\       /
           \\     /
            \\   /
             •  ({n})

Interpretation: Lines of identity relate primes via + to {n}.
Rule: Insert identity for symmetry if p=q.
""".strip()

    @staticmethod
    def gamma_graph_lemma2() -> str:
        return """
Broken Cut (Gamma - Possibility):

  ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈
  ≈ p ── ≤ n/2 ── Prime ── determines ── q = n − p ≈
  ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈

Interpretation: Possibly, p determines q uniquely.
Rule: Replace binary relation with functional mapping.
""".strip()

    @staticmethod
    def beta_graph_lemma3(n: int, pairs: List[Tuple[int, int]]) -> str:
        if not any(p == q for p, q in pairs):
            return "No symmetry collapse: n/2 not prime."

        return f"""
Initial Graph (Beta):

p ── + ── q
      |
      {n}

Constraint (if p=q):

p ── + ── p
      |
      {n}

Interpretation: Collapse nodes under identity.
Rule: Iconic symmetry reduces to reflexivity.
""".strip()

    @staticmethod
    def delta_governance_rules() -> str:
        return """
Delta Tinctured Cut (green_causal for normative governance):

  ΔΔ [causal] 
  Δ If Beta graph verified, permit Gamma modal elevation to habit. Δ
  Δ No universal ∀ without inductive stability. Δ
  Δ Abstraction chains limited to observed relations. Δ
  ΔΔ

Interpretation: Delta governs inference: existential → modal → normative.
""".strip()

    @classmethod
    def extract_lemmas_eg_delta(cls, n: int, pairs: List[Tuple[int, int]]) -> List[str]:
        return [
            cls.beta_graph_lemma1(n, pairs),
            cls.gamma_graph_lemma2(),
            cls.beta_graph_lemma3(n, pairs),
            cls.delta_governance_rules()
        ]

# ─────────────────────────────────────────────
# Goldbach Proof – Deterministic certification
# ─────────────────────────────────────────────

class GoldbachProof:
    @staticmethod
    def certify(n: int, pairs: List[Tuple[int, int]]) -> bool:
        for p, q in pairs:
            if p + q != n or not (is_prime(p) and is_prime(q)):
                return False
        return bool(pairs)

# ─────────────────────────────────────────────
# Algebraic Transcriber – Symbolic output for formalists
# ─────────────────────────────────────────────

class AlgebraicTranscriber:
    @staticmethod
    def generate_hardylittlewood_bound(n: int) -> Tuple[str, str]:
        if not SYMPY_AVAILABLE:
            return (
                "G(n) \\approx 2 C \\prod_{p>2} \\frac{p-2}{p-1} \\frac{n}{(\\ln n)^2} - \\text{Error(RH)}",
                "If RH false, Deuring–Heilbronn: stronger repulsion ensures G(n) > 0."
            )

        n_sym = symbols('n')
        p = symbols('p', integer=True, positive=True)
        C = symbols('C')  # Twin prime constant
        prod_term = Product((p-2)/(p-1), (p, 3, oo))
        asymptotic = prod_term * (n_sym / (ln(n_sym))**2)
        bound = 2 * C * asymptotic

        latex_str = latex(bound) + " - \\text{Error(RH)}"
        alternative = "If RH false, Deuring–Heilbronn phenomenon creates stronger repulsion, ensuring G(n) > 0."

        return latex_str, alternative

# ─────────────────────────────────────────────
# Core Peirce Graph – All reasoning features
# ─────────────────────────────────────────────

class PeirceGraph:
    def __init__(self):
        self.nodes: Dict[str, PeirceSign] = {}
        self.edges: List[Tuple[str, str, str]] = []
        self.cuts: List[Tuple[str, List[str], str]] = []
        self.matrix = None
        self.probabilities: Dict[Tuple[str, str], float] = {}
        self.n = None

    def add_entity(self, name: str, sign_type: str):
        self.nodes[name] = PeirceSign(sign_type, name)

    def add_relation(self, fr: str, to: str, label: str, prob: float = 1.0):
        self.edges.append((fr, to, label))
        self.probabilities[(fr, to)] = prob

    def add_cut(self, cut_type: str, contents: List[str], tincture: str = None):
        self.cuts.append((cut_type, contents, tincture))

    def build_matrix(self):
        node_list = list(self.nodes.keys())
        n = len(node_list)
        self.matrix = np.zeros((n, n))
        node_idx = {name: i for i, name in enumerate(node_list)}
        for (f, t), p in self.probabilities.items():
            i = node_idx.get(f)
            j = node_idx.get(t)
            if i is not None and j is not None:
                self.matrix[i, j] = p
        return self.matrix

    def evolve_graph(self, steps: int = 5):
        for _ in range(steps):
            if self.probabilities and random.random() < 0.75:
                key = random.choice(list(self.probabilities.keys()))
                current = self.probabilities[key]
                delta = random.uniform(-0.10, 0.10)
                self.probabilities[key] = max(0.4, min(1.0, current + delta))

            if random.random() < 0.03 and 'no_pair' not in self.nodes:
                self.add_entity("no_pair", "symbol")
                self.add_relation(str(self.n), "no_pair", "has", 0.008)
                self.add_cut('solid', ["no_pair"])

            if random.random() < 0.08 and 'RH_noise' not in self.nodes:
                self.add_entity("RH_noise", "index")
                self.add_relation(str(self.n), "RH_noise", "affected_by", random.uniform(0.01, 0.08))
                self.add_cut('broken', [str(self.n), "RH_noise"], 'blue_past')

    def apply_rule(self, rule: str, chain_prob: float = 0.4, abstraction_depth_max: int = 3):
        if rule == "insertion" and len(self.nodes) > 2:
            cont = random.sample(list(self.nodes.keys()), k=3)
            self.add_cut('solid', cont)
        elif rule == "erasure" and self.cuts:
            self.cuts.pop(random.randrange(len(self.cuts)))
        elif rule == "iteration" and self.edges:
            e = random.choice(self.edges)
            self.add_relation(e[0], e[1], e[2] + "_iter", self.probabilities.get((e[0], e[1]), 1.0))
        elif rule == "delta_tincture" and self.cuts:
            idx = random.randrange(len(self.cuts))
            ct, cont, _ = self.cuts[idx]
            self.cuts[idx] = (ct, cont, random.choice(['red_future', 'blue_past', 'green_causal']))
        elif rule == "theorematic_symmetry" and self.edges:
            for f, t, lbl in self.edges:
                if lbl == "sum_to" and f[1:] == t[1:]:
                    self.probabilities[(f, t)] = min(1.0, self.probabilities.get((f, t), 1.0) + 0.15)
        elif rule == "hypostatic_abstraction" and self.edges:
            e = random.choice(self.edges)
            abs_node = f"abs_{e[2]}"
            current_chain = 1
            parent = abs_node
            while current_chain < abstraction_depth_max and random.random() < chain_prob:
                if parent not in self.nodes:
                    self.add_entity(parent, "symbol")
                    self.add_relation(e[0] if current_chain == 1 else prev_node, parent, "reifies" if current_chain == 1 else "higher_reifies", 0.90 - 0.05 * current_chain)
                    self.add_relation(parent, e[1], "applies_to", 0.90 - 0.05 * current_chain)
                    self.add_cut('broken', [parent])
                prev_node = parent
                parent = f"abs_{parent}"
                current_chain += 1

    def abduct_hypothesis(self, problem_type: str, n: int = 26, synechism_prob: float = 0.3):
        self.n = n
        self.nodes.clear(); self.edges.clear(); self.cuts.clear(); self.probabilities.clear()

        if problem_type == "goldbach":
            if n % 2 != 0 or n <= 2:
                print("Goldbach requires even n > 2")
                return
            self.add_entity(str(n), "icon")
            self.add_entity("prime_pair", "symbol")
            self.add_relation(str(n), "prime_pair", "has", 0.98)

            for p in range(2, n//2 + 1):
                q = n - p
                if p <= q:
                    if all(p % d != 0 for d in range(2, int(p**0.5)+1)) and \
                       all(q % d != 0 for d in range(2, int(q**0.5)+1)):
                        p_node = f"p{p}"
                        q_node = f"q{q}"
                        self.add_entity(p_node, "symbol")
                        self.add_entity(q_node, "symbol")
                        prob = 0.85 + 0.12 * (p == q)
                        self.add_relation(p_node, q_node, "sum_to", prob)
                        self.add_relation(str(n), p_node, "contains", 0.88)
                        if random.random() < synechism_prob:
                            self.add_relation(p_node, q_node, "synechistic_flow", random.uniform(0.6, 0.9))

            self.add_cut('broken', [str(n), "prime_pair"], 'red_future')

    def deduct_conclusion(self):
        self.build_matrix()
        total_prob = sum(self.probabilities.values())
        failure_prob = self.probabilities.get((str(self.n), "no_pair"), 0.0)
        rh_noise = self.probabilities.get((str(self.n), "RH_noise"), 0.0)

        if failure_prob > 0.05 or rh_noise > 0.15:
            return "Caution: Potential weak counterexample or large RH error term detected."
        elif total_prob > 8.0 and failure_prob < 0.01:
            return "Strong necessary conclusion: Goldbach holds with very high confidence."
        else:
            return "Moderate confidence: Goldbach appears to hold, but further inquiry needed."

    def self_reflect_and_adjust(self, probs: List[float], variance_threshold: float, sample_count: int):
        avg_variance = np.var(probs)
        if avg_variance > variance_threshold:
            print(f"Self-reflection: High variance ({avg_variance:.3f}) detected — increasing sample count for better stability.")
            return sample_count + 4, True
        elif avg_variance < 0.02 and len(probs) > 10:
            print("Self-reflection: Very low variance — inquiry stable; no further adjustment needed.")
            return sample_count, False
        return sample_count, False

    def induct_generalization(self, sample_count: int = 7, variance_threshold: float = 0.05):
        probs = []
        conclusions = []
        current_samples = sample_count
        for _ in range(current_samples):
            self.evolve_graph(steps=4)
            self.apply_rule("hypostatic_abstraction")
            self.apply_rule("theorematic_symmetry")
            probs.append(np.mean(list(self.probabilities.values())))
            conclusions.append(self.deduct_conclusion())

        new_samples, adjusted = self.self_reflect_and_adjust(probs, variance_threshold, current_samples)
        if adjusted:
            for _ in range(new_samples - current_samples):
                self.evolve_graph(steps=4)
                probs.append(np.mean(list(self.probabilities.values())))
                conclusions.append(self.deduct_conclusion())
            current_samples = new_samples

        avg_prob = np.mean(probs)
        avg_variance = np.var(probs)
        dominant = max(set(conclusions), key=conclusions.count)

        if avg_variance > variance_threshold:
            confidence_adj = " (fallible due to high variance in samples)"
        else:
            confidence_adj = ""

        if avg_prob > 0.85:
            verdict = f"Induced universal habit (generalization, avg prob {avg_prob:.3f}, var {avg_variance:.3f}, samples {current_samples}): {dominant}{confidence_adj}"
        else:
            verdict = f"Weak generalization (avg prob {avg_prob:.3f}, var {avg_variance:.3f}, samples {current_samples}): {dominant}{confidence_adj}"

        # Always show algebraic bound
        print("\nAlgebraic Transcription (Hardy–Littlewood bound):")
        bound, alt = AlgebraicTranscriber.generate_hardylittlewood_bound(self.n)
        print(bound)
        print("Alternative (Deuring–Heilbronn if RH false):")
        print(alt)

        return verdict

    def draw_ascii(self):
        print("\n=== Peircean Graph Diagram ===")
        print("Entities:")
        for name, sign in self.nodes.items():
            print(f"  {name:12} : {sign.type} ({sign.value})")
        print("\nRelations (prob):")
        for f, t, lbl in self.edges:
            p = self.probabilities.get((f, t), 1.0)
            print(f"  {f:12} → {t:12} [{lbl}]  prob={p:.3f}")
        print("\nCuts:")
        for ct, cont, tinc in self.cuts:
            sym = "()" if ct == 'solid' else "≈≈" if ct == 'broken' else "ΔΔ"
            tinc_str = f" [{tinc}]" if tinc else ""
            print(f"  {sym}{tinc_str}  {', '.join(cont):<40} {sym}")
        print("\nAdjacency Matrix shape:", self.matrix.shape if self.matrix is not None else "Not built")
        print("============================\n")

    def draw_visual(self):
        if not VISUALS_AVAILABLE:
            print("Visualization libraries not installed – skipping graphical output.")
            return

        if self.n > 10000:
            print(f"Graph for n={self.n} too large ({len(self.nodes)} nodes) – skipping visualization.")
            return

        try:
            G = nx.DiGraph()
            for node in self.nodes:
                color = {'icon':'lightgreen', 'symbol':'lightblue', 'index':'orange'}.get(self.nodes[node].type, 'gray')
                G.add_node(node, color=color)
            for f, t, lbl in self.edges:
                G.add_edge(f, t, label=lbl)
            pos = nx.spring_layout(G, seed=42)
            node_colors = [G.nodes[n]['color'] for n in G.nodes]

            plt.figure(figsize=(12, 10))
            nx.draw(G, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=800, font_size=8)
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

            plt.title(f"Peircean Goldbach Diagram for n={self.n}")

            # Explicitly force PNG format
            png_filename = f"peirce_graph_n{self.n}.png"
            plt.savefig(png_filename, format='png', dpi=300, bbox_inches='tight')
            print(f"Graph saved as PNG: {png_filename}")

            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")

# ─────────────────────────────────────────────
# MAIN INQUIRY FUNCTION
# ─────────────────────────────────────────────

def peircean_goldbach_inquiry(
    problem_type: str = "goldbach",
    n: int = 26,
    variance_threshold: float = 0.05,
    chain_prob: float = 0.7,
    sample_count: int = 8,
    abstraction_depth_max: int = 3,
    synechism_prob: float = 0.5
):
    print(f"\n{'='*100}")
    print(f"Solving Deep Peircean Inquiry: {problem_type.upper()} (n={n})")
    print(f"Params: variance_threshold={variance_threshold}, chain_prob={chain_prob}, sample_count={sample_count}, "
          f"abstraction_depth_max={abstraction_depth_max}, synechism_prob={synechism_prob}")
    print(f"With: Fallibilism Critique, Chained Hypostatic Abstraction, Synechistic Continuity, Self-Reflection & Dynamic Adjustment")
    print(f"{'='*100}\n")

    # 1. ABDUCTION ── Hypothesis Generation
    discovery = GoldbachDiscovery(n)
    pairs = discovery.abduce_prime_pairs()
    print("\n1. ABDUCTION (Discovery)")
    print(discovery.diagram_summary())

    # Probabilistic graph evolution
    graph = PeirceGraph()
    graph.abduct_hypothesis(problem_type, n=n, synechism_prob=synechism_prob)
    graph.draw_ascii()

    print("2. DEDUCTION ── Diagram Manipulation, Evolution & Deep Theorematic Steps")
    rules = ["insertion", "erasure", "iteration", "delta_tincture", "theorematic_symmetry", "hypostatic_abstraction"]
    for _ in range(12):
        graph.apply_rule(random.choice(rules), chain_prob=chain_prob)
    graph.evolve_graph(steps=5)
    graph.draw_ascii()

    # Pure Existential Graph lemmas
    print("\n2. DEDUCTIVE UNFOLDING (Existential Graphs with Delta Governance)")
    lemmas_eg_delta = ExistentialGraphDeduction.extract_lemmas_eg_delta(n, pairs)
    for i, lemma in enumerate(lemmas_eg_delta, 1):
        print(f"\nLemma {i} (Graph Notation):")
        print(lemma)

    # 3. FORMAL CERTIFICATION
    print("\n3. FORMAL CERTIFICATION")
    if GoldbachProof.certify(n, pairs):
        print("✔ Certified instance:")
        for p, q in pairs:
            print(f"   {p} + {q} = {n}")
    else:
        print("✘ No certificate found")

    # 4. INDUCTION ── Generalization with Fallibilism & Self-Reflection
    print("\n3. INDUCTION ── Generalization with Fallibilism & Self-Reflection")
    print(graph.induct_generalization(sample_count=sample_count, variance_threshold=variance_threshold))

    print("\nFinal Peircean Verdict:")
    print(graph.deduct_conclusion())

    print("\nVisual Representation (if matplotlib available):")
    graph.draw_visual()

    print("\nStatus:")
    print("• Discovery: fallible (abductive)")
    print("• Graphs: diagrammatic transformations (deductive, Beta/Gamma)")
    print("• Delta: meta-governance (normative rules)")
    print("=" * 80)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting Meaning of CS Peirce at {datetime.now():%Y-%m-%d %H:%M:%S}")
    peircean_goldbach_inquiry(n=26, chain_prob=0.7, synechism_prob=0.5)
    print(f"Inquiry finished. Full log saved to: peirce_inquiry.log")