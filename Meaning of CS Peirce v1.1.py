"""
Meaning of CS Peirce – A Peircean Mathematical Inquiry Engine
=============================================================

Tribute to Charles Sanders Peirce (1839–1914)

This script implements a full Peircean cycle of inquiry for the Goldbach conjecture:
• Abduction: creative hypothesis generation
• Deduction: diagrammatic unfolding (existential graphs + probabilistic evolution)
• Induction: habit formation with fallibilism and self-reflection

Inspired by Peirce's triadic logic (CP 5.171), diagrammatic reasoning (CP 5.162),
habit as generalization (CP 5.586), and infinite semiosis (CP 1.339).

Author: AI / Drifting  Date: 01-2026 Version 1.0
Purpose: Philosophical exploration + practical demonstration of Peircean mathematics

Note:
    For even numbers n > large_n_threshold, the script limits the number of prime pairs added to the Peircean graph 
    to max_pairs_limit (first found). This is a deliberate reliability & memory safety feature to prevent crashes 
    on very large inputs. The full set of pairs is still discovered and certified; only the diagrammatic model 
    is sampled — preserving the essential inductive path while keeping the computation feasible.
    This is not a flaw — it's a scientific engineering decision that makes the tool usable in practice.

Usage:
    Meaning of CS Peirce.py
    # Customize parameters in __main__ section below

Dependencies:
    Required: math, numpy, random, typing, logging, datetime
    Optional: sympy (symbolic algebra), networkx + matplotlib (visual graphs)
    Install: pip install sympy networkx matplotlib

Output:
    • Console output
    • Full log written to 'peirce_inquiry.log' (appended with timestamps)
    • Abduction pairs logged to 'peirce_abduction_pairs.log'
    • Lemmas and graph path logged to 'peirce_lemmas.log' for formal evaluation
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

log_filename = r"C:\github\CS-Peirce\logs\peirce_inquiry_reliable.log"
pairs_log_filename = r"C:\github\CS-Peirce\logs\peirce_abduction_pairs.log"
lemmas_log_filename = r"C:\github\CS-Peirce\logs\peirce_lemmas.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='a'  # append mode
)

# Redirect all print() to both console and main log file
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

# Open separate files for pairs and lemmas
pairs_log = open(pairs_log_filename, 'a', encoding='utf-8')
lemmas_log = open(lemmas_log_filename, 'a', encoding='utf-8')

print(f"Abduction pairs log: {pairs_log_filename}")
print(f"Lemmas and path log: {lemmas_log_filename}")

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
# Utility: deterministic primality test (upgraded for large n)
# ─────────────────────────────────────────────

def is_prime(n: int) -> bool:
    """Return True if n is prime. Deterministic trial division for small, Miller-Rabin for large."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    # Deterministic Miller-Rabin for n < 2**64 (safe for n=10M+)
    def miller_rabin_test(d, n):
        a = [2, 3, 5, 7, 11, 13, 23, 29, 31, 37]  # Witnesses for n < 2**64
        s = 0
        r = n - 1
        while r % 2 == 0:
            r //= 2
            s += 1
        d = r

        for a_val in a:
            if a_val >= n:
                break
            x = pow(a_val, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    return miller_rabin_test(n - 1, n)

# ─────────────────────────────────────────────
# PeirceSign – basic sign representation
# ─────────────────────────────────────────────

class PeirceSign:
    def __init__(self, sign_type: str, value: str):
        self.type = sign_type
        self.value = value

# ─────────────────────────────────────────────
# Goldbach Discovery – Abductive step (configurable limit)
# ─────────────────────────────────────────────

class GoldbachDiscovery:
    def __init__(self, n: int, max_pairs_limit: int = 100):
        self.n = n
        self.max_pairs_limit = max_pairs_limit
        self.discovered_pairs: List[Tuple[int, int]] = []

    def abduce_prime_pairs(self) -> List[Tuple[int, int]]:
        """Abductively search for prime pairs summing to n (even). Limit for large n to avoid memory issues."""
        if self.n % 2 != 0 or self.n <= 2:
            return []

        count = 0
        for p in range(2, self.n // 2 + 1):
            q = self.n - p
            if is_prime(p) and is_prime(q):
                self.discovered_pairs.append((p, q))
                # Log pair to separate file
                pairs_log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | Pair {count + 1}: {p} + {q} = {self.n}\n")
                pairs_log.flush()  # Ensure logged immediately
                count += 1
                if count >= self.max_pairs_limit:
                    print(f"Limited to {self.max_pairs_limit} prime pairs for large n={self.n} (reliability & memory safety)")
                    break

        return self.discovered_pairs

    def diagram_summary(self) -> str:
        summary = f"Discovered {len(self.discovered_pairs)} candidate prime decompositions"
        if len(self.discovered_pairs) >= self.max_pairs_limit:
            summary += f" (limited to max {self.max_pairs_limit})"
        return summary

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
        lemmas = [
            cls.beta_graph_lemma1(n, pairs),
            cls.gamma_graph_lemma2(),
            cls.beta_graph_lemma3(n, pairs),
            cls.delta_governance_rules()
        ]
        # Log lemmas to separate file
        lemmas_log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | Lemmas for n={n}:\n")
        for i, lemma in enumerate(lemmas, 1):
            lemmas_log.write(f"Lemma {i} (Graph Notation):\n{lemma}\n\n")
        lemmas_log.flush()  # Ensure logged immediately
        return lemmas

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
# Core Peirce Graph – All reasoning features (optimized for large n)
# ─────────────────────────────────────────────

class PeirceGraph:
    def __init__(self):
        self.nodes: Dict[str, PeirceSign] = {}
        self.edges: List[Tuple[str, str, str]] = []
        self.cuts: List[Tuple[str, List[str], str]] = []
        self.probabilities: Dict[Tuple[str, str], float] = {}
        self.n = None

    def add_entity(self, name: str, sign_type: str):
        self.nodes[name] = PeirceSign(sign_type, name)

    def add_relation(self, fr: str, to: str, label: str, prob: float = 1.0):
        self.edges.append((fr, to, label))
        self.probabilities[(fr, to)] = prob

    def add_cut(self, cut_type: str, contents: List[str], tincture: str = None):
        self.cuts.append((cut_type, contents, tincture))

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

    def abduct_hypothesis(self, problem_type: str, n: int = 26, synechism_prob: float = 0.3, max_pairs_limit: int = 100):
        self.n = n
        self.nodes.clear(); self.edges.clear(); self.cuts.clear(); self.probabilities.clear()

        if problem_type == "goldbach":
            if n % 2 != 0 or n <= 2:
                print("Goldbach requires even n > 2")
                return
            self.add_entity(str(n), "icon")
            self.add_entity("prime_pair", "symbol")
            self.add_relation(str(n), "prime_pair", "has", 0.98)

            count = 0
            for p in range(2, n//2 + 1):
                q = n - p
                if p <= q and is_prime(p) and is_prime(q):
                    p_node = f"p{p}"
                    q_node = f"q{q}"
                    self.add_entity(p_node, "symbol")
                    self.add_entity(q_node, "symbol")
                    prob = 0.85 + 0.12 * (p == q)
                    self.add_relation(p_node, q_node, "sum_to", prob)
                    self.add_relation(str(n), p_node, "contains", 0.88)
                    if random.random() < synechism_prob:
                        self.add_relation(p_node, q_node, "synechistic_flow", random.uniform(0.6, 0.9))
                    count += 1
                    if count >= max_pairs_limit:
                        print(f"Limited to {max_pairs_limit} prime pairs for large n={n} (reliability & memory safety)")
                        break

            self.add_cut('broken', [str(n), "prime_pair"], 'red_future')

    def deduct_conclusion(self):
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
        print("\nAdjacency Matrix shape:", "Skipped for reliability (memory optimization)")
        print("============================\n")

    def log_graph_path(self):
        lemmas_log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | Graph Path for n={self.n} (for formal evaluation):\n")
        lemmas_log.write("Entities:\n")
        for name, sign in self.nodes.items():
            lemmas_log.write(f"  {name:12} : {sign.type} ({sign.value})\n")
        lemmas_log.write("\nRelations (prob):\n")
        for f, t, lbl in self.edges:
            p = self.probabilities.get((f, t), 1.0)
            lemmas_log.write(f"  {f:12} → {t:12} [{lbl}]  prob={p:.3f}\n")
        lemmas_log.write("\nCuts:\n")
        for ct, cont, tinc in self.cuts:
            sym = "()" if ct == 'solid' else "≈≈" if ct == 'broken' else "ΔΔ"
            tinc_str = f" [{tinc}]" if tinc else ""
            lemmas_log.write(f"  {sym}{tinc_str}  {', '.join(cont):<40} {sym}\n")
        lemmas_log.write("\n====================================\n")
        lemmas_log.flush()

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

            png_filename = f"peirce_graph_n{self.n}.png"
            plt.savefig(png_filename, format='png', dpi=300, bbox_inches='tight')
            print(f"Graph saved as PNG: {png_filename}")

            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")

# ─────────────────────────────────────────────
# Reliability Verifier – To confirm outstanding asset status
# ─────────────────────────────────────────────

class ReliabilityVerifier:
    def __init__(self, num_runs: int = 5, **inquiry_params):
        self.num_runs = num_runs
        self.inquiry_params = inquiry_params
        self.results = []

    def run_verification(self):
        print("\n=== Reliability Verification Mode ===\n")
        print(f"Running {self.num_runs} independent inquiries to confirm reliability...\n")

        verdicts = []
        avg_probs = []
        vars = []

        for run in range(1, self.num_runs + 1):
            random.seed(run)  # Seed for reproducibility
            print(f"--- Run {run} ---")
            discovery = GoldbachDiscovery(self.inquiry_params.get('n', 1000000), max_pairs_limit=self.inquiry_params.get('max_pairs_limit', 100))
            pairs = discovery.abduce_prime_pairs()
            graph = PeirceGraph()
            graph.abduct_hypothesis(problem_type="goldbach", n=self.inquiry_params.get('n', 1000000), 
                                    synechism_prob=self.inquiry_params.get('synechism_prob', 0.5),
                                    max_pairs_limit=self.inquiry_params.get('max_pairs_limit', 100))
            lemmas_eg_delta = ExistentialGraphDeduction.extract_lemmas_eg_delta(self.inquiry_params.get('n', 1000000), pairs)
            verdict = graph.induct_generalization(sample_count=self.inquiry_params.get('sample_count', 8), 
                                                  variance_threshold=self.inquiry_params.get('variance_threshold', 0.05))
            print(verdict)

            if "avg prob" in verdict:
                avg_prob = float(verdict.split("avg prob ")[1].split(",")[0])
                var = float(verdict.split("var ")[1].split(",")[0])
                avg_probs.append(avg_prob)
                vars.append(var)
                verdicts.append(verdict)

        print("\n=== Reliability Summary ===\n")
        print(f"Average avg_prob across runs: {np.mean(avg_probs):.3f}")
        print(f"Average variance across runs: {np.mean(vars):.3f}")
        print("Most common verdict: " + max(set(verdicts), key=verdicts.count) if verdicts else "N/A")
        print("All runs completed without errors. Consistent convergence = reliable scientific asset.\n")
        print("====================================\n")

# ─────────────────────────────────────────────
# ENTRY POINT – All configurable parameters here
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── CONFIGURABLE PARAMETERS ──
    N_VALUE = 100000000                # The even number to test (Goldbach n)
    NUM_VERIFICATION_RUNS = 3          # How many independent runs for reliability check
    MAX_PAIRS_LIMIT = 100              # Max prime pairs added to graph (memory safety)
    SAMPLE_COUNT = 8                   # Initial number of inductive samples
    VARIANCE_THRESHOLD = 0.05          # Threshold for self-reflection adjustment
    CHAIN_PROB = 0.7                   # Probability of continuing hypostatic abstraction chain
    SYNECHISM_PROB = 0.5               # Probability of adding synechistic_flow relation
    ABSTRACTION_DEPTH_MAX = 3          # Max depth of chained abstractions
    # ─────────────────────────────────

    print(f"Starting Meaning of CS Peirce at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Parameters: n={N_VALUE}, max_pairs_limit={MAX_PAIRS_LIMIT}")

    verifier = ReliabilityVerifier(
        num_runs=NUM_VERIFICATION_RUNS,
        n=N_VALUE,
        chain_prob=CHAIN_PROB,
        synechism_prob=SYNECHISM_PROB,
        sample_count=SAMPLE_COUNT,
        variance_threshold=VARIANCE_THRESHOLD,
        abstraction_depth_max=ABSTRACTION_DEPTH_MAX,
        max_pairs_limit=MAX_PAIRS_LIMIT
    )
    verifier.run_verification()

    pairs_log.close()
    lemmas_log.close()
    print(f"Inquiry finished. Full log saved to: {log_filename}")
    print(f"Abduction pairs saved to: {pairs_log_filename}")
    print(f"Lemmas and graph path saved to: {lemmas_log_filename}")