# Meaning of C.S. Peirce

A living philosophical and mathematical tribute to **Charles Sanders Peirce** (1839–1914) — Mathematician, Logician, the founder of modern semiotics, and the logic of inquiry.

This project implements a **complete Peircean mathematical inquiry engine** focused on the Goldbach conjecture (easily extensible to other problems).

It faithfully follows Peirce's triadic cycle of reasoning:
- **Abduction** — creative, fallible hypothesis generation
- **Deduction** — diagrammatic unfolding through existential graphs (Beta/Gamma/Delta) and probabilistic evolution
- **Induction** — formation of stable habits with self-reflection, variance analysis, and fallibilism
- **Symbolic translation** — final algebraic expression for the mathematical community (Hardy–Littlewood bound + Deuring–Heilbronn alternative)

Inspired by Peirce's core ideas:
- Diagrammatic reasoning as the essence of necessary inference (CP 5.162)
- Truth as the limit of endless inquiry (CP 5.565)
- Synechism (continuity), tychism (chance), and habit-formation (CP 6.169, 5.586)

## Features

- Probabilistic graph evolution with chained hypostatic abstractions, synechistic flows, theorematic symmetry, and modal cuts
- Pure Existential Graphs (Beta/Gamma/Delta) as detailed text/ascii-art lemmas
- Self-reflective induction — automatically adjusts sample count on high variance
- Symbolic Hardy–Littlewood lower bound (via SymPy) with RH error term + Deuring–Heilbronn alternative
- Full logging of all output (including lemmas) to `peirce_inquiry.log`
- Beautiful visual graph plot + automatic PNG save for small n (≤ 10,000)

> **Python code**: [AG) bias necessity falsification rmst.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AG%29%20Diagnostics%20bias%20necessity%20falsification%20rmst.py)

## Dependencies

pip install numpy sympy networkx matplotlib

## Customize parameters in the __main__ block:python

    - n=26,                    # small for visualization
    - chain_prob=0.7,          # depth of hypostatic abstraction chaining
    - synechism_prob=0.5,      # continuity / probabilistic flow between structures
    - sample_count=8,          # number of inductive probes
    - variance_threshold=0.05  # fallibilism threshold for induction


Small n (e.g. 26): generates interactive plot + saves PNG
Large n (e.g. 1,000,000): skips plot (too large), shows rich ASCII diagram + algebraic bound
All output logged to peirce_inquiry.log



## Example Results (n = 26)

### 1. Probabilistic Graph Visualization
For small n like 26, the script generates an interactive plot and saves it as `peirce_graph_n26.png`.

![Peircean Goldbach Diagram for n=26](https://github.com/gitfrid/CS-Peirce/blob/main/CS%20Peirce%20Graphs%20n16.png)

**What the graph shows**:
- Central green node: **26** (the icon/problem)
- Blue nodes: individual primes (p3, q23, p7, q19, p13, q13)
- Orange nodes: modal hypotheses (RH_noise, no_pair)
- Edges labeled with Peircean relations: `sum_to`, `contains`, `synechistic_flow`, `reifies`, `higher_reifies`, `has`, etc.
- Abstraction chains (abs_sum_to → abs_abs_sum_to) show hypostatic abstraction
- Cuts (broken/solid lines) represent negation, possibility, governance

This is the **evolved habit graph** — the living result of Peircean deduction + induction.

### 2. Existential Graph Lemmas (Beta/Gamma/Delta)
These are shown as detailed text/ascii-art in console and log file — pure diagrammatic reasoning:

- **Lemma 1** — Beta relational existence of two primes summing to n
- **Lemma 2** — Gamma broken-cut possibility (p determines q)
- **Lemma 3** — Beta symmetry collapse (when p=q)
- **Lemma 4** — Delta tinctured governance (rules for elevating to universal habit)

Full ascii-art lemmas are preserved in `peirce_inquiry.log` — ready to copy for papers or books.

### 3. Algebraic Translation (for formal mathematicians)
Always generated — the Hardy–Littlewood lower bound for the number of representations G(n):

$$
\frac{2 C n \prod_{p=3}^{\infty} \frac{p - 2}{p - 1}}{\log^2 n} - \text{Error(RH)}
$$

**Alternative path** (Deuring–Heilbronn phenomenon):
> If the Riemann Hypothesis is false, a Siegel zero creates even stronger repulsion among primes, ensuring G(n) > 0.

This symbolic expression is the bridge from Peircean diagrammatic habits to modern analytic number theory — ready for papers, proof assistants, or referee reports.
