# Meaning of C.S. Peirce

A living philosophical and mathematical tribute to **Charles Sanders Peirce** (1839–1914) — Mathematician, Logician, founder of modern semiotics, and the architect of the logic of inquiry.

This project implements a **genuine Peircean mathematical inquiry engine** focused on the Goldbach conjecture (easily extensible to other problems).

## This is a Prototype 

**The AI had built in subtle prior knowledge about the result (Hardy formula), which pull it towards the result, which is why the script seems to work.
What I have learned: Never blindly trust the results of AI or science!**

Shouldn't there be a goal to compare it to, so that the methode knows what it is looking for?

**You have touched on the most important point of conflict in artificial intelligence: how do you search for something when you don't know what it looks like?
In Peirce's philosophy, the goal is not a ‘formula’. The goal is the elimination of surprises. Peirce argued that the search only begins when we experience a ‘conflict’ between our expectations and reality.
In order for the script to reinvent the Hardy-Littlewood formula without prior knowledge, its ‘goal’ should be to find a state without surprises (zero residual error).**

**Why "Blank Sheet" is better than "Pre-Knowledge"**

If you give a script pre-knowledge, it is a Prisoner of the Formula. 
If the formula has a tiny mistake, the script can never fix it.

If the script's only goal is Eliminating Surprise:
- It is Fallible: It can make mistakes.
- It is Self-Correcting: It uses the mistake as the "clash" to find a better truth.
- It is Infinite: It will keep refining the formula (adding p=5,7,11...) until the surprise is so small it becomes "background noise."


**[The Meaning of This Script →](https://github.com/gitfrid/CS-Peirce/wiki)**  
(Explore other wiki pages for usage, technical notes, and deeper reflections on Peirce in computation.)

It faithfully follows Peirce's complete triadic cycle:
- **Abduction** — creative, fallible hypothesis generation
- **Deduction** — diagrammatic unfolding via existential graphs (Beta/Gamma/Delta) and probabilistic habit evolution
- **Induction** — stabilization of mathematical habits with self-reflection, variance analysis, and explicit fallibilism
- **Symbolic translation** — recognition and expression of the resulting habit in the language of analytic number theory

Inspired by Peirce's core ideas:
- Diagrammatic reasoning as the heart of necessary inference (CP 5.162)
- Truth as the limit of endless inquiry (CP 5.565)
- Synechism (continuity), tychism (chance), and habit-formation (CP 6.169, 5.586)

## Features

- Probabilistic graph evolution with chained hypostatic abstractions, synechistic flows, theorematic symmetry, and modal cuts
- Pure Existential Graphs (Beta/Gamma/Delta) rendered as detailed text/ascii-art lemmas
- Self-reflective induction — automatically adjusts sample count when variance is high
- Symbolic transcription of the recognized Hardy–Littlewood asymptotic (via SymPy) with RH error term + Deuring–Heilbronn alternative
- Full logging of abduction pairs, lemmas, and final graph path
- Visual graph plot + automatic PNG save for small n (≤ 10,000)

> **Prototype Python code**: [Meaning of CS Peirce.py](https://github.com/gitfrid/CS-Peirce/blob/main/Meaning%20of%20CS%20Peirce.py)  
> **Example log (n=26)**: [peirce_inquiry n26.log](https://raw.githubusercontent.com/gitfrid/CS-Peirce/refs/heads/main/logs/peirce_inquiry%20n26.log)  
> **Example log (n=1M)**: [peirce_inquiry 1M.log](https://raw.githubusercontent.com/gitfrid/CS-Peirce/refs/heads/main/logs/peirce_inquiry%201M.log)

## On Validity, Proof, and Induction

This project does **not** claim to prove the Goldbach conjecture.  
Instead, it faithfully models Charles S. Peirce’s account of how mathematical knowledge actually grows: through abductive discovery, diagrammatic deduction, and inductive stabilization of habits.

For each tested instance, deterministic verification is performed.  
For large n (e.g. n = 1,000,000), multiple inductive samples show near-zero variance — a maximally stable (but still fallible) habit that "Goldbach holds with very high confidence" within the framework.

In Peircean terms, this is **not** final truth, but the **limit of current inquiry** — explanatory, convergent, and open to future surprise.

## The Symbolic Translation Step

After the full inquiry cycle, the script **recognizes** that its stabilized habit aligns with the famous **Hardy–Littlewood asymptotic formula** (1923) for the number of representations G(n):

$$
G(n) \sim 2\, C_2 \prod_{p>2} \frac{p-2}{p-1} \cdot \frac{n}{(\ln n)^2} \quad \text{(with RH-dependent error term)}
$$

- The **singular product** \(\prod_{p>2} \frac{p-2}{p-1}\) and the **twin prime constant** \(C_2 \approx 0.6601618158\) are the classical form used by number theorists.
- The script does **not** derive this expression from first principles (that would require recreating 20th-century analytic number theory).  
- Instead, it **independently converges** to the same **conviction** through its own diagrammatic/probabilistic path, then **translates** the resulting habit into the standard symbolic language mathematicians publish.

Alternative consideration (Deuring–Heilbronn phenomenon):  
> If the Riemann Hypothesis is false, a Siegel zero would create even stronger repulsion among primes, further ensuring G(n) > 0.

This step bridges the **Peircean world of diagrams and habits** to the **conventional notation of analytic number theory** — not as a proof, but as an explanatory and convergent endpoint.

## Customize parameters in the __main__ block

```python
N_VALUE = 1000000                  # Even number to test (Goldbach n)
NUM_VERIFICATION_RUNS = 3          # Number of independent runs for reliability
MAX_PAIRS_LIMIT = 100              # Max prime pairs added to graph (memory safety)
SAMPLE_COUNT = 8                   # Initial inductive samples
VARIANCE_THRESHOLD = 0.05          # Fallibilism threshold for adjustment
CHAIN_PROB = 0.7                   # Depth of hypostatic abstraction chaining
SYNECHISM_PROB = 0.5               # Continuity/probabilistic flow
ABSTRACTION_DEPTH_MAX = 3          # Max chained abstractions
