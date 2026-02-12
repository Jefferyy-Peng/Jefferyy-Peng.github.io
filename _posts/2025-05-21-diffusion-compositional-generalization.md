---
layout: post
title: Compositional Generalization in Diffusion Models
date: 2026-01-12 16:40:16
description: Papers for "Compositional Generalization in Diffusion Models"
tags: Diffusion Compositional-Generalization
categories: Paper-Reading
---

## TLDR
- **Paper 1 (SIM / Swing-by Dynamics)** is a highly abstract wrapper of compositional generalization as **learning an identity mapping** on a structured Gaussian mixture. It ignores system internals (e.g., diffusion sampling, U-Nets) and develops theory for the learning dynamics of resulting simplified problem.
- **Paper 2 (CPC/LCS in Conditional Diffusion)** instead goes into **diffusion models directly**, showing that **local inductive bias** (sparse conditional dependencies in the score) is the mechanism that enables compositional / length generalization.

---

# Paper 1 — Swing-by Dynamics in Concept Learning and Compositional Generalization (SIM) (Yang et al., ICLR 2025)
*PDF: [2410.08309v3](https://arxiv.org/pdf/2410.08309v3.pdf)*  

## 1. Motivation: a theoretical wrapper for “concept space” diffusion results
Prior work evaluates a diffusion model by:
- mapping conditioning concepts to a vector,
- using a classifier to map generated images back to concept accuracy vectors,
- so a “good generator + classifier system” behaves like an **identity mapping** in concept space.

This paper argues:
> The salient part is the *structured organization* of concept space, not the diffusion internals.

So they introduce a simplified task: **Structured Identity Mapping (SIM)**.

---

## 2. SIM dataset: Gaussian mixture with structured centroids
Let $$d$$ be dimension and $$s\le d$$ be number of concept classes.

Training data consists of $$s$$ Gaussian clusters aligned with coordinate axes:
$$
x_k^{(p)} \sim \mathcal{N}\!\left(\mu_p \mathbf{1}_p,\; \mathrm{diag}(\sigma)^2\right),
\quad p\in [s],\; k\in[n].
$$

Interpretation:
- $$\mu_p$$ = concept signal strength (cluster mean distance from origin),
- $$\sigma_p$$ = concept diversity (variance along that axis).

The learning problem is identity mapping with MSE:
$$
L(\theta)
=
\frac{1}{2sn}\sum_{p=1}^s\sum_{k=1}^n
\|f(\theta; x_k^{(p)}) - x_k^{(p)}\|^2.
$$

Evaluation uses an OOD “composition” point:
$$
x_b = \sum_{p=1}^s \mu_p \mathbf{1}_p.
$$

---

## 3. Linearization: loss in terms of covariance
Assume the model is linear in input:
$$
f(\theta; x)=W_\theta x.
$$

Then (via trace trick) the loss becomes:
$$
L(\theta)
=
\frac{1}{2}\|(W_\theta - I)A^{1/2}\|_F^2,
$$
where the (population) covariance is diagonal:
$$
A=\mathbb{E}[xx^\top] = \mathrm{diag}(a),
\quad
a_p=
\begin{cases}
\sigma_p^2 + \frac{\mu_p^2}{s}, & p\le s,\\
0, & p>s.
\end{cases}
$$

Key: learning rates along coordinates are controlled by $$a_p$$, hence by $$\mu_p$$ and $$\sigma_p$$.

---

## 4. One-layer linear model: closed-form dynamics
For $$f(W;x)=Wx$$ under gradient flow, they derive:
$$
f(W(t),z)_k
=
\mathbf{1}_{k\le s}\bigl(1-e^{-a_k t}\bigr)z_k
+
\sum_{i=1}^s e^{-a_i t} w_{k,i}(0) z_i.
$$

Interpretation:
- A “growth” term drives the correct identity mapping,
- A “noise” term decays with time and initialization scale.

Consequences:
- Concepts with larger $$a_k$$ converge faster.
- Since $$a_k$$ increases with $$\mu_k$$ and $$\sigma_k$$, generalization order is governed jointly by **signal strength** and **diversity**.

This reproduces empirical “concept order” phenomena.

Limitation:
- coordinates evolve independently → no non-monotonic OOD behavior.

---

## 5. Deep linear model: symmetric 2-layer network and Swing-by dynamics
They analyze:
$$
f(U;x)=UU^\top x,
\quad W(t)=UU^\top.
$$

They obtain an evolution equation for Jacobian entries $$w_{i,j}$$ decomposed into:
- growth term,
- suppression term,
- noise term,

which yields **multi-stage dynamics**:
1. initial growth of many Jacobian entries,
2. one major diagonal entry grows first,
3. it suppresses associated off-diagonal “minor” entries,
4. next major entry grows, and so on.

This staged Jacobian evolution produces an OOD trajectory that:
- initially moves toward the OOD composition point,
- then detours back toward training cluster(s),
- then later returns to OOD performance.

They call this **Swing-by dynamics** and connect it to a double-descent-like test loss curve (for OOD).

---

## 6. Empirical bridge back to diffusion
Even though SIM is abstract, they verify in text-conditioned diffusion models that:
- OOD concept accuracy can be non-monotonic during training,
- matching the “Swing-by” mechanism predicted by theory.

---

## 7. Takeaway from Paper 2
This paper treats compositional generalization as a **wrapper identity mapping problem**:
- it ignores internal generative machinery,
- and explains sequential concept learning + non-monotonic OOD curves as consequences of optimization dynamics on structured data.

---

# Paper 2 — Local Mechanisms of Compositional Generalization in Conditional Diffusion (Bradley et al., 2025)
*PDF: [2509.16447v2](https://arxiv.org/pdf/2509.16447v2.pdf)*  

## 1. Motivation: why length generalization is hard
The paper studies **length generalization** in conditional diffusion: train on scenes with a small number of objects, then test with **more** conditions (e.g., more specified locations) than seen during training.

Key observation:
- Whether length generalization succeeds depends on whether the model learns a **compositional mechanism** (one object per condition) or a **shortcut mechanism** (condition triggers a typical scene, not additive per-condition behavior).

## 2. Setup: location-conditioned CLEVR experiments
They use CLEVR with location conditioning:
- **Experiment 1 (success):** conditioner labels **all** object locations.
- **Experiment 2 (failure):** conditioner labels **only one** randomly chosen object location (even if the image has 2–3 objects).
- **Experiment 3 (fix):** enforce an architecture that induces **local conditional score structure**, restoring length generalization.

Crucial point:
Even when training does not include multi-location conditioning, length generalization can still happen if the **right inductive bias** forces the model to represent the conditional distribution compositionally.

---

## 3. Definitions: Score functions and locality
A diffusion model learns the **conditional score**:
$$
s_t(x \mid c) \;=\; \nabla_x \log p_t(x \mid c).
$$

### 3.1 Local Conditional Scores (LCS)
They define **Local Conditional Scores (LCS)** as a sparsity condition on dependencies:

At each pixel $$i$$, the score depends only on:
- a subset of pixels $$N_i$$ (often a local neighborhood), and
- a subset of conditions $$L_i \subseteq J$$ (often nearby conditions for location-conditioning).

Informally:
> The score at pixel $$i$$ does not need the entire image nor all conditioners—only a sparse subset.

This generalizes “local scores” to conditional settings.

---

## 4. Conditional Projective Composition (CPC)
They define a strong form of compositionality of the **conditional distribution**.

Let $$J$$ index the set of conditions $$\{c_j\}_{j \in J}$$.
Let $$\{M_j\}_{j\in J}$$ be **disjoint** pixel subsets, and let $$M_J^c$$ denote the remaining pixels.

A **CPC** distribution factorizes as:
$$
p(x \mid c_J)
\;=\;
p(x_{M_J^c} \mid \varnothing)
\prod_{j \in J} p(x_{M_j} \mid c_j).
$$

Meaning:
- Each condition $$c_j$$ affects only its own region $$M_j$$.
- Different regions are conditionally independent.
- A background region $$M_J^c$$ may be unconditional.

This structure implies length generalization naturally: adding a new condition adds a new independent factor.

---

## 5. Lemma 1: CPC ⇒ LCS (exact)
If $$p(x \mid c_J)$$ satisfies CPC, then its score is LCS.

Sketch:
Take logs:
$$
\log p(x \mid c_J)
=
\log p(x_{M_J^c}\mid \varnothing)
+
\sum_{j\in J}\log p(x_{M_j}\mid c_j).
$$

Differentiate w.r.t. pixel $$i$$:
- If $$i \in M_j$$, then:
$$
\nabla_{x_i}\log p(x\mid c_J)
=
\nabla_{x_i}\log p(x_{M_j}\mid c_j),
$$
so dependence is only on $$x_{M_j}$$ and condition $$c_j$$.
- If $$i \in M_J^c$$, then the score depends only on the unconditional background term.

Thus compositional distributions have **local conditional** score structure.

---

## 6. Relaxation: approximate CPC ⇒ approximate LCS, and “more compositional at high noise”
Real models are not perfectly CPC. The paper relaxes the lemma:

- If $$p(x\mid c)$$ is **approximately** CPC, then the score is **approximately** LCS.
- The approximation becomes better at **higher noise** $$t$$ (intuitively, noise washes out detailed interactions, leaving large-scale compositional structure).

This supports a diffusion-time decomposition:
- **High noise:** conditional dependencies dominate; global structure (object count/layout) is established.
- **Low noise:** pixel dependencies dominate; local unconditional denoising fills in details.

This explains why local conditional mechanisms can “set” the compositional structure early.

---

## 7. Feature-space extension: CPC/LCS after an invertible transform
Pixel-space compositionality often fails for prompts like “watercolor cat sushi” (style and content interact everywhere).

They propose:
Let $$z = A(x)$$ be an invertible feature transform. If the pushforward distribution $$A_\#p(z\mid c)$$ is CPC, then the **feature-space score** is LCS.

This motivates “disentanglement” as CPC/LCS structure in a learned feature space.

### 7.1 Orthogonality heuristic for disentanglement
Define:
$$
\mu_i := \mathbb{E}_{z\sim A_\#p(\cdot\mid c_i)}[z],
\quad
\mu_b := \mathbb{E}_{z\sim A_\#p(\cdot\mid \varnothing)}[z],
\quad
d_i := \mu_i - \mu_b.
$$

A necessary (not sufficient) condition for CPC is pairwise orthogonality:
$$
d_i^\top d_j = 0 \quad \forall i\neq j.
$$

Practically they compute cosine similarity:
$$
\frac{d_i^\top d_j}{\|d_i\|\|d_j\|},
$$
where low off-diagonal similarity suggests feature-space disentanglement.

---

## 8. Causal evidence: enforcing LCS restores generalization
Experiment 3 performs a direct intervention:
- keep training distribution like Experiment 2,
- enforce architectural locality producing LCS-like score dependencies,
- observe restored length generalization.

Conclusion:
> The local conditional score structure is not merely correlated with generalization; it is a **causal mechanism**.

---

## 9. Takeaway from Paper 1
Compositional generalization in conditional diffusion hinges on an **inductive bias**:
- representing conditional effects in a sparse / local way (LCS),
- which corresponds to a compositional factorization of the conditional distribution (CPC).

---


# Synthesis: How the two papers complement each other
- **Paper 1**: compositionality depends on *mechanistic inductive bias in diffusion*: local conditional score structure.
- **Paper 2**: compositional phenomena can arise even in a stripped-down identity task: optimization + data geometry create staged learning and Swing-by.

Together:
- Paper 2 explains *when* and *in what order* concept directions emerge under training dynamics.
- Paper 1 explains *whether* a diffusion system will actually realize an additive compositional mechanism, via locality/sparsity constraints in the conditional score.