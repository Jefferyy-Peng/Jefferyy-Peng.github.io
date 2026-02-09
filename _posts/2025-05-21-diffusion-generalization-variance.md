---
layout: post
title: Generalization Through Variance in Diffusion Models
date: 2025-05-21 16:40:16
description: Paper summary for "Generalization through Variance in Diffusion Models"
tags: Learning
categories: sample-posts
---


"""
# Generalization Through Variance in Diffusion Models — Logical Summary

## 1. Core Question of the Paper

The paper asks a precise and nonstandard question:

> **Why do diffusion models trained with denoising score matching (DSM) generate samples that go *beyond* memorizing the training data, even when the true score would only reproduce the training examples?**

Equivalently:
- Why does training with the *proxy score* $$\tilde s(x,t;x_0)$$ lead to **generalization**, whereas the *true score* $$s(x,t)$$ would lead to exact memorization?

The paper’s answer is:
> **Generalization arises from variance in the learned score, not from bias.**

---

## 2. Forward Diffusion and Scores (Setup)

### Forward process
Data points $$x_0 \sim p_{\text{data}}$$ are corrupted by a linear SDE:
$$
\dot x_t = -\beta_t x_t + G_t \eta_t,
$$
leading to Gaussian conditionals:
$$
p(x \mid x_0, t) = \mathcal N(x; \alpha_t x_0, S_t).
$$

### Scores
- **True score**:
$$
s(x,t) := \nabla_x \log p(x \mid t).
$$

- **Proxy score** (DSM target):
$$
\tilde s(x,t;x_0) := \nabla_x \log p(x \mid x_0,t) = S_t^{-1}(\alpha_t x_0 - x).
$$

Key identity:
$$
\mathbb E_{x_0 \mid x,t}[\tilde s(x,t;x_0)] = s(x,t).
$$

So the proxy score is **unbiased**, but **noisy**.

---

## 3. DSM Objectives and the Key Observation

Two losses:
- Ideal (unavailable):
$$
J_0 = \mathbb E\, \| \hat s - s \|^2
$$
- Practical (DSM):
$$
J_1 = \mathbb E\, \| \hat s - \tilde s \|^2
$$

Although $$J_0$$ and $$J_1$$ share the same minimizer in infinite-capacity limits, **models trained with $$J_1$$ generalize better**.

The paper shows this is because the proxy score has **nontrivial covariance**:
$$
C(x,t) := \mathrm{Cov}_{x_0 \mid x,t}[\tilde s(x,t;x_0)] 
= S_t^{-1} + \nabla_x^2 \log p(x \mid t).
$$

This covariance is:
- large at small $$t$$ (since $$S_t \to 0$$),
- large near training points and between them (high curvature regions).

---

## 4. Sampling via PF-ODE and the Central Difficulty

Sampling uses the **probability flow ODE (PF-ODE)**:
$$
\dot x_t = -\beta_t x_t - D_t \hat s_\theta(x_t,t).
$$

For fixed $$\theta$$ and $$x_T$$, this is **deterministic**:
$$
x_0 = F(x_T, \theta).
$$

But:
- $$\hat s_\theta$$ depends on random training samples,
- hence $$F(x_T,\theta)$$ is a **random variable**.

We want the **distribution** of outputs, not their mean:
$$
\bar q(x_0 \mid x_T) = \mathbb E_\theta[\delta(x_0 - F(x_T,\theta))].
$$

This delta form is **not optional** — it is the definition of the distribution induced by a random variable.

---

## 5. Path Integral Reformulation (Key Technical Contribution)

Directly averaging $$F(x_T,\theta)$$ is intractable because:
- PF-ODE has no closed-form solution,
- $$\hat s_\theta$$ enters nonlinearly through the trajectory.

**Key trick**:
Rewrite PF-ODE sampling as a **path integral** enforcing the ODE as a constraint:
$$
q(x_0 \mid x_T; \theta)
=
\int \mathcal D[x_t]\mathcal D[p_t]\,
\exp\left(
\int_T^\epsilon i p_t \cdot 
[\dot x_t + \beta_t x_t + D_t \hat s_\theta(x_t,t)]\,dt
\right).
$$

- The auxiliary field $$p_t$$ enforces the ODE via a delta-functional.
- Crucially, $$\hat s_\theta$$ appears **linearly inside the exponent**.

This transforms the problem into averaging an **exponential of a linear functional of a random function**.

---

## 6. Averaging Over Training Randomness

Because of linearity, the ensemble average becomes a **cumulant expansion**:
$$
[ q(x_0 \mid x_T) ]
=
\int \mathcal D[x_t]\mathcal D[p_t]\,
\exp\left(
M_1 - \tfrac12 M_2 + \cdots
\right).
$$

Where:
- $$M_1$$ depends on the **mean score**
$$
s_{\text{avg}}(x,t) := [\hat s_\theta(x,t)].
$$
- $$M_2$$ depends on the **score covariance**
$$
V(x,t;x',t') := D_t\,\mathrm{Cov}_\theta[\hat s(x,t),\hat s(x',t')]\,D_{t'}.
$$

Assuming higher cumulants are negligible (Gaussian approximation), the averaged dynamics are equivalent to an SDE.

---

## 7. Main Result: Effective SDE for the Typical Learned Model

**Proposition (central result)**:
Sampling from $$[q(x_0 \mid x_T)]$$ is equivalent to integrating:
$$
\dot x_t
=
-\beta_t x_t
-
D_t s_{\text{avg}}(x_t,t)
+
\xi(x_t,t),
$$

where:
$$
\mathbb E[\xi] = 0, 
\qquad
\mathbb E[\xi(t)\xi(t')] = V(x_t,t;x_{t'},t').
$$

Interpretation:
- The **mean dynamics** follow the deterministic PF-ODE.
- **Generalization arises entirely from the noise term**.

If $$V = 0$$ → pure memorization.
If $$V \neq 0$$ → stochastic spreading → generalization.

---

## 8. Concrete Example: Naive Score Estimator

The paper shows this is not abstract theory by constructing a toy algorithm:

At each Euler step:
1. Sample $$x_{0t} \sim p(x_0 \mid x_t,t)$$,
2. Use a noisy score:
$$
\hat s(x_t,t)
=
s(x_t,t)
+
\sqrt{\kappa/\Delta t}
\,[\tilde s(x_t,t;x_{0t}) - s(x_t,t)].
$$

Despite using the proxy score directly (which looks like memorization),
the **resampling-induced variance** yields:
$$
V(x,t;x',t')
=
\kappa D_t C(x,t) D_{t'}\,\delta(t-t').
$$

Thus:
- Noise is strongest where proxy-score covariance $$C(x,t)$$ is large,
- i.e. near boundaries between training examples.

This proves:
> **Variance alone is sufficient to induce generalization.**

---

## 9. Final Interpretation

- The effective PF-ODE:
  - follows deterministic score flow **on average**,
  - most probable trajectories stay near deterministic paths,
  - but structured noise spreads probability mass.

Therefore:
- Training examples remain dominant modes,
- but probability mass fills in between them.

This explains **why diffusion models generalize without forgetting**.

---

## 10. One-Sentence Takeaway

> **Diffusion models generalize because randomness in score estimation induces an effective stochastic dynamics whose noise is strongest exactly where the model is uncertain, causing probability mass to spread between training examples while preserving their dominance.**
"""
