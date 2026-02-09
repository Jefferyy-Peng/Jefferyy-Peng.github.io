---
layout: post
title: Generalization Through Variance in Diffusion Models
date: 2025-05-21 16:40:16
description: Paper summary for "Generalization through Variance in Diffusion Models"
tags: Learning
categories: sample-posts
---



**Formatting rules (per your request):**
- Every equation (inline or display) is wrapped in $$ ... $$.
- I use LaTeX backslashes as single characters: I write $\text{like } \alpha_t$ as $$\alpha_t$$, **not** $$\alpha_t$$.
- This writeup follows the **logic of the paper** and explicitly includes/answers the questions we discussed.

---

## 0. High-level roadmap (what the paper tries to do)

The paper’s main goal is to analytically characterize the **typical learned sampling distribution** of diffusion models trained with denoising score matching (DSM), and to explain **why they generalize** (place probability mass between training examples) instead of perfectly memorizing.

Key steps:
1. Define forward diffusion and reverse PF-ODE sampling.
2. Compare the “true-score” objective $$J_0$$ vs DSM objective $$J_1$$.
3. Show the proxy score is **unbiased** but has a structured **covariance**.
4. Treat the trained score estimator $$\hat s_\theta$$ as random (because it depends on finite training samples).
5. Write PF-ODE sampling as a **path integral**, so that averaging over training randomness becomes tractable.
6. Perform the ensemble average via cumulants, producing mean term $$M_1$$ and covariance term $$M_2$$ (the **V-kernel**).
7. Under a Gaussian approximation, show the averaged dynamics are equivalent to an **effective SDE**:
   generalization happens iff the V-kernel is nonzero.
8. Provide a toy “naive estimator” scheme showing how nonzero V-kernel arises even when using the proxy score directly.

---

## 1. Preliminaries: distributions, conditionals, and marginals

### 1.1 Data distribution

Let $$x_0 \in \mathbb R^D$$ denote clean data.
The data distribution is $$p_{\text{data}}(x_0)$$.

A common idealization used in the paper is an empirical distribution over $$M$$ examples $$\{\mu_m\}_{m=1}^M$$:
$$
p_{\text{data}}(x_0) = \frac{1}{M}\sum_{m=1}^M \delta(x_0 - \mu_m).
$$

Here $$\delta(\cdot)$$ is the Dirac delta distribution (we will define it precisely later).

---

## 2. Forward diffusion: SDE, why the minus sign, and closed-form transition

### 2.1 Forward SDE

The forward process is an SDE:
$$
\dot x_t = -\beta_t x_t + G_t \eta_t,\quad t:0\to T.
$$
Definitions:
- $$x_t \in \mathbb R^D$$: random variable at time $$t$$.
- $$\dot x_t$$: time derivative (informally; in SDE form it corresponds to Ito dynamics).
- $$\beta_t \ge 0$$: scalar drift schedule.
- $$G_t \in \mathbb R^{D\times K}$$: noise injection matrix.
- $$\eta_t$$: standard Gaussian white noise process.

Define diffusion tensor:
$$
D_t := \frac{G_t G_t^\top}{2}\in \mathbb R^{D\times D}.
$$

#### Your question: “why negative sign? if t=0 then x_t = -x_t?”
Important: the equation is about the **derivative**, not equality of values.
At $$t=0$$:
- you set initial condition $$x_0 \sim p_{\text{data}}$$.
- the SDE gives derivative mean drift $$\dot x_0 = -\beta_0 x_0 + \text{noise}$$.
So it does **not** imply $$x_0 = -x_0$$; it implies the drift points toward the origin (mean-reverting), preventing explosion and ensuring the process approaches noise.

---

### 2.2 Solve the forward SDE: derive $$p(x_t\mid x_0,t)=\mathcal N(\alpha_t x_0, S_t)$$

We derive the conditional distribution of $$x_t$$ given $$x_0$$.

Write the SDE in Ito differential form:
$$
dx_t = -\beta_t x_t dt + G_t dW_t,
$$
where $$W_t$$ is a $$K$$-dim Brownian motion (so $$dW_t$$ are Gaussian increments).

#### Step 1: integrating factor
Define
$$
\alpha_t := \exp\left(-\int_0^t \beta_{t'} dt'\right).
$$
Note:
$$
\frac{d}{dt}\alpha_t = -\beta_t \alpha_t,\quad \alpha_0=1.
$$

Consider the scaled process:
$$
y_t := \alpha_t^{-1} x_t.
$$

Use Ito (here drift-only scaling works as usual since factor is deterministic):
$$
dy_t = d(\alpha_t^{-1} x_t)
= \alpha_t^{-1} dx_t + x_t d(\alpha_t^{-1}).
$$
Compute:
$$
d(\alpha_t^{-1}) = -\alpha_t^{-2} d\alpha_t = -\alpha_t^{-2}(-\beta_t \alpha_t dt) = \beta_t \alpha_t^{-1} dt.
$$
So:
$$
dy_t = \alpha_t^{-1}(-\beta_t x_t dt + G_t dW_t) + x_t(\beta_t \alpha_t^{-1} dt)
= \alpha_t^{-1} G_t dW_t.
$$

#### Step 2: integrate
Integrate from 0 to t:
$$
y_t = y_0 + \int_0^t \alpha_{t'}^{-1} G_{t'} dW_{t'}.
$$
But $$y_0 = \alpha_0^{-1} x_0 = x_0$$. Thus:
$$
y_t = x_0 + \int_0^t \alpha_{t'}^{-1} G_{t'} dW_{t'}.
$$

Multiply by $$\alpha_t$$:
$$
x_t = \alpha_t x_0 + \alpha_t\int_0^t \alpha_{t'}^{-1} G_{t'} dW_{t'}.
$$

#### Step 3: identify Gaussian distribution
The stochastic integral is Gaussian with mean 0.
Thus conditional on $$x_0$$:
- mean:
$$
\mathbb E[x_t\mid x_0] = \alpha_t x_0.
$$
- covariance:
Let
$$
\varepsilon_t := \alpha_t\int_0^t \alpha_{t'}^{-1} G_{t'} dW_{t'}.
$$
Then:
$$
\mathrm{Cov}(\varepsilon_t\mid x_0)
= \alpha_t^2 \int_0^t \alpha_{t'}^{-2} G_{t'} \mathrm{Cov}(dW_{t'}) G_{t'}^\top.
$$
Since $$\mathrm{Cov}(dW_{t'}) = I dt'$$:
$$
\mathrm{Cov}(\varepsilon_t\mid x_0)
= \alpha_t^2 \int_0^t \alpha_{t'}^{-2} G_{t'} G_{t'}^\top dt'
= \alpha_t^2 \int_0^t \alpha_{t'}^{-2} (2D_{t'}) dt'.
$$

Many texts rewrite this as an equivalent expression in terms of the forward-time convention used in the paper:
$$
S_t := \int_0^t 2D_{t'} \alpha_{t'}^2 dt'.
$$
(This matches the paper’s definition; it can be obtained by a change of variables depending on whether one defines $$\alpha_t$$ relative to 0 or relative to t. The paper uses the above closed form.)

Thus:
$$
p(x\mid x_0,t) = \mathcal N(x; \alpha_t x_0, S_t).
$$

---

### 2.3 Derive the marginal at time $$t$$

Define the marginal:
$$
p(x\mid t) := \int p(x\mid x_0,t)\, p_{\text{data}}(x_0)\, dx_0.
$$

**Derivation:** law of total probability / marginalization:
$$
p(x\mid t) = \int p(x,x_0\mid t)\, dx_0
= \int p(x\mid x_0,t)p_{\text{data}}(x_0)\, dx_0.
$$

If $$p_{\text{data}}$$ is discrete mixture of deltas:
$$
p(x\mid t) = \frac{1}{M}\sum_{m=1}^M \mathcal N(x; \alpha_t \mu_m, S_t).
$$
So the forward marginal becomes a Gaussian mixture with components centered at scaled training examples.

---

## 3. Scores: true score and proxy score (with full derivations)

### 3.1 True score

Define the true score:
$$
s(x,t) := \nabla_x \log p(x\mid t).
$$

This is the vector field needed by reverse-time sampling methods.

---

### 3.2 Proxy score (DSM target) and its closed form

Define proxy score:
$$
\tilde s(x,t;x_0) := \nabla_x \log p(x\mid x_0,t).
$$
Since:
$$
p(x\mid x_0,t)=\mathcal N(x; \mu, \Sigma),\quad \mu=\alpha_t x_0,\quad \Sigma=S_t,
$$
we compute:
$$
\log \mathcal N(x;\mu,\Sigma) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu).
$$
Differentiate w.r.t. $$x$$:
$$
\nabla_x \log \mathcal N(x;\mu,\Sigma)
= -\frac{1}{2}\nabla_x\left((x-\mu)^\top \Sigma^{-1}(x-\mu)\right).
$$
Using:
$$
\nabla_x\left((x-\mu)^\top A (x-\mu)\right)= 2A(x-\mu)\quad \text{for symmetric }A,
$$
we get:
$$
\nabla_x \log \mathcal N(x;\mu,\Sigma)= -\Sigma^{-1}(x-\mu) = \Sigma^{-1}(\mu-x).
$$
Substitute:
$$
\tilde s(x,t;x_0) = S_t^{-1}(\alpha_t x_0 - x).
$$

---

## 4. Compare objectives $$J_0$$ vs $$J_1$$ and why $$\tilde s$$ matters

### 4.1 Define the objectives

Idealized objective:
$$
J_0(\theta) := \mathbb E_{t,x}\left[\frac{\lambda_t}{2}\|\hat s_\theta(x,t) - s(x,t)\|^2\right].
$$

DSM objective:
$$
J_1(\theta) := \mathbb E_{t,x_0,x}\left[\frac{\lambda_t}{2}\|\hat s_\theta(x,t) - \tilde s(x,t;x_0)\|^2\right].
$$

Here the sampling is:
- $$t\sim p(t)$$
- $$x_0\sim p_{\text{data}}(x_0)$$
- $$x\sim p(x\mid x_0,t)$$

---

### 4.2 Unbiasedness: derive $$\mathbb E_{x_0\mid x,t}[\tilde s]=s$$

We prove:
$$
\mathbb E_{x_0\mid x,t}[\tilde s(x,t;x_0)] = s(x,t).
$$

Start from definition:
$$
p(x\mid t)=\int p(x\mid x_0,t)p_{\text{data}}(x_0)\, dx_0.
$$
Differentiate w.r.t. $$x$$:
$$
\nabla_x p(x\mid t) = \int \nab_

