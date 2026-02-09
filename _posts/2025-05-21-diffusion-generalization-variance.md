---
layout: post
title: Generalization Through Variance in Diffusion Models
date: 2025-11-21 16:40:16
description: Paper summary for "Generalization through Variance in Diffusion Models"
tags: Learning
categories: sample-posts
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

Here $$\delta(\cdot)$$ is the Dirac delta distribution.

---

## 2. Forward diffusion

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

### 2.4 Reverse process as an ODE
$$
\dot x_t = -\beta_t x_t - D_t\, s(x_t, t),\quad t:T\to \epsilon.
$$
This is a Probability Flow ODE (PF-ODE), which is equivalent to the reverse SDE for $$p(x_t)$$ for all $$t$$.
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
Substitute $$\mu$$ and $$\Sigma$$:
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
\nabla_x p(x\mid t) = \int \nabla_x p(x\mid x_0,t)p_{\text{data}}(x_0)\, dx_0.
$$
Rewrite:
$$
\nabla_x p(x\mid x_0,t) = p(x\mid x_0,t)\nabla_x \log p(x\mid x_0,t) = p(x\mid x_0,t)\tilde s(x,t;x_0).
$$
Thus:
$$
\nabla_x p(x\mid t)=\int p(x\mid x_0,t)\tilde s(x,t;x_0)p_{\text{data}}(x_0)\, dx_0.
$$
But:
$$
p(x_0\mid x,t) = \frac{p(x\mid x_0,t)p_{\text{data}}(x_0)}{p(x\mid t)}.
$$
So divide by $$p(x\mid t)$$:
$$
\frac{\nabla_x p(x\mid t)}{p(x\mid t)} = \int \tilde s(x,t;x_0) p(x_0\mid x,t)\, dx_0.
$$
Left-hand side equals:
$$
\nabla_x \log p(x\mid t) = s(x,t).
$$
Therefore:
$$
s(x,t)=\mathbb E_{x_0\mid x,t}[\tilde s(x,t;x_0)].
$$

---

### 4.3 Covariance of proxy score: derive the formula

Define:
$$
C_{ij}(x,t) := \mathrm{Cov}_{x_0\mid x,t}[\tilde s_i, \tilde s_j]
= \mathbb E[\tilde s_i\tilde s_j] - s_i s_j,
$$
where expectations are under $$p(x_0\mid x,t)$$.

We want the identity:
$$
C(x,t)=S_t^{-1} + \nabla_x^2 \log p(x\mid t).
$$

#### Step 1: compute Hessian of log marginal
Start with:
$$
s(x,t)=\nabla_x \log p(x\mid t)=\frac{\nabla_x p(x\mid t)}{p(x\mid t)}.
$$
Differentiate again:
$$
\nabla_x^2 \log p
= \nabla_x\left(\frac{\nabla_x p}{p}\right)
= \frac{\nabla_x^2 p}{p} - \frac{(\nabla_x p)(\nabla_x p)^\top}{p^2}.
$$
So:
$$
\nabla_x^2 \log p = \frac{\nabla_x^2 p}{p} - s s^\top.
$$

#### Step 2: express $$\nabla_x^2 p(x\mid t)$$ in terms of conditional objects
We have:
$$
p(x\mid t)=\int p(x\mid x_0,t)p_{\text{data}}(x_0)\, dx_0.
$$
Differentiate twice:
$$
\nabla_x^2 p(x\mid t)=\int \nabla_x^2 p(x\mid x_0,t)p_{\text{data}}(x_0)\, dx_0.
$$

Now compute $$\nabla_x^2 p(x\mid x_0,t)$$:
$$
\nabla_x^2 p = \nabla_x(p\tilde s)= (\nabla_x p)\tilde s^\top + p\nabla_x\tilde s
= p\tilde s\tilde s^\top + p\nabla_x\tilde s.
$$
Thus:
$$
\nabla_x^2 p(x\mid x_0,t) = p(x\mid x_0,t)\left(\tilde s\tilde s^\top + \nabla_x\tilde s\right).
$$

Integrate:
$$
\nabla_x^2 p(x\mid t)=\int p(x\mid x_0,t)\left(\tilde s\tilde s^\top + \nabla_x\tilde s\right)p_{\text{data}}(x_0)\, dx_0.
$$
Divide by $$p(x\mid t)$$ to convert to posterior expectation:
$$
\frac{\nabla_x^2 p(x\mid t)}{p(x\mid t)}
= \mathbb E_{x_0\mid x,t}\left[\tilde s\tilde s^\top + \nabla_x\tilde s\right].
$$

#### Step 3: substitute into Hessian identity
Recall:
$$
\nabla_x^2 \log p = \frac{\nabla_x^2 p}{p} - s s^\top.
$$
So:
$$
\nabla_x^2 \log p
= \mathbb E[\tilde s\tilde s^\top + \nabla_x\tilde s] - s s^\top
= \mathrm{Cov}(\tilde s) + \mathbb E[\nabla_x\tilde s].
$$
Therefore:
$$
\mathrm{Cov}(\tilde s) = \nabla_x^2 \log p - \mathbb E[\nabla_x\tilde s].
$$

#### Step 4: compute $$\nabla_x\tilde s$$ for Gaussian conditional
We have:
$$
\tilde s(x,t;x_0)=S_t^{-1}(\alpha_t x_0 - x).
$$
Differentiate w.r.t. $$x$$:
$$
\nabla_x\tilde s = \nabla_x(-S_t^{-1} x) = -S_t^{-1}.
$$
This is constant (independent of $$x_0$$). So:
$$
\mathbb E_{x_0\mid x,t}[\nabla_x\tilde s] = -S_t^{-1}.
$$

Substitute:
$$
\mathrm{Cov}(\tilde s) = \nabla_x^2 \log p - (-S_t^{-1})
= S_t^{-1} + \nabla_x^2 \log p.
$$

So:
$$
C(x,t)=S_t^{-1} + \nabla_x^2 \log p(x\mid t).
$$

**Interpretation (from our discussion):**
- As $$t\to 0$$, $$S_t\to 0$$ so $$S_t^{-1}$$ blows up: proxy-score covariance large at small times.
- Large curvature $$\nabla_x^2 \log p$$ happens near modes and boundaries (e.g., for discrete mixtures).

---

### 4.4 Why proxy-score covariance matters for training and generalization

Even though $$\tilde s$$ is unbiased for $$s$$, training on $$J_1$$ uses noisy targets.
Empirically, finite neural nets trained on $$J_1$$ produce a sampling distribution different from the true-score PF-ODE.

The paper’s thesis: **structured noise in the target induces structured variance in the learned estimator**, which translates into stochasticity in typical sampling, yielding generalization.

---

## 5. PF-ODE sampling and the “distribution of outputs”

### 5.1 PF-ODE dynamics

Sampling uses the probability flow ODE:
$$
\dot x_t = -\beta_t x_t - D_t s(x_t,t),\quad t:T\to \epsilon.
$$
In practice, you plug in the learned score estimator $$\hat s_\theta$$:
$$
\dot x_t = -\beta_t x_t - D_t \hat s_\theta(x_t,t).
$$

### 5.2 Deterministic mapping given $$\theta$$

Fix:
- the network parameters $$\theta$$
- the initial noise seed $$x_T$$

Then the ODE solution is deterministic:
$$
x_0 = F(x_T,\theta).
$$

### 5.3 Why the output distribution uses a delta

Your question: why write
$$
\bar q(x_0\mid x_T) = \mathbb E_\theta[\delta(x_0 - F(x_T,\theta))]?
$$

Because this is the standard identity: for any random variable $$Y$$, its density can be written as:
$$
p_Y(y) = \mathbb E[\delta(y-Y)].
$$

#### Definition of Dirac delta
The delta is defined by its action under integration:
$$
\int \delta(x-a) f(x) dx = f(a).
$$
Thus, if $$Y$$ is deterministic with value $$a$$, its distribution is $$\delta(x-a)$$.

Here $$Y=F(x_T,\theta)$$ is random only through $$\theta$$; averaging those deltas gives the typical distribution.

---

## 6. Key technical tool: path integral representation of PF-ODE outputs (Eq. 6)

### 6.1 Why we need it (connect to your earlier confusion)
Directly averaging over $$F(x_T,\theta)$$ is hard because:
- PF-ODE typically has no closed-form solution.
- $$F$$ depends on $$\hat s_\theta$$ in a complicated, nonlinear, trajectory-dependent way.

So the paper rewrites the deterministic constraint “this path satisfies the ODE” into an integral form where $$\hat s_\theta$$ appears linearly in an exponent.

---

### 6.2 Discrete-time derivation of the path integral idea (most intuitive)

Discretize time:
$$
t_k = T - k\Delta t,\quad k=0,1,\dots,N,\quad t_N=\epsilon.
$$

Euler update for PF-ODE using estimator:
$$
x_{k+1} = x_k + \Delta t\left(\beta_{t_k} x_k + D_{t_k}\hat s_\theta(x_k,t_k)\right).
$$
Rearrange:
$$
x_{k+1} - x_k - \Delta t(\beta_{t_k}x_k + D_{t_k}\hat s_\theta(x_k,t_k)) = 0.
$$

Enforce each step by a delta:
$$
\delta\left(x_{k+1}-x_k-\Delta t(\beta_{t_k}x_k + D_{t_k}\hat s_\theta(x_k,t_k))\right).
$$

Now enforce delta via Fourier representation (finite-dimensional):
$$
\delta(y) = \int \frac{dp}{(2\pi)^D}\exp(ip\cdot y).
$$

Introduce an auxiliary $$p_k$$ per time step:
$$
\delta(\cdots) = \int dp_k \exp\left(ip_k\cdot\left[x_{k+1}-x_k-\Delta t(\beta_{t_k}x_k + D_{t_k}\hat s_\theta(x_k,t_k))\right]\right).
$$

Multiply over all steps and integrate over intermediate states $$\{x_k\}$$ and auxiliaries $$\{p_k\}$$.
In the continuous-time limit, this becomes the functional integral (Eq. 6):
$$
q(x_0\mid x_T;\theta)
=
\int \mathcal D[p_t]\mathcal D[x_t]\,
\exp\left(
\int_T^\epsilon i p_t\cdot[\dot x_t+\beta_t x_t + D_t \hat s_\theta(x_t,t)]dt
\right).
$$

#### Your question: “If the residual is zero, how do paths contribute anything?”
Because integrating over $$p_t$$ creates a delta-functional:
$$
\int \mathcal D[p_t]\exp\left(i\int p_t\cdot R_t[x]dt\right) \propto \delta[R_t[x]].
$$
This delta-functional equals “1” on paths where $$R_t[x]=0$$ (ODE satisfied), and “0” otherwise.
So ODE-consistent paths contribute by **surviving**, not by making the exponent nonzero.

---

## 7. Ensemble averaging: cumulant expansion to get Eq. 7

### 7.1 What is random?
The score estimator $$\hat s_\theta(x,t)$$ is random because $$\theta$$ depends on random finite training data (and optimization randomness).
So $$q(x_0\mid x_T;\theta)$$ is random, and we want its ensemble average:
$$
[q(x_0\mid x_T)] := \mathbb E_\theta[q(x_0\mid x_T;\theta)].
$$

### 7.2 Why the average becomes tractable
Inside Eq. 6, the dependence on $$\hat s_\theta$$ is linear in the exponent:
$$
\exp\left(\int i p_t\cdot D_t \hat s_\theta(x_t,t)dt\right).
$$
So we need to average an exponential of a linear functional:
$$
\mathbb E_\theta\left[\exp\left(\int J_t\cdot \hat s_\theta(x_t,t) dt\right)\right],
$$
where $$J_t := i D_t^\top p_t$$.

This is a **characteristic functional**.
Its log admits a cumulant expansion:
$$
\log \mathbb E[e^{Z}] = \kappa_1(Z) + \frac{1}{2}\kappa_2(Z) + \frac{1}{6}\kappa_3(Z)+\cdots,
$$
where $$\kappa_n$$ are cumulants.

With $$Z$$ linear in $$\hat s_\theta$$:
- $$\kappa_1$$ depends on the mean of $$\hat s_\theta$$
- $$\kappa_2$$ depends on the covariance of $$\hat s_\theta$$
- higher cumulants correspond to non-Gaussianity

Thus the paper obtains:
$$
[q(x_0\mid x_T)]
=
\int \mathcal D[p_t]\mathcal D[x_t]\,
\exp\left(M_1 - \frac{1}{2}M_2 + \cdots\right).
$$

### 7.3 Define $$s_{\text{avg}}$$ and V-kernel precisely
Mean score:
$$
s_{\text{avg}}(x,t) := [\hat s_\theta(x,t)] = \mathbb E_\theta[\hat s_\theta(x,t)].
$$

Covariance kernel:
$$
\mathrm{Cov}_\theta[\hat s(x,t),\hat s(x',t')] := \mathbb E_\theta[(\hat s-s_{\text{avg}})(\hat s'-s_{\text{avg}}')^\top].
$$

V-kernel (as defined by paper):
$$
V(x,t;x',t') := D_t \mathrm{Cov}_\theta[\hat s(x,t),\hat s(x',t')] D_{t'}.
$$

### 7.4 Write $$M_1$$ and $$M_2$$
$$
M_1 := \int_T^\epsilon i p_t\cdot[\dot x_t + \beta_t x_t + D_t s_{\text{avg}}(x_t,t)]dt,
$$
$$
M_2 := \int_T^\epsilon\int_T^\epsilon p_t^\top V(x_t,t;x_{t'},t') p_{t'} dt dt'.
$$

---

## 8. Gaussian approximation and the Effective SDE (Eq. 8 / Proposition 3.1)

### 8.1 Dropping higher cumulants
Assume the distribution of the estimator across training randomness is approximately Gaussian.
Then higher cumulants vanish, so only $$M_1$$ and $$M_2$$ remain.

### 8.2 From quadratic form to noise: the key idea
With a Gaussian field, averaging produces a quadratic term $$M_2$$ in the exponent.
Quadratic terms in an action correspond to Gaussian noise in an equivalent SDE description.

The result: sampling from $$[q(x_0\mid x_T)]$$ is equivalent to integrating an SDE:
$$
\dot x_t = -\beta_t x_t - D_t s_{\text{avg}}(x_t,t) + \xi(x_t,t),
$$
with:
$$
\mathbb E[\xi(x_t,t)]=0,
$$
$$
\mathbb E[\xi(x_t,t)\xi(x_{t'},t')^\top] = V(x_t,t;x_{t'},t').
$$

### 8.3 Key conceptual consequence (the paper’s main interpretive statement)
- If $$\hat s$$ is unbiased so $$s_{\text{avg}}=s$$,
- then deterministic drift matches the true PF-ODE,
- and the only difference is the noise term controlled by V-kernel.

Thus:
- $$V=0$$ implies the effective dynamics reduce to deterministic PF-ODE, which (for discrete training data) reproduces training examples (memorization).
- $$V\ne 0$$ implies added stochasticity, causing probability mass to spread and generalize.

---

## 9. Toy construction: Naive score estimator generalizes (Proposition 4.1)

This part constructs an explicit scheme showing how a nontrivial V-kernel arises even if one uses proxy scores directly.

### 9.1 Discretized PF-ODE sampling with Euler steps
Integrate backward with Euler:
$$
x_{t-\Delta t} = x_t + \Delta t\beta_t x_t + \Delta t D_t (\text{score used at time }t).
$$

### 9.2 At each step sample an endpoint $$x_{0t}$$ from the posterior
Sample:
$$
x_{0t} \sim p(x_0\mid x_t,t) = \frac{p(x_t\mid x_0,t)p_{\text{data}}(x_0)}{p(x_t\mid t)}.
$$
This captures the ambiguity of which training example could have generated the current noisy point.

### 9.3 Define the naive estimator used in that step
Construct:
$$
\hat s(x_t,t) := s(x_t,t) + \sqrt{\frac{\kappa}{\Delta t}}\left[\tilde s(x_t,t;x_{0t}) - s(x_t,t)\right].
$$

Interpretation:
- The expected estimator equals $$s$$ because $$\mathbb E[\tilde s\mid x,t]=s$$.
- But its variance is nonzero because $$\tilde s$$ varies with sampled $$x_{0t}$$.

### 9.4 Plug into the Euler update (explicit form)
The paper writes:
$$
x_{t-\Delta t} = x_t + \Delta t\beta_t x_t + \Delta t D_t s(x_t,t)
+ \sqrt{\kappa\Delta t}\, D_t\left[\tilde s(x_t,t;x_{0t})-s(x_t,t)\right].
$$

This is exactly of “drift + sqrt(Delta t) noise” form (Euler–Maruyama).

### 9.5 Derive $$s_{\text{avg}}=s$$
Compute ensemble mean over the fresh sampling of $$x_{0t}$$:
$$
[\hat s(x_t,t)] = s(x_t,t) + \sqrt{\frac{\kappa}{\Delta t}}\left[\mathbb E(\tilde s\mid x_t,t)-s\right] = s(x_t,t).
$$

So the mean drift is the true PF-ODE drift.

### 9.6 Derive the V-kernel for this scheme
Noise at time step is:
$$
\Delta x_{\text{noise}} = \sqrt{\kappa\Delta t}\, D_t(\tilde s - s).
$$
So its covariance (conditioned on $$x_t,t$$) is:
$$
\mathrm{Cov}(\Delta x_{\text{noise}})
= \kappa\Delta t\, D_t \mathrm{Cov}(\tilde s\mid x_t,t) D_t.
$$
But $$\mathrm{Cov}(\tilde s\mid x_t,t)=C(x_t,t)$$ where:
$$
C(x,t)=S_t^{-1} + \nabla_x^2 \log p(x\mid t).
$$

If the samples $$x_{0t}$$ are independent across time steps, then noise is white in time.
In continuous limit this yields:
$$
V(x_t,t;x_{t'},t') = \kappa D_t C(x_t,t) D_{t'} \delta(t-t').
$$

### 9.7 Interpret the result (why this implies generalization)
- Even though this scheme uses proxy score draws tied to training data (so “memorization-like”),
- the resampling injects noise proportional to proxy-score covariance.
- Proxy-score covariance is high in boundary regions between training examples.
So the effective sampling SDE has more diffusion in those boundary regions, which spreads mass between examples: that is generalization.

---

## 10. Final qualitative statement (connect to your last paragraph)

The effective dynamics:
- follow deterministic PF-ODE dynamics **in expectation** because noise mean is zero:
$$
\mathbb E[\xi]=0.
$$
- and are **most likely** to remain near deterministic PF-ODE paths because large deviations have lower probability under Gaussian noise.
Thus the model still samples near training examples most of the time, but with nonzero V-kernel it also places mass in-between.

This precisely matches the paper’s intended notion of “generalize but still prioritize training regions.”

---

## 11. Consolidated answers to the specific questions you raised

### 11.1 Why negative sign in forward SDE?
Because it is a mean-reverting drift term ensuring stability and convergence to noise. It does not imply $$x_0=-x_0$$; it implies $$\dot x_0$$ points toward 0.

### 11.2 Why marginalization formula $$p(x\mid t)=\int p(x\mid x_0,t)p_{\text{data}}(x_0)dx_0$$?
It is the law of total probability: marginalize out the latent clean point $$x_0$$.

### 11.3 Why proxy score is $$S_t^{-1}(\alpha_t x_0-x)$$?
It is the gradient of the log of a Gaussian density in $$x$$.

### 11.4 Why $$\mathbb E[\tilde s\mid x,t]=s$$?
Differentiate the mixture expression for $$p(x\mid t)$$ under the integral sign and use Bayes rule.

### 11.5 Why covariance formula $$C=S_t^{-1}+\nabla_x^2\log p$$?
Compute Hessian of log marginal and express it via posterior expectations; use $$\nabla_x \tilde s=-S_t^{-1}$$ for Gaussian conditional.

### 11.6 Why distribution of deterministic output is delta?
For fixed $$\theta,x_T$$ the output is exactly $$F(x_T,\theta)$$, so:
$$
q(x_0\mid x_T;\theta)=\delta(x_0-F(x_T,\theta)).
$$

### 11.7 Why ensemble average is $$\mathbb E_\theta[\delta(x_0-F(x_T,\theta))]$$ and not $$\mathbb E[F]$$?
Because we want the full distribution (mass placement), not the mean output. The delta identity defines the induced distribution.

### 11.8 Why path integral helps?
It replaces “solve a complicated ODE” with “integrate over paths constrained to satisfy the ODE,” turning dependence on $$\hat s_\theta$$ into a linear functional in an exponent so cumulant methods apply.

### 11.9 Why “if residual is zero, how can it contribute”?
Because integrating over $$p_t$$ enforces the constraint via a delta-functional; ODE-consistent paths survive with nonzero weight, inconsistent paths cancel out.

### 11.10 Why averaging $$\mathbb E_\theta[\exp(\int p_t\cdot D_t \hat s_\theta dt)]$$ is tractable?
Because it is a characteristic functional; its log expands in cumulants determined by mean and covariance of $$\hat s_\theta$$.

---

## 12. Final takeaway (paper’s central message)

Diffusion models trained with DSM generalize because the learned score estimator varies across training realizations, producing a nonzero V-kernel that acts like an additional noise term in an effective SDE for typical sampling. This noise is largest precisely in regions of high proxy-score covariance, especially boundaries between training examples, causing probability mass to spread beyond memorization while still following deterministic score flow on average.

