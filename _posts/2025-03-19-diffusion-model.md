---
layout: post
title: How Diffusion Models Work
date: 2025-03-19 16:40:16
description: A learning notebook for diffusion models
tags: Learning
categories: sample-posts
---

As part of my research into the mechanisms of compositionality in generative AI, I am conducting a review of the foundational literature surrounding Diffusion Models. While these models have achieved ubiquity in text-to-image generation, understanding their underlying mathematical formulations and architectural constraints is essential for addressing their current limitations in factorization and generalization.

This post summarizes the theoretical framework of diffusion models, covering the probabilistic definitions, conditioning mechanisms, and the trade-offs between pixel-space and latent-space architectures.

---

## 2.1 Formulations of Diffusion Models

Diffusion models can be formalized through three primary frameworks: **Denoising Diffusion Probabilistic Models (DDPMs)**, **Score-based Generative Models (SGMs)**, and **Stochastic Differential Equations (SDEs)**.

### 2.1.1 Denoising Diffusion Probabilistic Models (DDPMs)
The DDPM framework defines a forward Markov chain that gradually adds Gaussian noise to data $$x_0$$ until it approaches an isotropic Gaussian distribution $$x_T$$.
The transition kernel is defined as:
$$
q(x_t \mid x_{t-1})=\mathcal{N}\!\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right),
$$ where $$\beta_t \in (0,1)$$ controls the noise variance.

Let $$\alpha_t := 1-\beta_t$$ and $$\bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s$$.
Then the marginal distribution is:

$$
q(x_t \mid x_0)=\mathcal{N}\!\left(x_t;\sqrt{\bar{\alpha}_t}\,x_0,(1-\bar{\alpha}_t)I\right).
$$

Using the reparameterization trick with $$\epsilon \sim \mathcal{N}(0,I)$$,
we can write:

$$
x_t=\sqrt{\bar{\alpha}_t}\,x_0
+
\sqrt{1-\bar{\alpha}_t}\,\epsilon.
$$

The generative capability arises from learning a parameterized reverse Markov chain. The model estimates the transition kernel $$p_\theta(x_{t-1}|x_t)$$ to iteratively denoise the latent variables:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Training is typically performed by optimizing a variational lower bound (ELBO) on the negative log-likelihood.
>**Step 1**: Marginalize likelihood: the probability of $$x_0$$ is obtained by integrating out all latent variables $$x_{1:T}$$. This integral is generally intractable.
$$
\mathbb{E}\big[-\log p_\theta(x_0)\big]
= \mathbb{E}\left[-\log \int p_\theta(x_{0:T}) \, dx_{1:T}\right]
$$
**Step 2**: Introduce an auxiliary (variational) distribution $$q(x_{1:T}\mid x_0)$$ by multiplying and dividing inside the integral.
$$
= \mathbb{E}\left[
-\log \int p_\theta(x_{0:T}) 
\frac{q(x_{0:T})}{q(x_{0:T})}
\, dx_{1:T}
\right]
$$
**Step 3**: Apply Jensen’s inequality $$-\log$$ is convex, moving the log inside the expectation. This produces an upper bound on the negative log-likelihood. 
$$
\le \mathbb{E}_q\left[
-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right]
$$
**Step 4**: Next, factorize the model and variational distributions
$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t),
\quad
q(x_{1:T}\mid x_0) = \prod_{t=1}^{T} q(x_t\mid x_{t-1})
$$
**Step 5**: Substituting these factorizations:
$$
= \mathbb{E}_q\left[
-\log p(x_T)- \sum_{t=1}^{T}
\log \frac{p_\theta(x_{t-1}\mid x_t)}
{q(x_t\mid x_{t-1})}
\right]
 := \mathcal{L}
$$
The first term $$\mathbb{E}_q[-\log p(x_T)]$$ is fixed and not optimizable, so we optimize the second term:
$$
  \mathbb{E}_{q(x_{t-1}, x_t)}
\left[
\log \frac{q(x_t \mid x_{t-1})}
{p_\theta(x_{t-1} \mid x_t)}
\right]  =
\mathbb{E}_{q(x_0, x_t)}
\Big[
\mathrm{KL}\big(
q(x_{t-1} \mid x_t, x_0)
\;\|\;
p_\theta(x_{t-1} \mid x_t)
\big)
\Big]
+
\text{const}$$
this is equavalent to a noise-prediction objective, where a neural network $$\epsilon_\theta$$ (typically a UNet) minimizes the error between the added noise $$\epsilon$$ and the predicted noise:
$$L = \mathbb{E} [ \lambda(t)||\epsilon - \epsilon_\theta(x_t, t)||^2 ]$$

### 2.1.2 Score-based Generative Models (SGMs) and SDEs
Alternative formulations focus on the gradient of the log-density of the data instead of noise being added, known as the score function $$\nabla_x \log p(x)$$.
*   **SGMs:** In score-based generative modeling, we train a **Noise-Conditioned Score Network (NCSN)** $$s_\theta(x,t)$$ to approximate the score (gradient of the log-density) of data corrupted by Gaussian noise. The classical score-matching objective is

$$
\frac{1}{2}\,\mathbb{E}_{p_{\text{data}}}
\big\|
s_\theta(x) - \nabla_x \log p_{\text{data}}(x)
\big\|^2 .
$$

Let $$x_0 \sim q(x_0)$$ denote clean data, and define a family of noisy distributions via

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; x_0, \sigma_t^2 I),
\qquad
q(x_t) = \int q(x_t \mid x_0) q(x_0)\,dx_0 .
$$

Conditioning the score network on the noise level $$t$$, the score-matching objective becomes (up to a constant factor):

$$
\mathbb{E}_{t \sim \mathcal{U}[1,T],\, x_0 \sim q(x_0),\, x_t \sim q(x_t \mid x_0)}
\left[\lambda(t)\,
\big\|
s_\theta(x_t, t) - \nabla_x \log q(x_t \mid x_0)
\big\|^2
\right].
$$

The relation between DDPM and SGM is then 

$$
\boxed{
\epsilon_\theta(x,t) = -\,\sigma_t\, s_\theta(x,t)
}
$$

*   **SDEs:** While DDPMs and SGMs are originally defined using a **finite, discretized noising process**, the diffusion process can be equivalently formulated in **continuous time** as a stochastic differential equation (SDE). This continuous view unifies diffusion models and enables flexible sampling algorithms.

#### General Forward Diffusion SDE

The forward noising process is described by the **Score SDE**:

$$
dx = f(x,t)\,dt + g(t)\,dw,
$$

where:
- $$f(x,t)$$ is the **drift (diffusion) term**,
- $$g(t)$$ controls the **noise magnitude**,
- $$w$$ is a standard Wiener process.

This SDE defines a family of marginal distributions $$q_t(x)$$ over time.

#### DDPM as a Special Case

The continuous-time limit of DDPM corresponds to the SDE:

$$
dx = -\frac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw,
$$

where $$\beta(t)$$ is a continuous noise schedule. This formulation mirrors the discrete DDPM noising process in the limit of infinitely many steps.

#### SGM as a Special Case

Score-based Generative Models (SGMs) correspond to the SDE:

$$
dx = \sqrt{\frac{d[\sigma_t^2]}{dt}}\,dw,
$$

where $$\sigma(t)$$ is the continuous noise scale. Here, the forward process is a pure diffusion without drift.

#### Reverse-Time SDE (Sampling Process)

For any forward SDE of the form above, the **reverse-time SDE** is given by:

$$
dx = \big[f(x,t) - g(t)^2 \nabla_x \log q_t(x)\big]\,dt + g(t)\,d\bar{w},
$$

where:
- $$\nabla_x \log q_t(x)$$ is the **score function**,
- $$\bar{w}$$ is a backward-time Wiener process.

Learning the score function enables generation of samples by simulating this reverse SDE.

#### Probability Flow ODE

In addition to the reverse SDE, there exists an equivalent **deterministic ODE** with identical marginal distributions:

$$
dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log q_t(x)\right]\,dt.
$$

This **probability flow ODE** allows sampling without stochasticity, using standard ODE solvers.

#### Sampling Methods

Given the learned score function, samples can be generated using:
- Reverse-time SDE solvers,
- Probability flow ODE solvers,
- Annealed Langevin Dynamics,
- Predictor–Corrector (PC) samplers combining SDE solvers with MCMC methods (e.g., Langevin MCMC or HMC).

## 2.2 Conditional Generation Mechanisms

To enable controllable generation (e.g., text-to-image synthesis), diffusion models must incorporate a conditional vector $c$. The reverse process is modified to $p_\theta(x_{t-1} | x_t, c)$. Two primary guidance methods dominate the literature:

### 2.2.1 Classifier Guidance
This approach leverages an auxiliary classifier $p_\phi(c | x_t)$ trained on noisy images. The denoising process can be expressed:
$$p_{\theta,\phi}(x_t \mid x_{t+1}, c)=Z \, p_\theta(x_t \mid x_{t+1}) \, p_\phi(c \mid x_t)$$
where $$Z$$ is a normalization constant.
Taking the gradient of the log of the conditional density (ignoring $$Z$$):

$$
\nabla_{x_t} \log \big(
p_\theta(x_t \mid x_{t+1}) \, p_\phi(c \mid x_t)
\big)=\nabla_{x_t} \log p_\theta(x_t \mid x_{t+1})
+
\nabla_{x_t} \log p_\phi(c \mid x_t).
$$

Using the relationship between score and noise prediction,

$$
\epsilon_\theta(x_t, t) = -\sigma_t s_\theta(x_t, t),
$$

the gradient becomes:

$$
=-\frac{1}{\sigma_t}\,\epsilon_\theta(x_t, t)+\nabla_{x_t} \log p_\phi(c \mid x_t).
$$
>**Key Intuition:** beside the original unconditional score function, the gradient of the classifier is added to guide the denoise process.

### 2.2.2 Classifier-free Guidance
To avoid the computational cost and complexity of training a separate noise-robust classifier, Ho and Salimans (2022) proposed classifier-free guidance. Here, a single diffusion model is trained to handle both conditional and unconditional inputs (where $c = \emptyset$). During sampling, the noise prediction is a linear combination of both outputs, weighted by a scale $w$:

$$\tilde{\epsilon}_\theta(x_t, c) = (1 + w)\epsilon_\theta(x_t, c) - w\epsilon_\theta(x_t)$$

This method implicitly maximizes the probability of the condition without an external classifier and has become the standard for state-of-the-art models.
>**Key Intuition:** trains one diffusion model to operate in two modes: with condition and without condition. During sampling, the model compares these two predictions. The difference tells you how the condition should push the sample, and scaling that difference lets you control how strongly the generation follows the condition. In effect, the model learns its own “internal classifier gradient” and uses it to guide sampling—without ever training an explicit classifier.

## 2.3 State-of-the-Art Architectures

Current text-to-image systems are generally categorized by whether the diffusion process occurs in pixel space or latent space.

### 2.3.1 Pixel-based Models
These models operate directly on high-dimensional image data.
*   **GLIDE:** Uses classifier-free guidance to generate photorealistic images and demonstrates capabilities in text-guided inpainting [15].
*   **Imagen:** A key finding from the development of Imagen is the scaling law regarding text encoders. The authors discovered that increasing the size of the language model (e.g., using T5-XXL) yields greater improvements in image fidelity and image-text alignment than increasing the size of the visual diffusion model itself [16].

### 2.3.2 Latent-based Models
To address the high computational costs of pixel-space diffusion, **Latent Diffusion Models (LDMs)** utilize an autoencoder to project data into a lower-dimensional latent space [18].
*   **Stable Diffusion:** Applies the diffusion process within this compressed latent space, utilizing cross-attention mechanisms to incorporate text conditioning. This architecture significantly improves inference efficiency [19].
*   **DALL-E 2 (unCLIP):** Utilizes the CLIP embedding space. It generates an image embedding from text and then decodes this embedding into an image, leveraging the joint multimodal space learned by CLIP [20].

## 2.4 Failure Modes and Limitations

Despite the fidelity of these models, systematic evaluations reveal persistent limitations in their reasoning capabilities:

1.  **Attribute Binding:** Models frequently fail to correctly bind attributes to objects. For example, in a prompt specifying a "red cube" and a "blue cube," DALL-E 2 may swap the colors or textures [22], [23].
2.  **Text Rendering:** While semantic understanding is high, the ability to render coherent alphanumeric text remains poor, likely due to tokenization schemes (BPE encoding) that obscure spelling information from the model [23], [24].
3.  **Physical Consistency:** Generated images often exhibit violations of physical laws, such as incorrect shadow placement or reflections that do not align with the object's geometry [25].
4.  **Bias toward Canonical Forms:** When prompted with unusual scenarios (e.g., "a car with triangular wheels"), models often revert to the mean of the training distribution (circular wheels), indicating a lack of true compositional generalization [26].

These failure modes suggest that while diffusion models excel at texture synthesis and semantic association, they struggle with precise factorization and compositional reasoning—a core focus of the subsequent chapters of this thesis [27].