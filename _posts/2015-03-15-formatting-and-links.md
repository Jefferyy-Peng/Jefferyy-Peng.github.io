---
layout: post
title: a post with formatting and links
date: 2015-03-15 16:40:16
description: march & april, looking forward to summer
tags: formatting links
categories: sample-posts
---

Here is the Markdown code for the blog post based on the sources provided. You can copy and paste this directly into a `.md` file.

```markdown
# De-mystifying the Magic: How Diffusion Models Actually Work
**Based on Section 2 of "Factorization and Compositional Generalization in Diffusion Models"**

If you have ever typed "an astronaut riding a horse on Mars" into DALL-E 2 or Stable Diffusion and watched a masterpiece appear, you have witnessed the power of **Diffusion Models**. But how do they turn random static into art?

In this post, we will walk through the core logic of these models—from the math of "destroying" data to the state-of-the-art architectures that reconstruct it.

---

## 2.1 Formulations of Diffusion Models

The intuition behind diffusion models is surprisingly simple: **destroy the data, then learn to fix it**. Formally, there are three main ways to describe this process.

### 2.1.1 Denoising Diffusion Probabilistic Models (DDPMs)
This is the most common framework. Imagine taking a crisp photo ($x_0$) and slowly adding "snow" (Gaussian noise) to it over many steps ($t=1, ..., T$).

**The Forward Process (Destruction)**
We add noise iteratively using a **Markov chain**, meaning each step only depends on the previous one. We use a "transition kernel" $q$ to add this noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$$

Here, $\beta_t$ tells us how much noise to add. Eventually, at step $T$, the image is destroyed—it is just pure Gaussian noise ($x_T \approx \mathcal{N}(0, I)$).

> **Figure 2.1 Visualization:** Imagine a sequence of images. On the far left is a cat. As you move right, the image becomes grainier. On the far right, it looks like a TV tuned to a dead channel. That is the forward process.

**The Backward Process (Creation)**
The generative magic happens in reverse. The model learns to reverse time, starting from noise and removing it step-by-step to recover the image. A neural network predicts the mean ($\mu_\theta$) and variance ($\Sigma_\theta$) for the reverse step:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**The Objective**
How do we train it? We basically teach the neural network ($\epsilon_\theta$) to look at a noisy image and **guess the noise** that was added to it. The loss function measures how close the guess is to the real noise:

$$L = \mathbb{E} [ ||\epsilon - \epsilon_\theta(x_t, t)||^2 ]$$

### 2.1.2 Score-based Generative Models (SGMs)
Instead of predicting noise directly, SGMs estimate the **score function**: the gradient of the log-density of the data ($\nabla_x \log p(x)$).
To generate an image, we use **Langevin Dynamics**. We start with a random point (noise) and iteratively move it toward high-density regions (real data) using the estimated score.

### 2.1.3 Stochastic Differential Equations (SDEs)
If we add noise in infinite, tiny steps rather than discrete ones, the process becomes continuous.
*   **Forward:** Described by a differential equation $dx = f(x, t)dt + g(t)dw$.
*   **Reverse:** Solved using a "probability flow ODE" to turn noise back into data.

---

## 2.2 Conditional Diffusion Models

Generating random images is fun, but usually, we want control (e.g., "generate a *dog*"). We achieve this by feeding a condition vector $c$ (like a class label or text embedding) into the model,.

There are two main ways to guide the AI:

### 2.2.1 Classifier Guidance
This method uses a separate **classifier** trained to identify noisy images. If you want a "cat," the classifier looks at the current noisy image and calculates the gradient to push the image to look more like a cat ($\nabla_{x_t} \log p_\phi(c | x_t)$). We subtract this gradient from the noise prediction,.

### 2.2.2 Classifier-free Guidance
This is the method used by modern giants like DALL-E 2. Instead of a separate classifier, we train the diffusion model to handle both specific prompts ($c$) and unconditional inputs (by dropping the label, $c=\emptyset$).
During generation, we mix the two predictions. We push the model away from the "unconditional" result and toward the "conditional" result using a guidance weight $w$:

$$\tilde{\epsilon}_\theta(x_t, c) = (1 + w)\epsilon_\theta(x_t, c) - w\epsilon_\theta(x_t)$$

This makes the images stick much closer to your text prompt!

---

## 2.3 Text-to-Image Generative Models

How are these mathematical principles applied in the real world?

### 2.3.1 Pixel-based Models
These models work directly on the high-resolution pixels.
*   **GLIDE (OpenAI):** Uses classifier-free guidance to generate photorealistic images. It showed that we don't need complex separate classifiers to get great results.
*   **Imagen (Google):** This model made a huge discovery—**the text encoder matters more than the image generator**. By using a massive Large Language Model (T5-XXL) to understand the prompt, Imagen could generate highly accurate concepts.

> **Figure 2.4 Example:** Imagen can generate a "brain riding a rocketship heading towards the moon" or "a dragon fruit wearing a karate belt." This proves it "understands" the text deeply.

### 2.3.2 Latent-based Models
Working with pixels is slow and expensive. **Latent Diffusion Models (LDMs)** solve this by compressing images into a smaller "latent space" (using an Autoencoder) and doing the diffusion there.
*   **Stable Diffusion:** Based on LDM. It uses **cross-attention** to inject text prompts into the generation process. Because it runs in a compressed space, it is incredibly efficient.
*   **DALL-E 2 (unCLIP):** Uses OpenAI's **CLIP** model. It first converts text into a CLIP image embedding, then decodes that embedding into an image.

> **Figure 2.5 Architecture:** Think of LDM as a sandwich. The top and bottom bread are the Encoder and Decoder (handling pixels). The meat in the middle is the Diffusion Process, operating in the efficient, compressed Latent Space,.

### 2.3.3 Failure Modes
Even these powerful models make funny mistakes. Section 2.3.3 highlights several "failures" that reveal how the models think:

1.  **Binding Issues:** If you ask for a "red cube on a blue cube," DALL-E 2 often mixes them up (e.g., a blue cube on a red cube). It struggles to "bind" the attribute (color) to the correct object.
2.  **Text Generation:** If you ask for a sign that says "Deep Learning," the model might produce gibberish like "Diep Lerpt." It mimics the *look* of letters but doesn't fully understand spelling,.
3.  **Unusual Scenarios:** Ask GLIDE for a "car with triangular wheels," and it will likely draw circular wheels. The model is biased by its training data—it has seen millions of round wheels and refuses to believe triangles can roll!.
4.  **Physics Violations:** Reflections in mirrors sometimes show the wrong object or angle, proving the model lacks a true 3D understanding of the world.

***

**Summary:** Diffusion models work by mastering the art of noise removal. Whether working in pixel space (Imagen) or latent space (Stable Diffusion), and using clever tricks like classifier-free guidance, they have revolutionized AI art. However, their struggle with logic (binding) and physics shows we still have work to do!
```