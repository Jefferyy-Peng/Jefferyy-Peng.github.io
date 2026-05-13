---
layout: post
title: Mechanistic Understanding of Hallucination in Multimodal Models
date: 2026-05-02 
description: Idea
tags: Hallucination
categories: Idea
---

**TL;DR:** 

# Research Idea 1: Mechanistic Understanding of Hallucination in Multimodal Models

## Working title

**When Do Multimodal Models Hallucinate? Mechanistic Statistics for Separating Grounded and Hallucinated Responses**

## Core research question

Multimodal large language models (MLLMs / LVLMs) often hallucinate: they mention objects, attributes, relations, or reasoning conclusions that are not supported by the image. The goal of this project is not only to detect hallucination from outputs, but to understand **what changes inside the model** when hallucination happens.

The central question is:

> Can internal model statistics separate hallucinated and non-hallucinated generations before or during decoding?

Examples of candidate statistics:

- Attention pattern between text tokens and image tokens
- Vision-aware attention head behavior
- Token-level attribution to image vs. language context
- Circuit attribution / causal edge attribution
- Representation geometry of grounded vs. hallucinated tokens
- Logit contribution from visual tokens vs. language prior
- Layer-wise shift from visual evidence to textual prior

## Motivation

Current hallucination work in MLLMs focuses heavily on:

1. **Benchmarks**: measuring whether a model hallucinates.
2. **Detection**: classifying an output as hallucinated or not.
3. **Mitigation**: decoding or training methods that reduce hallucination.

But for a research paper, a stronger mechanistic question is:

> When the model hallucinates, does it fail because it cannot see the visual evidence, or because later language-model components override the visual evidence?

This is important because two visually similar failures may have different causes:

- **Perception failure**: the visual encoder or cross-modal projection does not represent the object.
- **Grounding failure**: the model sees the object but does not use the visual signal.
- **Language-prior failure**: the decoder follows a likely textual pattern instead of the image.
- **Reasoning failure**: the model grounds local objects correctly but draws an unsupported conclusion.
- **Instruction/prompt-induced failure**: the prompt biases the model toward a wrong answer.

## Related work landscape

### Hallucination benchmarks

Useful benchmarks for labeling hallucinated vs. non-hallucinated outputs:

- **CHAIR**: classic object hallucination metric for image captioning.
- **POPE**: evaluates object hallucination using yes/no probing.
- **AMBER**: evaluates hallucination across existence, attribute, and relation hallucination.
- **MMHal-Bench**: evaluates open-ended hallucination in multimodal responses.
- **HallusionBench**: tests visual illusion, knowledge conflict, and reasoning-related hallucination.
- **MME-Hall / MME variants**: commonly used for MLLM hallucination evaluation.

AMBER is useful because it explicitly separates hallucination types such as existence, attribute, and relation hallucination. POPE and CHAIR are useful for object hallucination but narrower.

### Detection and mitigation work

Existing methods often use external evaluators, uncertainty, contrastive decoding, or attention-based signals. Examples include:

- **VADE**: uses visual grounding information encoded in transformer attention maps for hallucination detection.
- **DAMRO**: analyzes attention relationships between visual encoder and LLM decoder and uses this for hallucination mitigation.
- **VCD / LCD / ICD / CICD / SECOND-style methods**: reduce hallucination by contrasting visual or language-prior distributions during decoding.
- **Internal representation / attention divergence methods**: use internal attention or hidden-state differences as hallucination indicators.

These papers suggest that internal signals are informative, but many of them are designed mainly for detection or mitigation rather than a fine-grained mechanistic explanation.

### Mechanistic analysis work

Some recent work is closer to the proposed direction:

- Studies on prompt-induced hallucination identify specific attention heads whose ablation can reduce hallucination.
- Vision-aware head divergence metrics attempt to quantify whether attention heads are sensitive to visual context.
- Causal mediation analysis has been used to test which modules or representations contribute to hallucination.
- Some work argues that LVLM hallucination is often caused by language-prior dominance rather than total visual blindness.

This makes the proposed project plausible: hallucination may have identifiable internal signatures.

## Proposed hypothesis

The main hypothesis:

> Hallucinated tokens have lower causal dependence on image representations and higher dependence on language-prior circuits than grounded tokens.

More specific hypotheses:

1. **Attention hypothesis**  
   Hallucinated object/attribute tokens attend less to relevant image patches or attend more diffusely than grounded tokens.

2. **Attribution hypothesis**  
   For hallucinated tokens, the logit attribution from image tokens or vision-aware heads is lower, while attribution from previous text tokens is higher.

3. **Circuit hypothesis**  
   Grounded generations rely on circuits connecting image tokens to answer tokens; hallucinated generations rely more on text-only or prompt-copying circuits.

4. **Representation hypothesis**  
   Grounded and hallucinated token states become separable in hidden-state space at middle-to-late decoder layers.

5. **Language-prior override hypothesis**  
   The model may encode correct visual information early but lose or override it during decoding.

## Possible experimental setup

### Step 1: Build a hallucination-labeled dataset

Use existing benchmarks to get examples labeled as hallucinated / non-hallucinated.

Possible sources:

- POPE for object existence hallucination
- AMBER for existence / attribute / relation hallucination
- HallusionBench for reasoning and illusion-type hallucination
- MMHal-Bench for open-ended multimodal hallucination

For each example, collect:

- Image
- Prompt
- Model response
- Hallucination label
- Hallucination type
- Hallucinated span if available
- Correct visual evidence if available

### Step 2: Run target MLLMs

Candidate models:

- LLaVA-style models
- Qwen-VL / Qwen2.5-VL / Qwen3-VL style models
- InstructBLIP / MiniGPT-style models
- Open-source models with accessible activations and attention maps

For each generation, save:

- Token logits
- Hidden states per layer
- Attention maps
- Cross-modal attention or image-token attention
- Image patch embeddings
- Output token probabilities
- Optional gradients if the model allows backward pass

### Step 3: Define internal statistics

Candidate features:

#### Attention-based

- Ratio of attention mass on image tokens vs. text tokens
- Attention entropy over image patches
- Attention concentration on object-relevant patches
- Attention sink ratio
- Layer-wise visual attention decay
- Head-level visual sensitivity

#### Attribution-based

- Integrated gradients from image tokens to generated token logit
- Gradient × activation for image tokens
- Logit lens contribution from image-conditioned residual stream
- Difference between clean image and corrupted/blank image attribution
- Token-level causal mediation score

#### Circuit-based

- Edge attribution from visual tokens to answer tokens
- Head-level causal importance
- Image-to-text circuit strength
- Prompt-to-answer circuit strength
- Ratio: visual circuit contribution / language-prior circuit contribution
- Circuit difference between hallucinated and grounded examples

#### Representation-based

- Linear probe accuracy for hallucination label
- CKA / CCA difference between hallucinated and grounded states
- Distance to object concept direction
- Hidden-state separability of correct vs. hallucinated object tokens
- Layer where hallucination becomes linearly decodable

#### Decoding-based

- Logit gap between visual-conditioned and image-ablated forward pass
- Sensitivity to image corruption
- Sensitivity to prompt perturbation
- Token probability under image-present vs. image-removed settings

### Step 4: Test whether statistics separate hallucination

Train simple classifiers or use correlation analysis:

- Logistic regression
- Linear probe
- Random forest / XGBoost
- AUROC / AUPRC for hallucination detection
- Calibration analysis
- Cross-model generalization
- Cross-dataset generalization
- Cross-hallucination-type generalization

The key is not just high detection performance, but interpretability:

> Which internal statistic best separates hallucinated from grounded generation, and at which layer/head/circuit does this happen?

## Possible contribution

A strong paper could contribute:

1. **A taxonomy of internal hallucination mechanisms**
   - Perception failure
   - Grounding failure
   - Language-prior override
   - Reasoning failure
   - Prompt-induced hallucination

2. **A set of mechanistic hallucination statistics**
   - Attention grounding score
   - Vision attribution ratio
   - Circuit grounding ratio
   - Representation separability score
   - Visual sensitivity score

3. **Empirical finding**
   - Some hallucination types are separable by attention patterns.
   - Some are better separated by causal attribution.
   - Reasoning hallucination may require representation/circuit analysis rather than raw attention.

4. **Mitigation direction**
   - Use the discovered statistic as a training reward, decoding constraint, or intervention signal.

## Possible paper structure

### 1. Introduction

- MLLMs hallucinate even when visual evidence is available.
- Most work evaluates or mitigates hallucination.
- Less is known about what changes internally during hallucination.
- We ask whether internal statistics can separate hallucinated and grounded generations.

### 2. Related Work

- MLLM hallucination benchmarks
- Hallucination detection and mitigation
- Contrastive decoding and language-prior mitigation
- Mechanistic interpretability for VLMs
- Attribution and causal mediation methods

### 3. Problem Setup

Define a generation as:

```text
y = M(image, prompt)
```

For each generated token or span, assign a hallucination label:

```text
h_t = 1 if token/span is unsupported by image
h_t = 0 otherwise
```

Goal:

```text
find internal statistic s_t such that s_t separates h_t = 1 from h_t = 0
```

### 4. Method

- Collect hallucination-labeled examples.
- Run model and extract internal activations.
- Compute attention, attribution, circuit, and representation statistics.
- Analyze separability.

### 5. Experiments

Evaluation dimensions:

- Object hallucination
- Attribute hallucination
- Relation hallucination
- Reasoning hallucination
- Prompt-induced hallucination

Evaluation metrics:

- AUROC for hallucination separation
- Correlation with hallucination severity
- Cross-dataset transfer
- Cross-model transfer
- Layer/head localization

### 6. Analysis

Key questions:

- Which layers first encode hallucination risk?
- Are hallucinated tokens less visually grounded?
- Do some attention heads systematically amplify prompt bias?
- Can causal ablation of identified heads reduce hallucination?
- Does representation separability appear before the hallucinated token is generated?

### 7. Mitigation

Use discovered signals to intervene:

- Penalize low visual-attribution tokens during decoding.
- Rerank generations by visual grounding score.
- Ablate or downweight hallucination-associated heads.
- Train with reward favoring visual dependence.
- Add a contrastive loss between image-present and image-absent prompts.

## What is novel here?

The novelty is not simply “detect hallucination.” The stronger framing is:

> Build a mechanistic map of hallucination types and identify which internal statistics explain each type.

A good research gap:

- Existing benchmarks say **whether** hallucination happened.
- Existing decoding methods reduce hallucination but may not explain **why**.
- Your project asks **where and how hallucination emerges inside the model**.

## Concrete first experiment

Start with a narrow, publishable version:

### Experiment: Object hallucination in LLaVA-style models

1. Use POPE or AMBER existence hallucination examples.
2. Generate answers with an open-source LVLM.
3. Extract layer-wise attention mass from generated object tokens to image tokens.
4. Compute image ablation sensitivity:

```text
visual_sensitivity = logit_original(object_token) - logit_image_ablated(object_token)
```

5. Compute gradient attribution from image tokens to object-token logit.
6. Train a linear probe to classify hallucinated vs. grounded object mentions.
7. Analyze which layers/heads best separate the two classes.
8. Ablate high-risk heads or apply visual-sensitivity decoding to test mitigation.

This gives a clean first paper direction.

## Risks and limitations

- Attention is not always a faithful explanation.
- Hallucination labels can be noisy.
- Open-ended generation makes span-level labeling difficult.
- Some hallucinations require world knowledge, not only image grounding.
- Attribution methods may be expensive for large MLLMs.
- Cross-model generalization may be weak.

## Possible title options

- **Mechanistic Signatures of Hallucination in Multimodal Large Language Models**
- **When Vision-Language Models Hallucinate: Internal Statistics for Grounding Failure**
- **Do Multimodal Models See Before They Hallucinate? A Mechanistic Study**
- **Separating Grounded and Hallucinated Generations via Vision-Aware Internal Signals**
- **From Attention to Circuits: Understanding Hallucination in Multimodal Models**

## Key references to start from

- Bai et al., *Hallucination of Multimodal Large Language Models: A Survey*, 2024.  
  https://arxiv.org/abs/2404.18930

- Wang et al., *AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation*, 2023.  
  https://arxiv.org/abs/2311.07397

- Prabhakaran et al., *VADE: Visual Attention Guided Hallucination Detection and Mitigation in VLMs*, ACL Findings 2025.  
  https://aclanthology.org/2025.findings-acl.773.pdf

- Gong et al., *DAMRO: Dive into the Attention Mechanism of LVLM to Reduce Object Hallucination*, EMNLP 2024.  
  https://aclanthology.org/2024.emnlp-main.439.pdf

- Rudman et al., *Mechanisms of Prompt-Induced Hallucination in Vision-Language Models*, 2026.  
  https://arxiv.org/abs/2601.05201

- Yang et al., *Understanding and Mitigating Hallucination in Large Vision-Language Models*, OpenReview.  
  https://openreview.net/forum?id=Bjq4W7P2Us

- Leng et al. / related works on visual contrastive decoding for LVLM hallucination mitigation.

## One-sentence pitch

This project studies whether hallucinated and grounded multimodal generations can be separated by internal visual-grounding statistics, such as attention, attribution, circuit contribution, and representation geometry, thereby explaining when MLLMs hallucinate and providing signals for mitigation.
