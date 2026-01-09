---
layout: page
title: Data-Driven MRI Postprocessing
description: A data-driven, deep-learning based MRI postprocessing framework to improve medical image analysis
img: assets/img/ismrm-project.jpg
importance: 1
category: work
related_publications: true
---

Synthetic MRI (SyMRI) enables the generation of quantitative tissue relaxation maps—T1, T2, and proton density (PD)—which can be used to synthesize a wide variety of contrast-weighted images. However, both conventional structural MRI (e.g., T1-weighted and T2-weighted scans) and current SyMRI post-processing pipelines typically rely on manually selected acquisition parameters such as repetition time (TR), echo time (TE), and inversion time (TI). These parameters are often chosen through expert-driven heuristics or visual inspection, resulting in inconsistent contrast quality across scanners and limited scalability for large cohort studies.

To overcome these limitations, we propose a physics-informed, learning-based framework that automatically adjusts SyMRI contrast parameters in tandem with downstream tasks such as tissue segmentation. Inspired by the concept of joint optimization of data transformations and task objectives, our approach embeds a differentiable MRI physics layer into the training pipeline. This enables end-to-end optimization of both contrast generation and segmentation modules, allowing the model to discover contrast settings that maximize task performance—while remaining faithful to underlying physical constraints.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ismrm-project.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    SynMRI-based learnable contrast generation and segmentation schematic workflow. Quantitative maps (T1, T2, PD) are
processed by the LCM with differentiable physical parameters (TI, etc.). The generated contrasts are concatenated and fed into the DTM. For this work, we used a 2D U-Net for subcortical segmentation. The Dice loss is computed and backpropagated through both modules, enabling joint optimization of contrast parameters and segmentation weight.
</div>

