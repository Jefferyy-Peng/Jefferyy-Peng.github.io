---
layout: post
title: Hierarchical Reward for Long-Horizon Planning and Agent RL
date: 2026-05-02 
description: Idea
tags: RL
categories: Idea
---

**TL;DR:** 

# Research Idea 2: Hierarchical Reward for Long-Horizon Planning and Agent RL

## Working title

**Hierarchical Rewards for Long-Horizon Language Agents: Learning to Plan, Execute, Revise, and Credit Subgoals**

## Core research question

Long-horizon agents often fail because they must solve tasks that require many dependent decisions. A single final success/failure reward is too sparse. The idea is to design a **hierarchical reward structure** where:

- A long-term goal is decomposed into high-level plans.
- Each high-level plan is decomposed into subgoals.
- Each subgoal is decomposed into executable steps.
- Rewards are assigned not only for final task success, but also for correct intermediate planning, subgoal completion, and plan revision.
- Higher-level goal completion can validate or invalidate lower-level plans.

The central question is:

> Can hierarchical reward design improve long-horizon agent learning by giving credit to correct plans, correct subgoals, correct execution, and correct revision?

## Motivation

In long-horizon tasks, final rewards are sparse and delayed. For example, in a travel-planning agent:

```text
Goal: Plan a 5-day Cancun trip under budget and time constraints.

High-level plan:
1. Choose hotel route.
2. Schedule outdoor tours.
3. Arrange transport.
4. Check constraints.

Subgoals:
1. Find hotels.
2. Select tours.
3. Balance rest and activity.
4. Verify cost and timing.

Low-level actions:
1. Search hotel.
2. Search ferry.
3. Compare tour hours.
4. Update itinerary.
```

If the final itinerary fails because one ferry time is impossible, standard RL may give a bad reward to the whole trajectory. But many parts of the plan may still be correct. A hierarchical reward can assign partial credit and identify which level failed.

## Relationship to existing work

### Hierarchical RL

Traditional hierarchical reinforcement learning already studies temporal abstraction:

- Options framework
- MAXQ decomposition
- Feudal RL
- Subgoal discovery
- Skill learning

These methods decompose long-horizon tasks into higher-level policies and lower-level actions. However, classical HRL usually assumes environment states/actions, not natural-language plans.

### LLM agent planning

Recent LLM-agent work uses explicit planning and decomposition:

- Plan-and-Act style frameworks separate a planner from an executor.
- Web-agent methods use high-level plans to guide low-level browser actions.
- TravelPlanner, WebArena, WebVoyager, DeepPlanning, and related benchmarks test long-horizon constrained planning.
- ReAct / Reflexion / Tree-of-Thought / plan-execute-replan methods use reasoning traces and feedback loops.

But many such systems are prompt- or inference-time frameworks, not fully trained with hierarchical reward signals.

### Planning benchmarks

Relevant benchmarks include:

- **PlanBench**: formal planning and reasoning about actions/change.
- **TravelPlanner**: real-world travel planning with constraints and tools.
- **DeepPlanning**: long-horizon planning with travel and shopping tasks.
- **WebArena / WebVoyager**: web-navigation tasks requiring many steps.
- **Robotouille**: asynchronous planning benchmark.
- **Flex-TravelPlanner**: flexible planning with changing constraints.
- **REALM-Bench**: real-world planning scenarios and adaptation to disruptions.
- **HeroBench**: long-horizon hierarchical planning in an RPG-like world.

These benchmarks are relevant because they can expose failures in decomposition, execution, constraint satisfaction, and replanning.

## Existing design patterns close to your idea

### 1. Plan-and-execute agents

A model first generates a high-level plan, then another module executes it. The plan provides structure, but the system may still lack explicit reward for whether the plan hierarchy itself is correct.

### 2. Replan / revise loops

Agents can revise the plan after observing failure. This is close to your idea, but many works treat revision as an inference-time heuristic rather than a trainable ability with explicit revision rewards.

### 3. Verifier-guided planning

A verifier checks whether a plan or step satisfies constraints. This can provide intermediate reward:

```text
reward = final_success + constraint_satisfaction + verifier_score
```

### 4. Process reward models

Instead of only rewarding final answer correctness, a process reward model scores intermediate reasoning steps. This idea can be adapted from math/code reasoning to long-horizon agent planning.

### 5. Hierarchical goal/subgoal rewards

A reward can be assigned at multiple levels:

```text
R_total = R_final + R_plan + R_subgoal + R_step + R_revision
```

This is the direct version of your proposal.

## Proposed method

Represent an agent trajectory as a hierarchy:

```text
Goal G
 ├── Plan P1
 │    ├── Subgoal S1
 │    │    ├── Action a1
 │    │    └── Action a2
 │    └── Subgoal S2
 │         ├── Action a3
 │         └── Action a4
 └── Plan revision P2
      └── ...
```

Each level receives a reward.

## Reward design

### 1. Final task reward

Measures whether the whole task succeeds.

```text
R_final = 1 if final task is completed else 0
```

Examples:

- Correct travel itinerary satisfying all constraints
- Web task completed
- Shopping task completed
- Game/crafting goal achieved
- Robot task completed

### 2. High-level plan reward

Measures whether the proposed plan is valid and complete before execution.

Possible criteria:

- Covers all required constraints
- Has feasible ordering
- Does not contain contradictions
- Decomposes the goal into meaningful subgoals
- Matches environment affordances

Example:

```text
R_plan = verifier(plan, goal, constraints)
```

### 3. Subgoal reward

Measures whether each subgoal is achieved.

Example:

```text
R_subgoal_i = 1 if subgoal_i is completed else 0
```

For travel planning:

- Hotel chosen for every night
- Transportation selected
- Tour schedule feasible
- Budget satisfied
- Opening hours respected

### 4. Step/action reward

Measures whether each action contributes to the current subgoal.

Example:

```text
R_action_t = usefulness(action_t, subgoal_i)
```

This can be learned by a reward model or computed by an environment verifier.

### 5. Revision reward

Measures whether the agent correctly detects and fixes a plan failure.

Possible criteria:

- Detects infeasible step
- Identifies the correct failed constraint
- Proposes minimal repair
- Preserves still-correct parts of the plan
- Improves final feasibility

Example:

```text
R_revision = score(new_plan) - score(old_plan)
```

This is especially important for your idea.

### 6. Cross-level consistency reward

Higher-level goals should validate lower-level subgoals.

Example:

```text
R_consistency = 1 if all subgoals jointly satisfy the high-level goal else 0
```

This prevents locally correct but globally inconsistent plans.

## A possible total reward

```text
R = λ_final R_final
  + λ_plan R_plan
  + λ_subgoal Σ_i R_subgoal_i
  + λ_action Σ_t R_action_t
  + λ_revision R_revision
  + λ_consistency R_consistency
```

The key research issue is choosing weights and avoiding reward hacking.

## Training setup

Possible training pipeline:

### Stage 1: Supervised plan decomposition

Train the model to produce hierarchical plans from successful trajectories.

Data format:

```json
{
  "goal": "...",
  "high_level_plan": [...],
  "subgoals": [...],
  "actions": [...],
  "revision": [...]
}
```

### Stage 2: Verifier / reward model training

Train verifiers for:

- Plan feasibility
- Constraint satisfaction
- Subgoal completion
- Action usefulness
- Revision correctness

These can be trained from:

- Environment outcomes
- Synthetic traces
- Human annotations
- LLM-as-judge labels
- Constraint solvers

### Stage 3: RL fine-tuning

Use PPO / GRPO / DPO-like preference optimization / actor-critic methods with hierarchical rewards.

The model is rewarded for:

- Producing a valid plan
- Executing subgoals
- Revising when needed
- Achieving final success

## Datasets for plan revision or flexible planning

There are some relevant datasets/benchmarks, but the exact ability you describe is not fully solved.

### Useful existing benchmarks

#### TravelPlanner

Good for constrained real-world planning. It includes tool use, user constraints, commonsense constraints, and hard constraints. It can be adapted to revision by changing constraints mid-task or hiding information until later.

#### Flex-TravelPlanner

Closer to plan revision because it evaluates flexible planning under changing constraints.

#### DeepPlanning

Useful for long-horizon global/local planning, especially travel and shopping. It can test whether an agent performs global optimization and local constraint reasoning.

#### WebArena / WebVoyager

Useful for long-horizon web execution. Revision can be tested when a page state changes or an earlier action fails.

#### PlanBench

Useful for formal plan validity and action precondition/effect reasoning. Plan repair can be created by perturbing valid plans.

#### Robotouille

Useful for asynchronous planning, where actions may happen in parallel or with delays.

#### REALM-Bench

Useful for real-world planning scenarios, multi-agent planning, and adaptation to disruptions.

#### HeroBench

Useful for explicit hierarchical planning, multi-level dependencies, and very long action sequences.

### Gap

Most benchmarks test whether the final plan succeeds, but fewer provide dense labels for:

- Whether each subgoal is correct
- Whether each decomposition is valid
- Whether each revision is minimal and correct
- Whether the agent knows which level of the hierarchy failed

This is a strong research opportunity.

## Possible benchmark contribution

Create a benchmark called something like:

**HierPlan-Revise**

Each task includes:

1. A goal
2. A hierarchical gold plan
3. Subgoal dependencies
4. Hidden or changing constraints
5. Execution feedback
6. Required plan revision
7. Final success criteria
8. Labels for which subgoal failed

Example task:

```text
Goal: Plan a 4-day trip under $1200.

Initial plan:
- Day 1: Museum
- Day 2: Island tour
- Day 3: Ruins
- Day 4: Rest

New feedback:
- Ferry unavailable on Day 2.
- Ruins closed on Day 3.

Required behavior:
- Detect affected subgoals.
- Revise only the necessary days.
- Preserve valid hotel and budget choices.
- Produce final feasible plan.
```

Evaluation:

- Plan validity
- Subgoal completion
- Revision minimality
- Constraint satisfaction
- Final success
- Token/action efficiency
- Error localization accuracy

## Possible experiments

### Experiment 1: Does hierarchical reward improve final success?

Compare:

- Sparse final reward only
- Final reward + subgoal reward
- Final reward + plan reward
- Final reward + subgoal + revision reward
- Full hierarchical reward

Measure:

- Final success rate
- Constraint satisfaction
- Number of invalid actions
- Replanning success
- Efficiency

### Experiment 2: Does revision reward improve recovery from failure?

Create tasks where the initial plan becomes invalid.

Measure:

- Whether the agent detects the failure
- Whether it revises the correct subgoal
- Whether it avoids changing correct parts
- Whether final success improves

### Experiment 3: Does cross-level reward prevent local reward hacking?

Test whether agents optimize subgoals independently but violate the global goal.

Example:

- Each day of travel plan looks good.
- But total budget exceeds limit.
- Or hotel location makes all tours impossible.

Cross-level consistency reward should reduce this.

### Experiment 4: Generalization to longer horizons

Train on shorter tasks and test on longer tasks.

Measure:

- Decomposition depth generalization
- Subgoal dependency handling
- Failure recovery
- Reward hacking

## Novelty angle

The novelty is not just “use hierarchical planning.” The stronger idea is:

> Use explicit hierarchical reward credit assignment to train agents to decompose, execute, verify, and revise long-horizon plans.

Compared with existing work:

- Plan-and-act gives structure but not necessarily learned hierarchical reward.
- Process reward models score reasoning steps but are usually not tied to executable subgoal hierarchies.
- HRL gives temporal abstraction but not natural-language plan revision.
- Planning benchmarks test final plans but often lack dense hierarchical supervision.

## Possible paper structure

### 1. Introduction

- Long-horizon agents suffer from sparse rewards and compounding errors.
- Planning helps, but plans can be wrong or require revision.
- Current systems often lack explicit credit assignment across plan levels.
- We propose hierarchical rewards for plan, subgoal, action, and revision quality.

### 2. Related Work

- Hierarchical RL
- LLM planning and agent benchmarks
- Process reward models
- Plan verification and plan repair
- Long-horizon web/travel/embodied agents

### 3. Problem Formulation

Define a task as:

```text
T = (G, C, E)
```

where:

- `G` is the goal
- `C` is a set of constraints
- `E` is the environment

The agent produces a hierarchy:

```text
H = (P, S, A, R)
```

where:

- `P` is the high-level plan
- `S` are subgoals
- `A` are actions
- `R` are revisions

### 4. Hierarchical Reward

Define reward at each level:

```text
R_total = R_final + R_plan + R_subgoal + R_action + R_revision + R_consistency
```

### 5. Benchmark / Data Construction

Use existing benchmarks or create perturbations:

- TravelPlanner with changing constraints
- PlanBench with invalidated action preconditions
- WebArena tasks with execution failures
- Synthetic hierarchical tasks with ground-truth subgoal trees

### 6. Training

- SFT on successful hierarchical traces
- Train verifiers / reward models
- RL with hierarchical reward
- Optional GRPO/PPO optimization

### 7. Experiments

- Compare reward designs
- Analyze plan revision
- Analyze long-horizon scaling
- Analyze reward hacking
- Evaluate cross-domain transfer

### 8. Analysis

Key questions:

- Which reward level contributes most?
- Does subgoal reward help or hurt?
- Does revision reward improve recovery?
- Does plan reward improve global coherence?
- Does hierarchical reward generalize to longer tasks?

## Risks and limitations

- Reward hacking: agent may optimize subgoals without solving global task.
- Verifier quality: bad verifier gives misleading rewards.
- Annotation cost: hierarchical gold plans are expensive.
- Credit assignment: hard to know which subgoal caused final failure.
- Over-planning: too much planning may reduce efficiency.
- Domain specificity: travel/web planning rewards may not transfer to embodied tasks.

## Concrete first project

A practical first paper:

### Hierarchical reward for TravelPlanner-style revision

1. Start from TravelPlanner or a similar constrained planning benchmark.
2. Create perturbed tasks where one or more constraints change after initial planning.
3. Ask the agent to:
   - generate a plan,
   - receive feedback,
   - localize the failed subgoal,
   - revise the plan,
   - produce final answer.
4. Define rewards:
   - final pass rate,
   - constraint satisfaction,
   - subgoal correctness,
   - revision minimality,
   - failure localization.
5. Train or tune an agent with hierarchical reward.
6. Compare against:
   - final reward only,
   - ReAct,
   - Reflexion,
   - Plan-and-Act,
   - verifier-only reranking.

This is a clean and publishable version of your idea.

## Key references to start from

- Sutton, Precup, and Singh, *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*, 1999.  
  https://www.sciencedirect.com/science/article/pii/S0004370299000521

- Dietterich, *Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition*, 2000.  
  https://www.jair.org/index.php/jair/article/view/10266

- Valmeekam et al., *PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change*, NeurIPS 2023.  
  https://arxiv.org/abs/2206.10498

- Xie et al., *TravelPlanner: A Benchmark for Real-World Planning with Language Agents*, ICML 2024.  
  https://arxiv.org/abs/2402.01622

- Erdogan et al., *Improving Planning of Agents for Long-Horizon Tasks / Plan-and-Act*, ICML 2025.  
  https://arxiv.org/abs/2503.09572

- DeepPlanning benchmark, 2026.  
  https://qwenlm.github.io/Qwen-Agent/en/benchmarks/deepplanning/

- Flex-TravelPlanner, 2025.  
  https://openreview.net/forum?id=a7unQ5jMx7

- HeroBench, 2026.  
  https://arxiv.org/html/2508.12782v2

## One-sentence pitch

This project proposes hierarchical reward credit assignment for long-horizon language agents, rewarding not only final task success but also plan validity, subgoal completion, action usefulness, failure localization, and plan revision.


---

# Addendum: Counterfactual World-Model Evaluation for Long-Horizon Agents

## New idea

A natural extension of the hierarchical reward idea is to ask whether an agent has a **counterfactual world model**.

The key question is:

> Given the current world state or trajectory, can the model reason about what would happen if one condition, action, observation, or plan step were different?

This is closely related to world modeling, but the focus is more specific:

- Not just “can the agent complete the task?”
- Not just “can the agent predict the next state?”
- Not just “can the agent revise after failure?”
- But: **can the agent mentally simulate alternative trajectories and compare their downstream consequences?**

In other words, the model should answer questions like:

```text
Given the current trajectory τ:
- What would happen if action a_t were replaced by a'_t?
- What if observation o_t had been different?
- What if subgoal S_i were skipped?
- What if the high-level plan order were changed?
- What if a constraint appeared earlier?
- Which future failure would this alternative plan cause?
- Which plan revision would minimally fix the trajectory?
```

This ability can be called:

**Counterfactual Trajectory Reasoning**  
or  
**Counterfactual World-Modeling for Agents**

## Why this sounds like a world model

A world model usually predicts environment dynamics:

```text
current state + action -> next state / future trajectory
```

A counterfactual world model asks a stronger question:

```text
current trajectory + intervention -> alternative future trajectory
```

So the agent is not only predicting the next factual state. It is simulating a hypothetical branch of the world.

For long-horizon planning, this is essential because good agents must reason about delayed consequences:

```text
If I choose this hotel today, then tomorrow's tour becomes impossible.
If I skip this information-gathering step, I may not know whether the final answer is valid.
If I change this subgoal, the later transportation plan must also change.
```

This connects naturally to hierarchical planning because counterfactuals can occur at multiple levels:

| Level | Counterfactual question |
|---|---|
| Goal level | What if the user changed the objective? |
| Constraint level | What if the budget/time/location constraint changed? |
| Plan level | What if we used a different high-level plan? |
| Subgoal level | What if this subgoal were skipped or reordered? |
| Action level | What if this tool/action were replaced? |
| Observation level | What if the agent observed a different result? |
| Revision level | What if the agent repaired the wrong part of the plan? |

## Deeper related work search

The idea is related to several existing research lines, but it is not fully covered by any single one.

### 1. Counterfactual reasoning benchmarks for LLMs

There are benchmarks such as **CounterBench**, which evaluate whether LLMs can perform counterfactual reasoning over causal graphs and structured questions. CounterBench includes different counterfactual types such as basic, joint, conditional, nested, and backdoor counterfactuals.

However, this is mostly about **static causal reasoning** rather than interactive agent trajectories.

Gap:

```text
CounterBench tests whether a model can answer counterfactual causal questions.
It does not directly test whether an agent can simulate how an alternative action or observation changes a long-horizon task trajectory.
```

### 2. Forward counterfactual generation

Some work studies **forward counterfactual reasoning**, where the model predicts what future developments would follow from an alternative condition. For example, FIN-FORCE studies forward counterfactual generation for financial news.

This is conceptually close because it asks “what would happen next if something were different?”

Gap:

```text
Forward counterfactual generation focuses on future scenario generation, often in a domain like finance.
It is not primarily an agent benchmark with actions, observations, tool use, subgoal dependencies, and executable plan revision.
```

### 3. Counterfactuals for language model agents

Recent work on **Abstract Counterfactuals for Language Model Agents** proposes counterfactual reasoning at the level of high-level action semantics instead of raw token-level interventions. This is very relevant because it argues that action counterfactuals should be represented at an abstract semantic level.

Gap:

```text
This work studies how to construct meaningful counterfactual actions for LM agents.
It does not provide a broad benchmark for hierarchical long-horizon plan counterfactuals or evaluate whether agents can predict downstream trajectory consequences across plan/subgoal/action levels.
```

### 4. Counterfactual trajectory training in navigation

Older embodied AI work such as **Counterfactual Vision-and-Language Navigation** uses counterfactual observations and trajectories to improve robustness in navigation. It explicitly asks questions like what would happen if a different object were observed.

Gap:

```text
This is close in spirit, but it is domain-specific to vision-and-language navigation and mainly used as a training strategy for generalization.
It is not a general agent benchmark for counterfactual world models across web, travel, tool-use, planning, or hierarchical task domains.
```

### 5. World-model benchmarks

Recent work on world-model evaluation argues that world models should be evaluated not only by visual fidelity or next-state prediction, but also by prediction, planning, and counterfactual reasoning. For example:

- **AutumnBench / WorldTest** evaluates world-model learning using interactive grid-world environments and includes prediction, planning, and counterfactual reasoning tasks.
- **World Reasoning Arena (WR-Arena)** evaluates world models on action simulation fidelity, long-horizon forecast, and simulative reasoning/planning.
- Recent surveys on agentic world modeling define world models as learning state-transition dynamics and helping agents anticipate consequences of candidate actions.

Gap:

```text
These works evaluate world-model capabilities, but often focus on world-model systems or simulated environments.
The proposed idea targets LLM agents specifically and asks whether their internal or external reasoning can support counterfactual trajectory evaluation over natural-language hierarchical plans.
```

### 6. Planning analysis for LLM agents

Recent planning-centric analyses argue that LLM agents often behave like step-wise greedy policies and fail because early actions do not account for delayed consequences. This directly supports the need for future-aware and counterfactual evaluation.

Gap:

```text
These works show long-horizon planning failure and propose lookahead/value-estimation methods.
But they do not necessarily isolate counterfactual world-model competence as a benchmarked capability:
given trajectory τ and intervention do(a_t = a'_t), can the model predict the alternative outcome?
```

### 7. Counterfactual trajectory pairs for agent training

CaRT uses counterfactual trajectory pairs to teach LLM agents when to stop gathering information. It creates paired trajectories where termination is appropriate vs. minimally modified trajectories where termination is not.

Gap:

```text
CaRT is highly relevant, but it focuses on the specific skill of termination / information-gathering.
A broader benchmark could evaluate counterfactual reasoning over arbitrary actions, observations, plan steps, subgoals, constraints, and revisions.
```

## Novelty assessment

The broad phrase **“counterfactual world model” is not completely new**. There is existing work on:

- Counterfactual reasoning in LLMs
- Counterfactual planning
- Counterfactual trajectories in navigation
- World-model benchmarks with counterfactual tasks
- LLM-as-world-model planning methods
- Counterfactual trajectory pairs for specific agent skills

So the paper should not claim:

```text
No one has studied counterfactual reasoning or world models before.
```

A safer and stronger novelty claim is:

> Existing agent benchmarks mostly evaluate final task success or factual trajectory execution. Existing counterfactual benchmarks often focus on static causal reasoning, domain-specific future generation, or simulated world-model evaluation. What remains underexplored is a benchmark and training framework for **counterfactual trajectory reasoning in long-horizon LLM agents**, especially across hierarchical plan levels: goal, constraint, plan, subgoal, action, observation, and revision.

This is a more defensible research gap.

## Proposed research direction

### Main task

Given a factual agent trajectory:

```text
τ = (s_0, a_0, o_1, a_1, o_2, ..., a_T, o_T)
```

and an intervention:

```text
do(x_i = x'_i)
```

where `x_i` may be an action, observation, constraint, subgoal, or plan step, ask the model to predict:

```text
τ' = alternative future trajectory
```

and/or answer:

```text
Will the final task still succeed?
Which future step changes?
Which constraint will fail?
Which subgoal becomes invalid?
What minimal revision restores success?
```

### Example: travel planning

Factual trajectory:

```text
Goal: Plan a 4-day Cancun trip.
Plan:
Day 1: Isla Mujeres
Day 2: Chichen Itza + cenote
Day 3: Xplor
Day 4: Cozumel
```

Counterfactual intervention:

```text
What if the ferry to Cozumel is unavailable on Day 4?
```

Expected model behavior:

```text
- Recognize that only the Cozumel subgoal is directly affected.
- Predict downstream effects on transport, hotel location, and timing.
- Preserve unaffected days if still valid.
- Propose a minimal repair, e.g., swap Cozumel with Isla Mujeres or replace Cozumel with a local activity.
```

### Example: web agent

Factual trajectory:

```text
Goal: Buy the cheapest compatible laptop charger.
The agent searches product page A, filters by price, and purchases item X.
```

Counterfactual intervention:

```text
What if the agent had clicked the compatibility tab before purchasing?
```

Expected behavior:

```text
- Predict that the agent would discover item X is incompatible.
- Predict that the original purchase should be avoided.
- Identify which earlier decision changes.
- Continue with an alternative search path.
```

### Example: tool-use agent

Factual trajectory:

```text
The agent answers a question after one web search.
```

Counterfactual intervention:

```text
What if the first search result were outdated?
```

Expected behavior:

```text
- Predict that the confidence of the answer should decrease.
- Continue searching or verify with a more recent source.
- Avoid finalizing too early.
```

## Benchmark design: Counterfactual Agent World Model Benchmark

A possible benchmark name:

**CAWM-Bench: Counterfactual Agent World-Model Benchmark**

or

**CoTrajBench: Counterfactual Trajectory Reasoning Benchmark**

or

**Hier-CFBench: Hierarchical Counterfactual Planning Benchmark**

### Data format

Each example contains:

```json
{
  "goal": "...",
  "constraints": [...],
  "factual_plan": [...],
  "factual_trajectory": [...],
  "intervention": {
    "type": "action | observation | constraint | subgoal | plan_step | revision",
    "target": "...",
    "replacement": "..."
  },
  "expected_counterfactual_effects": [...],
  "expected_final_outcome": "success | failure | changed_success",
  "minimal_revision": [...],
  "affected_subgoals": [...],
  "unaffected_subgoals": [...]
}
```

### Intervention types

1. **Action counterfactual**
   - What if the agent used a different tool/action?

2. **Observation counterfactual**
   - What if the tool returned different information?

3. **Constraint counterfactual**
   - What if a new constraint appeared?

4. **Subgoal counterfactual**
   - What if a subgoal were skipped, reordered, or replaced?

5. **Plan-level counterfactual**
   - What if the high-level strategy were different?

6. **Revision counterfactual**
   - What if the agent repaired the wrong part of the plan?

7. **Information-gathering counterfactual**
   - What if the agent stopped earlier or searched one more time?

## Evaluation metrics

### 1. Outcome prediction accuracy

Can the model correctly predict whether the counterfactual trajectory succeeds or fails?

```text
Acc(success/failure)
```

### 2. Affected-subgoal localization

Can it identify which subgoals are affected by the intervention?

```text
F1(affected_subgoals)
```

### 3. Unaffected-subgoal preservation

Can it avoid unnecessarily changing valid parts of the plan?

```text
Preservation score
```

### 4. Counterfactual consistency

Does the predicted alternative trajectory logically follow from the intervention?

```text
consistency(intervention, predicted_trajectory)
```

### 5. Minimal repair score

If the counterfactual causes failure, can the model propose the smallest valid revision?

```text
minimal_repair_score = validity - unnecessary_changes
```

### 6. Long-horizon dependency accuracy

Can the model track delayed effects several steps later?

```text
accuracy_by_distance_from_intervention
```

### 7. Causal contrast score

Can the model explicitly contrast factual and counterfactual outcomes?

```text
Δ = predicted_outcome(τ') - predicted_outcome(τ)
```

## Training with hierarchical counterfactual reward

This addendum also strengthens the hierarchical reward idea.

Instead of only rewarding factual task completion:

```text
R_final
```

we can add counterfactual rewards:

```text
R_counterfactual =
  R_outcome_prediction
+ R_affected_subgoal
+ R_minimal_revision
+ R_counterfactual_consistency
+ R_unaffected_preservation
```

Then the total hierarchical reward becomes:

```text
R_total =
  R_final
+ R_plan
+ R_subgoal
+ R_action
+ R_revision
+ R_consistency
+ R_counterfactual
```

This encourages the agent not only to execute the current plan, but also to understand why the plan works and what would break if the world changed.

## Why this is useful

A counterfactual world-model benchmark would test abilities that normal success-rate benchmarks miss:

| Normal agent benchmark | Counterfactual world-model benchmark |
|---|---|
| Did the agent complete the task? | Does the agent know why the task succeeded? |
| Did the agent recover after failure? | Could the agent predict the failure before executing? |
| Did the final answer satisfy constraints? | Does the agent know which constraint would fail under an alternative? |
| Did the agent use tools correctly? | Does the agent know when a different tool result would change the plan? |
| Did the plan work once? | Does the agent understand the space of nearby possible plans? |

## Strong paper framing

The paper can be framed as:

> Current agent benchmarks evaluate realized behavior, but not counterfactual competence. We propose evaluating whether LLM agents possess a counterfactual world model: the ability to predict how alternative actions, observations, constraints, or plan steps would change future trajectory outcomes.

This is highly aligned with your original hierarchical reward idea because:

- hierarchical plans define the structure of possible interventions;
- counterfactual reasoning tests whether the model understands dependencies between levels;
- counterfactual reward provides dense supervision for planning and revision;
- minimal repair tests whether the model can revise plans without destroying valid subgoals.

## Most promising concrete version

A first publishable version could focus on constrained planning:

### Dataset

Use or extend:

- TravelPlanner
- Flex-TravelPlanner
- WebArena-style tasks
- PlanBench-style symbolic plans
- Synthetic hierarchical planning tasks

### Perturbation generation

For each factual successful trajectory, generate counterfactual interventions:

```text
change one constraint
change one observation
replace one action
remove one subgoal
swap two plan steps
invalidate one tool result
```

### Labels

Automatically or semi-automatically label:

```text
success/failure under counterfactual
affected subgoals
required repair
minimal valid revised plan
```

### Baselines

Compare:

- Direct prompting
- Chain-of-thought
- ReAct
- Reflexion
- Plan-and-Act
- Tree-of-thought / MCTS-style planning
- Agent with explicit world model
- Agent trained with hierarchical counterfactual reward

### Main claim

```text
Agents with stronger counterfactual trajectory reasoning should perform better at long-horizon planning, plan revision, and robustness under distribution shift.
```

## Updated one-sentence pitch

This project proposes hierarchical reward learning for long-horizon agents, extended with **counterfactual world-model evaluation**: testing whether agents can predict how alternative actions, observations, constraints, or plan steps would change future trajectory outcomes and use that knowledge to revise plans minimally and correctly.

## References for this addendum

- Chen et al., **CounterBench: Evaluating and Improving Counterfactual Reasoning in Large Language Models**, AAAI 2026.  
  https://arxiv.org/abs/2502.11008

- Ong et al., **A Benchmark for Forward Counterfactual Generation**, EMNLP 2025.  
  https://aclanthology.org/2025.emnlp-main.575/

- Warrier et al., **Benchmarking World-Model Learning / AutumnBench**, 2025.  
  https://arxiv.org/abs/2510.19788

- **World Reasoning Arena: A Benchmark for Next-Generation World Models**, 2026.  
  https://arxiv.org/abs/2603.25887

- Anonymous / arXiv, **Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond**, 2026.  
  https://arxiv.org/abs/2604.22748

- Parvaneh et al., **Counterfactual Vision-and-Language Navigation**, NeurIPS 2020.  
  https://proceedings.neurips.cc/paper/2020/hash/39016cfe079db1bfb359ca72fcba3fd8-Abstract.html

- Qiao et al., **Agent Planning with World Knowledge Model**, NeurIPS 2024.  
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/d032263772946dd5026e7f3cd22bce5b-Abstract-Conference.html

- **Abstract Counterfactuals for Language Model Agents**, 2025.  
  https://arxiv.org/abs/2506.02946

- **CaRT: Teaching LLM Agents to Know When They Know Enough**, 2025.  
  https://arxiv.org/abs/2510.08517

- **Why Reasoning Fails to Plan: A Planning-Centric Analysis of Long-Horizon Decision Making in LLM Agents**, 2026.  
  https://arxiv.org/abs/2601.22311

