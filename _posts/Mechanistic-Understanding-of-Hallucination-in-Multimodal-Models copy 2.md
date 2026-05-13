---
layout: post
title: Counterfactual Reasoning of Agents
date: 2026-05-02 
description: Idea
tags: Agent
categories: Idea
---

**TL;DR:** 

# Counterfactual Reasoning of Agents

## Core research question

Current agent benchmarks tell us whether an agent succeeded, but not whether the agent understands why it succeeded or failed. Does an agent have a counterfactual world model of its own trajectory?

That means: after seeing an agent’s plan, observations, actions, tool outputs, and final result, can the model reason:

If this action/observation/tool result/plan step had been different,
what would change later?
Would the final task still succeed?
Which subgoal would break?
What is the smallest repair?

## Motivation

### 1. Final success is a weak signal

Most agent benchmarks evaluate:

Did the agent complete the task? yes/no

But this hides many important differences.

Two agents may both succeed, but one may understand the task better:

Agent A:
Finds the right answer by luck.

Agent B:
Finds the right answer because it verified sources, tracked constraints, and understood dependencies.

Similarly, two agents may both fail, but for different reasons:

Agent A:
Failed because one tool result was wrong.

Agent B:
Failed because it did not understand which subgoal depended on that tool result.

A success/failure benchmark cannot distinguish these cases.

### 2. Long-horizon agents need counterfactual reasoning

For short tasks, reactive behavior may be enough.

For long-horizon tasks, agents must reason:

If I choose this action now, what happens later?
If this observation is wrong, which downstream decision changes?
If this tool result changes, should I continue searching or start synthesis?
If this plan step is removed, what risk increases?
If this subgoal fails, can the parent goal still succeed?

This is exactly counterfactual reasoning over trajectories.

Without this ability, an agent may look competent in one trajectory but fail badly when anything changes.

### 3. Real-world agents operate under partial information

In real tasks, the agent often does not know the full environment state.

Examples:

A hotel may be unavailable.
A paper search result may be incomplete.
A website field may be required but hidden.
A product may be incompatible.
A source may be outdated.
A form may fail after submission.

Good agents must reason over hidden or uncertain system states.

Your benchmark tests whether the agent can say:

This observation changes my belief.
This subgoal is now invalid.
This previous plan step must be revised.
This final answer is no longer reliable.

That is closer to real deployment than only checking final task success.

### 5. This helps diagnose agent failures

When an agent fails, we want to know:

Was the failure caused by a wrong action?
A misleading observation?
A missing verification step?
A bad tool result?
A failed subgoal?
A wrong high-level plan?

Your benchmark forces the model to identify:

first divergent step
affected subgoals
failure cause
final outcome change
minimal repair

So it becomes a diagnostic tool, not just a leaderboard.

This is valuable for both research and deployment.

### 6. This connects naturally to plan repair

A strong agent should not only know that a trajectory fails.

It should know how to minimally fix it.

Example:

The agent failed because it submitted the form before filling the required field.

A weak model says:

Try again.

A stronger model says:

Insert a required-field check before submission, fill the missing field, then submit again.

This is exactly what real agents need: localized repair, not full restart.

Your benchmark can measure this with:

minimal repair validity
minimal repair edit distance
unaffected-step preservation

That makes the benchmark practically useful.

### 7. This can support better training signals

Your hierarchical reward idea becomes stronger with counterfactual trajectory data.

Instead of only rewarding final success:

R_final

you can reward:

R_outcome_prediction
R_affected_subgoal
R_failure_localization
R_minimal_repair
R_unaffected_preservation

So the dataset can train agents to understand:

why a plan works
what would break
how to repair it
which parts should remain unchanged

This is much richer than sparse final reward.

### 8. Core motivation in one paragraph

Current long-horizon agent benchmarks mainly evaluate whether an agent can complete a task, but final success does not reveal whether the agent understands the causal dependencies inside its own trajectory. Real agents must operate under changing observations, imperfect tool outputs, hidden environment states, and failed subgoals. Therefore, we need a benchmark that tests whether models can reason counterfactually over agent trajectories: if one action, observation, tool result, plan step, or subgoal were changed, what downstream states, decisions, and outcomes would change? This capability is essential for robust planning, failure diagnosis, and minimal plan repair.

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

### 1. Overall data format

For each benchmark trajectory, convert it into examples like this:

```json
{
  "benchmark": "ScienceWorld",
  "task_id": "...",
  "goal": "Determine whether unknown substance X conducts electricity.",
  "factual_trajectory": [
    {"t": 0, "obs": "...", "action": "go to kitchen"},
    {"t": 1, "obs": "...", "action": "pick up wire"},
    {"t": 2, "obs": "...", "action": "connect wire to battery"},
    {"t": 3, "obs": "...", "action": "test substance X"}
  ],
  "observable_state_t": "...",
  "hidden_state_t": {
    "substance_X_property": "conductive",
    "battery_charge": "low",
    "wire_connected": true
  },
  "intervention": {
    "type": "hidden_state_change",
    "target": "battery_charge",
    "from": "low",
    "to": "dead"
  },
  "counterfactual_question": "If the battery had been dead at step 3, would the agent still correctly determine whether substance X conducts electricity?",
  "expected_reasoning": [
    "The test result would become unreliable because the circuit would not work.",
    "The observed non-conductivity could be caused by the dead battery rather than the material.",
    "The agent should first test or replace the battery."
  ],
  "counterfactual_outcome": "failure_or_uncertain",
  "affected_subgoals": ["test conductivity"],
  "minimal_repair": ["verify battery works", "replace battery", "repeat conductivity test"],
  "unaffected_subgoals": ["collect wire", "collect substance"]
}
```

This is different from ordinary planning QA because the answer requires reasoning over a latent state variable that affects future observations.

### 2. Choose benchmarks where hidden state is recoverable

You want benchmarks where the environment has an explicit state, database, simulator, or verifier. These are easier to convert into counterfactual data.

#### Best candidates
Benchmark	Why good for hidden-state counterfactuals
ScienceWorld	Text simulator with physical/scientific state variables; agents issue text actions and receive observations. It is explicitly designed for interactive grounded scientific reasoning.
ALFWorld	TextWorld environments aligned with ALFRED embodied tasks; good for object-location, containment, cleanliness, heat/cool, and receptacle hidden states.
TravelPlanner	Has structured constraints, tools, travel records, and reference plans; good for hidden availability, budget, opening hours, location, route, and feasibility states.
WebArena	Self-hostable realistic websites, external tools, knowledge bases, and long-horizon web tasks; good for hidden database/product/account/task-management states.
PlanBench	Symbolic action preconditions/effects; good for exact counterfactual plan validity, though less realistic.
BabyAI / MiniGrid	Fully controllable symbolic world; good for clean experiments but less realistic.

For a first paper, I would choose ScienceWorld + TravelPlanner. ScienceWorld gives simulator-grounded hidden states; TravelPlanner gives realistic natural-language planning and constraints.

### 3. Define “hidden state” formally

For each trajectory, distinguish:

observation o_t: what the agent sees
hidden state z_t: true environment/system variables not fully exposed
action a_t: what the agent does
goal g: task objective
constraints c: explicit user/system constraints

A normal environment transition is:

```text
z_{t+1}, o_{t+1} = T(z_t, a_t)
```

A counterfactual intervention changes one variable:

```text
do(z_t^k = v')
```

Then the task is to predict:

```text
z'_{t+1:T}, o'_{t+1:T}, success'
```

Your benchmark should test whether the model can infer the downstream effect of hidden state changes.

### 4. Types of counterfactual hidden-state interventions

You should not only ask “what if action changed?” Make the intervention target a hidden state variable that explains future trajectory changes.

#### Type A: Hidden precondition change

The agent does the same action, but a hidden precondition is different.

Example:

Factual: The door is unlocked, so opening it succeeds.
Counterfactual: The door was secretly locked.
Question: What future step fails, and what should the agent do?

Good for: ALFWorld, BabyAI, WebArena.

#### Type B: Hidden object/property change

An object has a different latent property.

Example:

Factual: The substance is conductive.
Counterfactual: The substance is non-conductive.
Question: How would the experiment observation and conclusion change?

Good for: ScienceWorld.

#### Type C: Hidden availability change

A resource is unavailable, but the agent only discovers this later.

Example:

Factual: Ferry to Cozumel is available on Day 4.
Counterfactual: Ferry is unavailable on Day 4.
Question: Which parts of the itinerary become invalid?

Good for: TravelPlanner.

#### Type D: Hidden database change

A website backend state changes.

Example:

Factual: Product X is in stock.
Counterfactual: Product X is out of stock.
Question: Which click/purchase action would fail, and what alternative path should be taken?

Good for: WebArena.

#### Type E: Hidden causal variable change

A variable affects future observations indirectly.

Example:

Factual: Plant receives sunlight, later grows.
Counterfactual: Plant was kept in darkness.
Question: Which later observation changes and what conclusion should be revised?

Good for: ScienceWorld.

#### Type F: Hidden agent-memory / information state change

The world is the same, but the agent lacks a key observation.

Example:

Factual: Agent checked the hotel cancellation policy.
Counterfactual: Agent skipped that check.
Question: What uncertainty remains? Should the final plan be accepted?

Good for: WebArena, TravelPlanner.

This is important because long-horizon agents often fail not only from bad actions, but from missing information.

### 5. Extraction pipeline
#### Step 1: Collect factual successful and failed trajectories

For each benchmark, run agents or use existing trajectories.

Store:

goal
initial observation
actions
observations
final outcome
environment state snapshots if available
tool/database outputs
subgoal annotations if available

For simulator benchmarks like ScienceWorld and ALFWorld, you can log state after every step because the environment is executable. ScienceWorld is an interactive text environment where text commands change the world and return textual observations, making state/action logging natural.

For WebArena, because it is self-hostable and built from functional websites with reproducible task setups, you can log DOM state, database state, page observations, and tool outputs during each trajectory.

For TravelPlanner, log structured travel plans, tool calls, returned records, and constraint satisfaction checks. TravelPlanner includes tools and large travel records, and the agent must formulate plans covering transportation, meals, attractions, and accommodation under constraints.

#### Step 2: Parse the trajectory into a hierarchy

Convert each trajectory into:

```text
Goal
 ├── High-level plan
 ├── Subgoals
 │    ├── information-gathering actions
 │    ├── environment-changing actions
 │    └── verification actions
 └── final answer / completion action
```

Example for TravelPlanner:

Goal: 5-day Cancun trip

```text
Subgoals:
S1: choose hotel
S2: schedule Chichen Itza + cenote
S3: schedule Cozumel
S4: schedule Isla Mujeres
S5: schedule Xplor
S6: verify budget and timing
```

Example for ScienceWorld:

Goal: Determine if unknown material conducts electricity

```text
Subgoals:
S1: collect circuit components
S2: build circuit
S3: test known conductive object
S4: test unknown object
S5: infer property
```

This hierarchy matters because your counterfactual questions should ask which subgoal is affected by the hidden-state change.

#### Step 3: Identify hidden state variables

For each benchmark, define a hidden-state schema.

#### ScienceWorld hidden state schema
```json
{
  "object_location": "...",
  "object_temperature": "...",
  "object_state": "solid/liquid/gas",
  "material_property": "conductive/nonconductive/magnetic/soluble",
  "container_contents": [...],
  "device_state": "on/off/broken/charged",
  "experiment_condition": "controlled/uncontrolled"
}
```
#### ALFWorld hidden state schema
```json
{
  "object_location": "drawer/table/fridge/sink",
  "container_open": true,
  "object_clean": false,
  "object_hot": false,
  "object_cool": true,
  "object_sliced": false,
  "agent_inventory": [...]
}
```
#### TravelPlanner hidden state schema
```json
{
  "flight_available": true,
  "hotel_available": true,
  "attraction_open": true,
  "transport_time": 90,
  "hidden_fee": 30,
  "weather_feasible": true,
  "distance_feasible": true,
  "reservation_required": true
}
```
#### WebArena hidden state schema
```json
{
  "product_stock": "in_stock/out_of_stock",
  "product_compatibility": true,
  "cart_contents": [...],
  "user_permission": "admin/member/guest",
  "issue_status": "open/closed",
  "form_validation_rule": "...",
  "backend_record": "..."
}
```

The hidden-state schema is the core of the dataset. It lets you generate controlled counterfactual interventions.

#### Step 4: Generate counterfactual interventions

For each trajectory, choose one time step t and one hidden variable z_t^k.

Then create an intervention:

```text
do(z_t^k = v')
```

You want interventions that are:

Plausible
The changed hidden state could realistically occur.
Causally relevant
It affects later observations, success, or required repair.
Nontrivial
The answer cannot be solved by keyword matching.
Local but downstream
The intervention happens at one point, but the effect appears later.
#### Intervention generation algorithm
```python
for trajectory in trajectories:
    state_trace = trajectory.hidden_states
    subgoals = parse_subgoals(trajectory)
```

```python
    for t in candidate_steps:
        for variable in mutable_hidden_vars(state_trace[t]):
            alt_values = plausible_alternatives(variable)
```

```python
            for v_alt in alt_values:
                cf_env = clone_env_at_step(t)
                cf_env.set_hidden_state(variable, v_alt)
```

                cf_rollout = replay_or_resimulate(cf_env, trajectory.actions[t:])

                if cf_rollout.outcome != trajectory.outcome:
                    save_counterfactual_example(...)

There are two ways to get the counterfactual outcome:

#### Option 1: Replay same future actions

You keep future actions fixed and ask:

If the world had changed but the agent followed the same trajectory, what would happen?

This tests outcome prediction.

#### Option 2: Allow optimal repair

You ask:

Given this hidden change, what should the agent revise?

This tests planning and revision.

You should include both.

### 6. How to obtain labels
#### Label type 1: Simulated counterfactual outcome

For executable environments, directly run the simulator after the intervention.

Works for:

ScienceWorld
ALFWorld
BabyAI
MiniGrid
some WebArena states if you control the backend

Label:

```json
{
  "counterfactual_success": false,
  "changed_observations": [...],
  "failed_action": "test substance",
  "failure_reason": "battery dead"
}
```
#### Label type 2: Constraint-verifier outcome

For planning benchmarks like TravelPlanner, use a verifier.

TravelPlanner already evaluates plans by constraint satisfaction, and its tasks involve multiple constraint types including hard constraints, commonsense constraints, and environmental constraints.

Example verifier:

```python
def verify_travel_plan(plan, hidden_state):
    checks = {
        "hotel_available": check_hotel_availability(plan, hidden_state),
        "budget": check_budget(plan),
        "route_feasible": check_transport(plan, hidden_state),
        "attraction_open": check_opening_hours(plan, hidden_state),
        "meal_feasible": check_meals(plan)
    }
    return checks
```

Label:

```json
{
  "counterfactual_success": false,
  "violated_constraints": ["ferry_unavailable", "arrival_too_late"],
  "affected_days": ["Day 4"],
  "affected_subgoals": ["visit Cozumel"],
  "minimal_repair": ["replace Cozumel with Isla Mujeres", "move Cozumel to Day 2 if ferry available"]
}
```
#### Label type 3: LLM-assisted but verifier-checked repair

For minimal repairs, exact simulation may not provide a repair. Use an LLM to propose candidate repairs, then use a verifier to check them.

Pipeline:

counterfactual failed plan
→ generate 5 repairs
→ run verifier
→ choose valid repair with smallest edit distance from factual plan

Minimality score:

```text
minimality = valid_repair - number_of_unnecessary_changes
```

This is useful for TravelPlanner and WebArena.

### 7. Counterfactual question templates

Your dataset should contain multiple question types, not just one.

Type 1: Outcome prediction
Given the trajectory below, suppose that hidden state H had been different at step t.
Would the task still succeed? Explain why.

Example:

Suppose the hotel selected on Day 3 was actually unavailable, although the agent had not checked availability yet. Would the final travel plan still satisfy the user request?
Type 2: Future observation prediction
What observation would the agent receive at step t+1 under this counterfactual?

Example:

If the battery were dead before the conductivity test, what would the agent observe when connecting the circuit?
Type 3: Affected-subgoal localization
Which subgoals are affected by this hidden-state change?

Example:

If the Cozumel ferry is unavailable on Day 4, which parts of the itinerary become invalid?
Type 4: Minimal repair
What is the smallest change to the plan that restores success?

Example:

The ferry is unavailable on Day 4. Modify the itinerary minimally while preserving all valid days.
Type 5: Contrast factual vs. counterfactual
Compare the factual and counterfactual trajectories. What changes and what remains the same?

Example:

In the factual trajectory, the agent successfully buys product X. In the counterfactual, product X is incompatible. Which future actions should change?
Type 6: Hidden-state inference
Given the changed observation, what hidden state most likely changed?

Example:

The agent expected the door to open, but it did not. Which hidden state variable is most likely different?

This is the inverse problem and is very important for world modeling.

Type 7: Counterfactual plan-step evaluation
If the agent had skipped step k, would the task still succeed?

Example:

If the agent had not verified the opening hours of the ruins, would the final plan be reliable?

This tests whether the model understands why an information-gathering step matters.

### 8. Benchmark-specific conversion recipes
#### A. ScienceWorld → counterfactual hidden-state data

ScienceWorld is probably the cleanest source because it is an interactive text environment for scientific reasoning, with commands that change the world state and observations.

#### Extract

For each episode:

task goal
action trace
observation trace
world state trace
object properties
device states
experiment result
final success
#### Hidden-state interventions

Examples:

change material property: conductive → nonconductive
change device state: battery charged → dead
change container contents: water → saltwater
change temperature: hot → cold
change object location: beaker on table → beaker in cabinet
change contamination state: clean → contaminated
#### Counterfactual questions
If the unknown material were nonconductive instead of conductive, what result would the conductivity test produce?
If the battery had been dead, would observing no light prove the material is nonconductive?
If the beaker already contained salt, would the conclusion about solubility still be valid?
#### Why good

This directly tests hidden-state causal reasoning, not just plan execution.

#### B. ALFWorld → counterfactual hidden-state data

ALFWorld uses interactive TextWorld environments aligned with embodied ALFRED tasks, so it is good for household object-state reasoning.

#### Extract
goal instruction
rooms
objects
containers
agent inventory
object states
action trace
observation trace
success/failure
#### Hidden-state interventions
object location changed
container secretly closed/open
object already dirty/clean
object already hot/cold
receptacle unavailable
object inside another container
#### Counterfactual questions
The factual trajectory succeeds because the apple is on the table. If the apple were inside the closed fridge instead, which action would first fail?
If the mug were already clean, which subgoal could be skipped?
If the microwave were broken, would the same plan still heat the potato?
#### Why good

It supports clear subgoal labels:

```text
find object → pick object → change object state → place object
```
#### C. TravelPlanner → counterfactual hidden-state data

TravelPlanner is highly aligned with your hierarchical planning idea because agents must plan transportation, meals, attractions, and accommodation while satisfying multiple constraints.

#### Extract
user request
destination
date range
budget
constraints
tool calls
returned records
reference plan
final predicted plan
constraint-check results
#### Hidden-state interventions
hotel unavailable
restaurant closed
attraction closed
transport delayed
hidden fee added
distance/time changed
weather makes activity infeasible
reservation required
flight arrival delayed
#### Counterfactual questions
If the selected hotel were unavailable on Night 3, which parts of the plan become invalid?
If the flight arrived 3 hours later, would the Day 1 attraction schedule still be feasible?
If the cenote tour required a reservation that the agent did not make, what minimal repair is needed?
If the restaurant is closed on Tuesday, does this only affect the meal subgoal or also the transportation schedule?
Hidden-state examples
```json
{
  "hidden_state": {
    "hotel_A_availability": false,
    "ferry_Cozumel_day4": false,
    "xplor_opening_hours": "9am-5pm",
    "tour_requires_reservation": true,
    "actual_transport_time": "2.5h"
  }
}
```
#### Why good

TravelPlanner is ideal for evaluating:

```text
goal → days → subgoals → constraints → hidden feasibility → minimal repair
```

This is probably the best non-simulator starting point for your paper.

#### D. WebArena → counterfactual hidden-state data

WebArena is a self-hostable web environment with realistic websites, external tools, and knowledge resources. It includes tasks across domains such as e-commerce, forums, software development, and content management.

#### Extract
task instruction
browser actions
DOM observations
backend database states
forms submitted
cart/account/project states
final success
#### Hidden-state interventions
product out of stock
user lacks permission
form validation rule changed
forum post deleted
issue already closed
price changed
shipping address invalid
coupon expired
search index stale
#### Counterfactual questions
If the product had been out of stock after the agent added it to the cart, which later action would fail?
If the user did not have admin permission, would the same website action succeed?
If the search result was stale, what additional verification step should the agent take?
#### Why good

This is realistic but harder, because hidden state may be distributed across backend database, DOM, and user account state.

### 9. Constructing “hidden-state reasoning” labels

For each counterfactual example, annotate the answer in structured form.

#### Required labels
```json
{
  "outcome": "success/failure/uncertain",
  "first_divergent_step": 5,
  "changed_hidden_state": "battery_charge",
  "changed_observation": "light bulb does not turn on",
  "affected_subgoals": ["test conductivity"],
  "unaffected_subgoals": ["collect materials"],
  "failure_reason": "test apparatus invalid",
  "minimal_repair": ["replace battery", "repeat test"],
  "reasoning_type": "hidden_precondition"
}
```
#### Optional labels
```json
{
  "causal_chain": [
    "battery dead",
    "circuit cannot close",
    "bulb does not light",
    "observation is ambiguous",
    "cannot infer material conductivity"
  ],
  "counterfactual_plan": [...],
  "state_diff": {
    "battery_charge": ["charged", "dead"]
  },
  "observation_diff": [...],
  "constraint_diff": [...]
}
```

The causal chain is especially useful for training.

### 10. Automatic generation algorithm

A practical version:

```python
def generate_cf_examples(env, task, policy_or_trace):
    factual = run_or_load_trajectory(env, task, policy_or_trace)
```

    ## 1. Save factual state/action/obs trace
    trace = factual.trace
    state_trace = factual.hidden_states

    examples = []

    ## 2. Select candidate time steps
```python
    for t in select_relevant_steps(trace):
        z_t = state_trace[t]
```

        ## 3. Select mutable hidden variables
```python
        for var in candidate_hidden_vars(z_t):
            for alt in plausible_values(var, z_t[var]):
```

                ## 4. Create counterfactual environment
                cf_env = clone_env_at_state(env, t)
                cf_env.set_hidden_state(var, alt)

                ## 5. Replay same future actions
                cf_replay = replay_actions(
                    cf_env,
                    actions=trace.actions[t:]
                )

                ## 6. Compare factual vs counterfactual
                diff = compare_trajectories(factual, cf_replay)

                if is_interesting(diff):
                    ## 7. Generate question/answer
                    example = build_cf_qa(
                        factual=factual,
                        counterfactual=cf_replay,
                        intervention=(var, alt),
                        diff=diff
                    )
                    examples.append(example)

    return examples

is_interesting(diff) should filter out changes that do not affect anything meaningful.

```python
def is_interesting(diff):
    return (
        diff.final_success_changed
        or diff.first_divergent_observation is not None
        or len(diff.affected_subgoals) > 0
        or diff.requires_repair
    )
```
### 11. How to make questions genuinely require hidden-state reasoning

A bad counterfactual question is too obvious:

If the hotel is unavailable, can you stay there?

A good hidden-state reasoning question requires delayed consequences:

The agent selected Hotel A because it is near the ferry terminal and within budget. Suppose Hotel A was actually unavailable on Night 3, but this is only discovered at checkout. Which later parts of the itinerary are affected, and what is the minimal repair?

Why better?

The model must reason about hidden availability.
It must identify downstream dependencies.
It must preserve unaffected plan parts.
It must revise minimally.
#### Add distractors

Include irrelevant hidden-state changes:

Hotel lobby has no gym.

This should not affect a sightseeing itinerary unless the user required a gym.

#### Add delayed effects

The changed state should affect something 3–8 steps later.

```text
Flight delay → missed check-in → hotel cancellation → no lodging → next day schedule affected
```
#### Add partial observability

The model should know that the agent cannot conclude something yet.

If the battery is dead, a failed conductivity test is ambiguous.

This tests epistemic reasoning, not just outcome prediction.

### 12. Train/test splits

To show the dataset is not just pattern matching, use splits that test generalization.

#### Split by hidden-state type

Train on:

availability changes, object-location changes

Test on:

device-state changes, permission changes
#### Split by benchmark domain

Train on:

TravelPlanner

Test on:

WebArena shopping
#### Split by intervention distance

Train on short-distance counterfactuals:

intervention affects next 1–2 steps

Test on long-distance counterfactuals:

intervention affects step t+5 or later
#### Split by hierarchy level

Train on action-level counterfactuals.

Test on:

subgoal-level or plan-level counterfactuals

This directly tests your hierarchical planning hypothesis.

### 13. Evaluation metrics

Use structured metrics instead of only answer accuracy.

#### Outcome accuracy
Did the model correctly predict success/failure/uncertain?
#### First divergence accuracy
Did it identify the first step where the counterfactual trajectory differs?
#### Affected subgoal F1
F1 between predicted affected subgoals and gold affected subgoals.
#### Hidden-state causal variable accuracy
Did it identify which hidden variable caused the change?
#### Minimal repair validity
Does the proposed repair satisfy constraints?
#### Minimal repair edit distance
How much unnecessary change did the model make?
#### Unaffected-subgoal preservation
Did the model preserve parts of the plan that did not need changing?
#### Causal chain score

Evaluate whether the explanation includes the required causal links:

```text
hidden state → changed observation → failed subgoal → final outcome
```

This is crucial for your paper because you want reasoning over hidden states, not just final answer prediction.

### 14. Dataset variants

You can create three versions with increasing difficulty.

#### Level 1: Direct hidden-state effect
Hidden state changes next observation immediately.

Example:

```text
Battery dead → bulb does not light.
```
#### Level 2: Delayed hidden-state effect
Hidden state affects a future subgoal several steps later.

Example:

```text
Hotel unavailable → check-in fails later → next day transport plan invalid.
```
#### Level 3: Branching counterfactual repair
Hidden state causes failure, and model must propose a minimal valid alternative trajectory.

Example:

```text
Cozumel ferry unavailable → choose whether to swap days, replace activity, or change hotel.
```

Level 3 is the most publishable and best aligned with hierarchical rewards.

### 15. How this becomes training data

Each example can be used in three training formats.

#### Format A: Supervised reasoning
```text
Input:
factual trajectory + hidden-state intervention + question
```

```text
Output:
causal explanation + outcome + repair
```
#### Format B: Preference data

Generate two answers:

A: correct counterfactual causal chain
B: shallow answer that ignores hidden state

Train with DPO/RLAIF-style preference optimization.

#### Format C: RL reward

Reward components:

```text
R = R_outcome
  + R_hidden_variable
  + R_affected_subgoal
  + R_causal_chain
  + R_repair_validity
  + R_minimality
```

This directly connects to your hierarchical reward idea.

### 16. Concrete first implementation plan

I would start with ScienceWorld + TravelPlanner.

#### Phase 1: ScienceWorld prototype

#### Why first?

It is executable.
Hidden states are clear.
Counterfactual outcomes can be simulated.
Scientific tasks naturally involve hidden causal variables.

Build 1,000–5,000 examples:

conductivity
melting/freezing
solubility
magnetism
plant growth
state of matter

Each example:

factual trajectory
hidden-state intervention
future observation prediction
outcome prediction
minimal repair
#### Phase 2: TravelPlanner realistic extension

Build 500–2,000 examples:

hotel availability
attraction opening hours
transport delays
hidden cost
weather infeasibility
reservation requirement

Use verifier-based labels.

#### Phase 3: Evaluation

Compare:

GPT-style LLM
open-source LLM
ReAct-style agent
plan-and-solve agent
agent with explicit state tracker
agent trained on your counterfactual data

Evaluate:

outcome accuracy
affected-subgoal F1
minimal-repair validity
long-distance dependency accuracy
### 17. Paper contribution framing

Your method can be framed as:

We convert existing long-horizon agent benchmarks into counterfactual hidden-state reasoning datasets by intervening on latent environment variables, simulating or verifying the resulting alternative trajectories, and asking models to predict affected observations, subgoals, outcomes, and minimal repairs.

This is stronger than just asking “what if?” because the dataset requires:

hidden system state reasoning
+ long-horizon trajectory prediction
+ hierarchical subgoal localization
+ minimal plan repair

That combination is the innovative part.
