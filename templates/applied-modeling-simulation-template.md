+++
title = "[TITLE]: [MODELING QUESTION OR PAYOFF]"
subtitle = "[OPTIONAL SUBTITLE]"
date = "YYYY-MM-DDT06:00:00-07:00"
draft = true
categories = ["Applied Modeling and Simulation"]
tags = ["[DOMAIN]", "Python", "[METHOD]", "[OTHER TAG]"]
author = "Tom Archer"
listThumb = "[THUMBNAIL].png"
+++

<!--
SIGNAL & SYNTAX TEMPLATE: APPLIED MODELING AND SIMULATION

Purpose:
Start with a real-world question, translate it into a model, implement the model,
test it, visualize it, and explain what the model does and does not establish.

Writing standard:
- Begin with the real phenomenon, not the code.
- Clearly separate observed facts from modeling assumptions.
- State assumptions before relying on them.
- Explain the governing equations and units.
- Build code in understandable stages.
- Validate against external observations/data when possible.
- Treat model output as model output, not proof of reality.
- Discuss limitations when simplifications materially affect conclusions.
-->

<figure style="float: right; margin: 0 20px 10px 20px; width: 250px; text-align: center;">
  <img src="./[IMAGE]" alt="[DESCRIPTIVE ALT TEXT]" width="250" style="display: block; margin: 0 auto;">
  <figcaption style="font-size: 0.9em; color: #555; margin-top: 5px;">
    <em>[CAPTION CONNECTING THE IMAGE TO THE MODELING QUESTION.]</em>
  </figcaption>
</figure>

[REAL-WORLD HOOK / OBSERVATION / QUESTION]

[WHY THE QUESTION IS SURPRISING OR WORTH MODELING.]

---

<!-- Optional content note when subject matter warrants one. -->
*Note: [CONTENT NOTE].*

---

> *"[QUOTE OR ONE-SENTENCE STATEMENT OF THE PUZZLE.]"*

---

<!--more-->

<!-- Learning goals can use the same compact sidebar convention when useful. -->
<div class="learning-sidebar">
<strong>What You'll Learn</strong>

After reading this post, you'll be able to explain:

- [PHYSICAL/MATHEMATICAL PRINCIPLE]
- [MODELING TECHNIQUE]
- [HOW VARIABLES INTERACT]
- [WHAT THE SIMULATION PREDICTS]
</div>

[ONE PARAGRAPH PREVIEW OF THE MODEL AND ITS PAYOFF.]

> **TL;DR:**
> - [RESULT]
> - [RESULT]
> - [RESULT]
> - [GENERALIZATION]

---

## The Thought Experiment

[FRAME THE PROBLEM AS A QUESTION WE CAN TURN INTO VARIABLES AND RELATIONSHIPS.]

---

## Modeling the Problem

[DEFINE STATE, INPUTS, OUTPUTS, AND IMPORTANT CONSTRAINTS.]

**Assumptions:**

- [ASSUMPTION]
- [ASSUMPTION]
- [ASSUMPTION]
- [ASSUMPTION]

### [GOVERNING EQUATION / PROCESS]

\[
[EQUATION]
\]

Where [DEFINE TERMS AND UNITS].

### [SECOND GOVERNING EQUATION / PROCESS]

\[
[EQUATION]
\]

[EXPLANATION]

---

## Building the Simulation

<!--
Build from small functions/data structures toward the complete simulation.
If AI was materially used to scaffold code, retain the "Prompt example" pattern.
Otherwise omit prompt boxes.
-->

### [COMPONENT 1]

**Prompt example:**

> [OPTIONAL PROMPT]

**Python:**
```python
# Focused component with docstrings and explicit units.
```

[EXPLAIN THE ROLE OF THIS COMPONENT.]

### [COMPONENT 2]

**Python:**
```python
# Next component.
```

### [COMBINED MODEL]

**Python:**
```python
# Compose the pieces into the simulation.
```

---

## Simulation: [REAL-WORLD CASE]

[STATE ACTUAL INPUT PARAMETERS AND THEIR SOURCES.]

**Python:**
```python
# Run the model for the focal case.
```

**Sample Output:**
```text
[OUTPUT]
```

[INTERPRET THE OUTPUT IN PLAIN LANGUAGE.]

---

## Validating the Model

<!-- Compare predictions against observations, historical cases, or known limits. -->

[VALIDATION METHOD]

**Python:**
```python
# Optional validation code.
```

[WHAT MATCHES, WHAT DOES NOT, AND WHAT THAT MEANS.]

---

## Making the Results Visual

### [VISUALIZATION 1]

**Python:**
```python
# Plot the most informative relationship.
```

<figure>
  <img src="./[PLOT].png" alt="[ALT TEXT]">
  <figcaption><em>[WHAT THE PLOT SHOWS.]</em></figcaption>
</figure>

### [OPTIONAL VISUALIZATION 2]

[REPEAT ONLY IF IT ADDS A DIFFERENT INSIGHT.]

---

## What We Learned

1. **[FINDING]:** [INTERPRETATION]
2. **[FINDING]:** [INTERPRETATION]
3. **[FINDING]:** [INTERPRETATION]
4. **[FINDING]:** [INTERPRETATION]

---

## Limitations

<!-- Include whenever simplifying assumptions materially constrain the claim. -->

- [LIMITATION]
- [LIMITATION]
- [WHAT A MORE COMPLETE MODEL WOULD ADD]

---

## Exercises for the Reader

**Beginner Level:**
1. [CHANGE ONE PARAMETER / REPRODUCE RESULT]
2. [SIMPLE EXTENSION]

**Intermediate Level:**
1. [ADD A VARIABLE OR MORE REALISTIC RELATIONSHIP]
2. [COMPARE SCENARIOS]

**Advanced Level:**
1. [PROBABILISTIC / NUMERICAL / DATA-DRIVEN EXTENSION]
2. [VALIDATE AGAINST A NEW DATASET OR CASE]

---

## Closing Thoughts

[RETURN TO THE ORIGINAL REAL-WORLD QUESTION AND STATE WHAT THE MODEL HELPS US UNDERSTAND.]

---

## Try It Yourself

[Download the full code on GitHub]([GITHUB URL])

---

## Further Reading

[PRIMARY DATA SOURCES, PAPERS, DOMAIN REFERENCES, AND MODELING SOURCES.]

---
