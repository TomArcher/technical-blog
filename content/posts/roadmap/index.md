+++
title = "Start Here: The Signal & Syntax Roadmap"
subtitle = "What Signal & Syntax has covered, what is in progress, and what is planned"
author = "Tom Archer"
date = 2026-08-22
draft = false
pinned = true
+++

Signal & Syntax works through the mathematics, mechanics, and engineering of large language models from first principles, with a secondary interest in applied modeling and simulation. Each post is written to stand on its own, but many of them also fit into a larger arc. This page is where those arcs are visible in one place.

The roadmap is public because I want readers to know where the project is going, and because a public roadmap is a commitment device. It updates as I learn things that change the plan, so the "planned" section is not a fixed order or a promise of dates. Consider it an honest snapshot of what I intend to write next, subject to revision when the material teaches me something.

---

{{< roadmap-jump >}}

## Categories

The blog is organized into four categories, each doing different work.

**[AI and the Mathematics of Language](/categories/ai-and-the-mathematics-of-language/)**: The core of the project. Each post uses a specific mathematical lens (discrete math, linear algebra, probability, calculus, geometry) to explain how large language models work. Read together, they build a compound picture of how these systems represent, process, and generate language. Individual posts stand on their own, but the category as a whole is a sustained argument that the mathematics of language is not one subject but several, working together.

**[Applied Modeling and Simulation](/categories/applied-modeling-and-simulation/)**: Physics, probability, and engineering problems worked through with code. This category is where I explore modeling as a craft, using specific problems as anchors for broader techniques.

**[Essays and Perspectives](/categories/essays-and-perspectives/)**: Occasional pieces that step back from the technical material to reflect on process, career direction, or observations about the field.

**[Python Techniques and Tooling](/categories/python-techniques-and-tooling/)**: Focused posts on specific Python patterns, libraries, or comparisons between tools. Smaller in scope than the technical or modeling posts, but useful reference material.

---

## Published

### AI and the Mathematics of Language

This sequence walks through different mathematical views of how LLMs work. Reading order reflects conceptual dependencies: you need to understand tokenization before reading about code, embeddings before attention, and so on. The posts were not all written in reading order, but the connections between them establish a natural path.

- [How Large Language Models Tokenize Text](/posts/how-large-language-models-tokenize-text/). Subword tokenization, vocabulary construction, and why token boundaries matter more than they look like they should.

- [How Large Language Models Read Code](/posts/how-large-languag-models-read-code/). Why code is a special kind of text that models treat as probabilistic patterns rather than logical instructions, and where they succeed and fail.

- [How Large Language Models Think](/posts/how-large-languag-models-think/). Embeddings, linear algebra, and the geometry of meaning: how tokens become vectors and vectors navigate high-dimensional semantic space.

- [How Large Language Models Learn](/posts/how-large-languag-models-learn/). Gradient descent, backpropagation, and calculus in motion: how models improve through billions of derivative calculations.

- [How Large Language Models Handle Context Windows](/posts/how-large-language-models-handle-context-windows/). What a context window is, how it constrains what models can attend to, why longer is not always better, and the mathematics of attention dilution.

- [Inside Attention, Part 1: The Mechanism](/posts/inside-attention-part-1/). Scaled dot-product attention, the variance argument behind \\(\sqrt{d_k}\\), multi-head attention, and what interpretability research has revealed about the patterns and circuits attention can learn.

- [How Large Language Models Know Things They Were Never Taught](/posts/how-large-language-models-know-things-they-were-never-taught/). Emergence, generalization, retrieval-augmented generation, and the distinction between what models learned (frozen in weights) and what they can read (injected into context).

- [The Discrete Mathematics Hiding Inside LLMs](/posts/discrete-math-in-large-language-models/). How set theory, predicate logic, Boolean algebra, and modular arithmetic show up inside transformer architectures—not as separate concerns but as continuous approximations of discrete structures.

- [Temperature, Top-P, and the Creativity Knobs](/posts/temperature-top-p-creativity-knobs/). How sampling parameters shape model output: how temperature reshapes probability distributions and how top-p (nucleus sampling) dynamically truncates them.

### Applied Modeling and Simulation

- [The Birthday Paradox](/posts/birthday-paradox/). Why counterintuitive probability results have practical implications for hashing, security, and system design.

- [The Edmund Fitzgerald](/posts/edmund-fitzgerald/). A physics-and-modeling look at the 1975 sinking on Lake Superior.

- [The Five-Second Rule](/posts/five-second-rule/). Bacterial transfer modeled with real timescales.

- [The Meeting Diet](/posts/meeting-diet/). Applying optimization thinking to calendar management.

- [The Rain Paradox](/posts/rain-paradox/). Walking vs. running in the rain, worked from first principles.

- [Safe Distance in Traffic](/posts/safe-distance-in-traffic/). Following distance as a function of speed, reaction time, and stopping physics.

- [Thermodynamics and Water Energy Balance](/posts/thermo-water-energy-balance/). Heating water as a lens on thermodynamic bookkeeping.

- [Three-D Packing](/posts/three-d-packing/). Volume, geometry, and the limits of stacking.

### Essays and Perspectives

- [Hash Collisions](/posts/hash-collisions/). Why hash collisions matter more than they seem to, and what they teach about probability.

- [The Tom-First-Principles ChatGPT](/posts/tfp-chatgpt/). Reflections on using AI tools as thinking partners rather than answer machines.

### Python Techniques and Tooling

- [Python Dispatch Maps](/posts/python-dispatch-maps/). Cleaner alternatives to long if/elif chains.

- [Python Integer Division and Modulus](/posts/python-integer-division-and-modulus/). What the operators actually do, especially with negative numbers.

- [Python SymPy vs. NumPy](/posts/python-sympy-vs-numpy/). When symbolic beats numeric, and when it does not.

---

## In Progress

**Inside Attention (three-part sub-series within AI and the Mathematics of Language).** A dedicated arc on the attention mechanism, split across three posts because a single post could not do the material justice.

- *Part 2: Masking and the Function Class.* Causal vs. bidirectional masking, why the decoder-only architecture won, and how the choice of mask constrains the function class the model can learn. Outlined.

- *Part 3: The Production Stack.* The KV cache problem, multi-query and grouped-query attention, sliding window attention, and Flash Attention. The engineering layer that exists because the textbook formula does not survive scale. Outlined.

---

## Planned

These are the topics I intend to write next, in roughly the order I plan to tackle them. Ordering may shift as the material teaches me things.

**The Cooperative Witness Problem.** A piece on the ways language models tend to accept and continue user premises rather than push back on them, and the training dynamics that produce that tendency. Follows naturally from Part 2 of Inside Attention, which sets up the function-class framing this piece depends on.

**Pre-training vs. Post-training.** The distinction between the compute-heavy foundation training phase and the alignment phase that shapes model behavior (supervised fine-tuning, RLHF, DPO, Constitutional AI). One of the highest-confusion topics for readers new to the field and a natural companion to the cooperative witness piece.

**Softmax and Cross-Entropy Loss.** A focused post on the connective tissue between probability and learning. Softmax as the operation that turns real numbers into distributions; cross-entropy as the loss that measures how well those distributions match the truth. Short, foundational, unlocks the probability layer.

**The Feedforward Sublayer.** A standalone post on the FFN sublayer inside each transformer block: what it does, why it is often much larger than the attention sublayer, and what interpretability research has shown about the concepts stored there.

**In-Context Learning.** How models adapt to patterns within a single prompt without parameter updates, and why induction heads (covered in Inside Attention Part 1) are part but not all of the mechanistic story.

**Reasoning Models and Test-Time Compute.** The class of models that spend additional compute at inference time to improve their answers, and why this changes what "capability" means.

**Scaling Laws and Emergence.** The empirical relationships between compute, data, parameters, and capability, and where the sharp transitions in behavior come from.

**A Mechanistic Interpretability Primer.** An introduction to the research program of reverse-engineering trained transformers, at the level of concrete circuits and features rather than high-level intuitions.

Shorter topics also flagged for eventual treatment: Mixture of Experts, quantization, advanced decoding strategies, and the specific engineering of the KV cache (which may end up as part of Inside Attention Part 3 rather than a standalone post).

---

## How to Read This Page

The categories are stable. The published list grows as posts publish. The in-progress list reflects what I am actively working on at any given time. The planned list is a genuine intention but not a commitment to fixed dates or a fixed order. If a topic starts teaching me something that changes my sense of what should come next, the plan updates.

If you are reading this and there is a topic in the planned list you want to see sooner, or a topic missing that you think belongs here, [get in touch](https://www.linkedin.com/in/tom-archer-dev/). I do not always follow reader requests, but I always read them.
