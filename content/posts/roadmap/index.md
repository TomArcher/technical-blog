+++
title = "What's Next: The Signal & Syntax Roadmap"
subtitle = "Want to know what's coming? Explore our roadmap to see what we're building, what's queued up, and what's already live."
author = "Tom Archer"
date = 2026-08-26
draft = false
pinned = true
+++

If there is a topic in the planned list you want to see sooner, or a topic missing that you think belongs here, [get in touch](https://www.linkedin.com/in/tom-archer-dev/).

{{< roadmap-jump >}}

---

## In Progress

**Implementing a Minimal Transformer in PyTorch.** Building a decoder-only Transformer from first principles, including token and positional embeddings, scaled dot-product attention, multi-head attention, feed-forward networks, residual connections, causal masking, training, and autoregressive generation. The implementation uses a character-level Shakespeare model to connect the mathematics of Transformer architecture directly to working PyTorch code. Drafted; in pre-publication review.

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

## Published

{{< roadmap-published >}}

---