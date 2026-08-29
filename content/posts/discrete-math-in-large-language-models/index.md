+++
author = "Tom Archer"

date = '2026-03-31T06:00:00-07:00'
draft = false

title = "The Discrete Mathematics Hiding Inside LLMs"
subtitle = "How set theory, predicate logic, and formal proofs show up in modern AI"


learningPaths = [
    "How Large Language Models Work",
    "Inside the Transformer"
]
categories = ["AI and the Mathematics of Language"]
tags = ["AI", "attention mechanisms", "embeddings", "discrete mathematics", "LLM", "predicate logic", "set theory"]

listThumb = "discrete-math-in-large-language-models-th.png"

hero = "discrete-math-in-large-language-models.png"
heroAlt = "A robotic head in profile with discrete mathematics concepts filling its mind: Venn diagrams, truth tables, logical symbols like P→Q and ∀x∃y, graph theory networks, binary matrices, and set notation, all connecting to neural network patterns"
heroLabel = "Open full-size discrete math in large language models"
heroCaption = "It's true what they say: discrete math is everywhere you look."

whatYoullLearn = [
    "How attention behaves like a soft version of predicate logic",
    "How top-k and top-p sampling use ideas from set theory",
    "How Boolean logic determines which tokens can attend to each other",
    "Why chain-of-thought reasoning resembles the structure of a proof",
    "How positional encoding uses periodic patterns to represent position",
    "How discrete math helps explain why common LLM techniques work"
]

[[references]]
key = "vaswani2017"
citation = "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30.*"
url = "https://arxiv.org/abs/1706.03762"

[[references]]
key = "wei2022"
citation = "Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems, 35.*"
url = "https://arxiv.org/abs/2201.11903"

[[references]]
key = "press2021"
citation = "Press, O., Smith, N. A., & Lewis, M. (2021). Train short, test long: Attention with linear biases enables input length generalization. *arXiv preprint arXiv:2108.12409.*"
url = "https://arxiv.org/abs/2108.12409"
+++

A recent [LinkedIn post](https://www.linkedin.com/posts/activity-7435689322504167424-_v46) from Michael Palmer described how {{< term "discrete-mathematics" "discrete mathematics" >}} is the foundation for how computers reason about problems. That thread got me thinking about just how many discrete math concepts show up inside systems that seem purely statistical. LLMs are often described in terms of neural networks, {{< term "gradient-descent" "gradient descent" >}}, and probability distributions. If you've taken discrete mathematics and wondered what it has to do with modern AI, the answer is: more than you'd expect.

Underneath the calculus and linear algebra, the same structures you learn in a discrete math course keep appearing: {{< term "set-theory" "sets" >}}, {{< term "predicate-logic" "predicate logic" >}}, {{< term "boolean-algebra" "Boolean operations" >}}, {{< term "modular-arithmetic" "modular arithmetic" >}}, formal proof patterns. This post traces those connections.

This post explores how core discrete mathematics concepts appear in LLM architecture and operation. You'll see set theory in token selection, predicate logic in attention mechanisms, Boolean algebra in prompt constraints, proof structure in chain-of-thought reasoning, and modular arithmetic in positional encoding. If you've followed my earlier posts on [how LLMs think](/posts/how-large-language-models-think/) and [how they learn](/posts/how-large-language-models-learn/), you've already seen the calculus and linear algebra. This post adds the discrete layer.

> **TL;DR:**
> * Attention mechanisms behave like soft, differentiable versions of predicate evaluation over tokens
> * Token filtering (top-k, top-p) can be understood through set construction and intersection
> * Masking operations are Boolean algebra applied to sequences
> * Chain-of-thought prompting often resembles informal proof structure found in training data
> * Positional encoding uses periodic functions, echoing the cyclic behavior seen in modular arithmetic
> * Understanding these connections clarifies why certain prompting strategies work
> * Discrete math isn’t separate from AI; it shows up as continuous approximations inside the architecture

---

## Predicate Logic in Attention

In discrete mathematics, predicate logic extends Boolean logic with quantifiers and variables. Instead of just “true or false,” you can express statements like:

\\[
\forall x \in S : P(x) \rightarrow Q(x) 
\\] 

“For all \\(x\\) in set \\(S\\), if \\(P(x)\\) is true, then \\(Q(x)\\) follows.” 

Large language models do not execute predicate logic in this formal sense. They are not constructing symbolic expressions, assigning truth values, or applying inference rules step by step. But the *structure* of predicate logic shows up in how they process language. This structure appears most clearly in {{< term "attention" "attention mechanisms" >}}. Recall that attention computes a weighted sum over tokens in the context. Each token evaluates how relevant other tokens are and aggregates information accordingly. Rather than applying a discrete predicate that returns true or false, attention computes a *continuous relevance score*. 

Here’s the standard attention formula: 

\\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V 
\\] 

We can interpret this through a predicate-logic lens:
- **{{< term "query" "Query" >}} (Q)**: a learned representation of “what I’m looking for” 
- **{{< term "key" "Keys" >}} (K)**: representations of what each token “offers” 
- **\\(QK^T\\)**: a similarity score measuring how well each token matches that intent 
- **{{< term "softmax" >}}**: converts those scores into a probability distribution 
- **{{< term "value" "Values" >}} (V)**: the information extracted from relevant tokens 

Viewed this way, attention behaves like a *soft, differentiable scan over a set of tokens*, where each element is evaluated for relevance and contributes proportionally to the result. Instead of: “Select all tokens where \\(P(x)\\) is \\(true\\)” the model effectively does: “Assign each token a degree of relevance based on how well it matches a learned pattern, then aggregate accordingly” This is not predicate logic in the formal sense. But it is a close structural analogue: evaluating a condition across a set and acting on the results. The difference is that everything is continuous, probabilistic, and learned rather than discrete and symbolic. That shift, from crisp logical predicates to soft geometric approximations, is where modern AI lives.

#### Python

```python
import numpy as np

def attention_as_predicate_logic():
    """
    Demonstrate how attention implements predicate-like selection.
    """
    # Simplified example: 5 tokens, embedding dimension 4
    tokens = ["The", "cat", "sat", "on", "mat"]
    
    # Pretend embeddings (in reality these are learned)
    embeddings = np.random.randn(5, 4)
    
    # Query: "I'm looking for nouns"
    # In reality, this is learned; here we'll simulate
    query = np.array([0.1, 0.8, 0.2, 0.1])  # "noun-detecting" query
    
    # Keys: what each token "advertises"
    keys = embeddings  # Simplified: keys = embeddings
    
    # Compute relevance scores (QK^T)
    scores = np.dot(keys, query)
    
    # Softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    print("Predicate: 'Which tokens are relevant to my query?'\n")
    print("Token\t\tScore\t\tWeight")
    print("-" * 40)
    for token, score, weight in zip(tokens, scores, weights):
        bar = "█" * int(weight * 30)
        print(f"{token}\t\t{score:.2f}\t\t{weight:.2%} {bar}")
    
    print("\nAttention implements: ∀x ∈ context : relevance(x) → weight(x)")

attention_as_predicate_logic()
```

The attention mechanism doesn't use explicit logical symbols, but it computes the same structure: evaluate a predicate across all elements, then act on the results.

---

## Set Operations in Token Selection

Set theory is foundational to discrete mathematics. Union, intersection, complement, and membership define how collections relate to each other.

Token selection in LLMs is set theory applied to vocabularies.

**{{< term "top-k-sampling" "Top-k sampling" >}}** selects the k highest-probability tokens:

\\[
S_k = \{t \in V : \text{rank}(P(t)) \leq k\}
\\]

This is set construction with a membership predicate. "The set of all tokens in vocabulary \\(V\\) whose probability rank is at most \\(k\\)."

**{{< term "top-p-sampling" "Top-p (nucleus) sampling" >}}** selects tokens until cumulative probability exceeds \\(p\\):

\\[
S_p = \{t_1, t_2, ..., t_m\} \text{ where } \sum_{i=1}^{m} P(t_i) \geq p
\\]

This is building a set by accumulation until a threshold condition is met.

**Combining methods** is set intersection:

\\[
S_{final} = S_k \cap S_p
\\]

"Tokens that are both in the top-k and contribute to the top-p cumulative probability."

#### Python

```python
def token_selection_as_sets():
    """
    Show how top-k and top-p sampling are set operations.
    """
    # Vocabulary with probabilities
    vocab = {
        "the": 0.25,
        "a": 0.18,
        "cat": 0.15,
        "dog": 0.12,
        "sat": 0.10,
        "ran": 0.08,
        "quickly": 0.05,
        "slowly": 0.04,
        "yesterday": 0.02,
        "tomorrow": 0.01,
    }
    
    # Sort by probability
    sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])
    
    # Top-k: select top 5 tokens
    k = 5
    top_k_set = set(token for token, _ in sorted_vocab[:k])
    
    # Top-p: select until cumulative probability >= 0.7
    p = 0.7
    top_p_set = set()
    cumulative = 0
    for token, prob in sorted_vocab:
        top_p_set.add(token)
        cumulative += prob
        if cumulative >= p:
            break
    
    # Set operations
    intersection = top_k_set & top_p_set
    union = top_k_set | top_p_set
    
    print("Set Theory in Token Selection:\n")
    print(f"Top-k (k=5):     {top_k_set}")
    print(f"Top-p (p=0.7):   {top_p_set}")
    print(f"Intersection:    {intersection}")
    print(f"Union:           {union}")
    print(f"\nFinal sampling pool = Top-k ∩ Top-p = {intersection}")

token_selection_as_sets()
```

#### Output

```
Set Theory in Token Selection:

Top-k (k=5):     {'the', 'a', 'cat', 'dog', 'sat'}
Top-p (p=0.7):   {'the', 'a', 'cat', 'dog'}
Intersection:    {'the', 'a', 'cat', 'dog'}
Union:           {'the', 'a', 'cat', 'dog', 'sat'}

Final sampling pool = Top-k ∩ Top-p = {'the', 'a', 'cat', 'dog'}
```

The vocabulary is the universal set. Every sampling method defines a subset. Combining methods is combining sets.

---

## Boolean Algebra in Masking

Boolean algebra deals with operations on true/false values: AND, OR, NOT, XOR. In LLMs, masking operations apply Boolean logic to sequences.

**Attention masking** prevents tokens from attending to certain positions:

- **{{< term "causal-mask" "Causal mask" >}}**: Token at position i can only attend to positions ≤ i
- **{{< term "padding-mask" "Padding mask" >}}**: Ignore padding tokens entirely
- **Custom masks**: Task-specific attention patterns

These are Boolean matrices. A 1 means "allowed," a 0 means "blocked."

#### Python

```python
def boolean_masking():
    """
    Show how attention masking is Boolean algebra.
    """
    import numpy as np
    
    seq_len = 5
    tokens = ["The", "cat", "sat", "on", "mat"]
    
    # Causal mask: lower triangular matrix
    # Token i can attend to tokens 0, 1, ..., i
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=int))
    
    print("Causal Mask (Boolean matrix):")
    print("Can token[row] attend to token[col]?\n")
    print("\t" + "\t".join(tokens))
    for i, token in enumerate(tokens):
        row = "\t".join(str(x) for x in causal_mask[i])
        print(f"{token}\t{row}")
    
    # Padding mask example
    # Suppose last token is padding
    padding_mask = np.array([1, 1, 1, 1, 0])  # 0 = ignore
    
    print(f"\nPadding mask: {padding_mask}")
    print("(1 = real token, 0 = padding)")
    
    # Combined mask: AND operation
    # Can attend only if causal allows AND token is not padding
    combined = causal_mask * padding_mask  # Broadcasting
    
    print("\nCombined mask (causal AND not-padding):")
    print("\t" + "\t".join(tokens))
    for i, token in enumerate(tokens):
        row = "\t".join(str(x) for x in combined[i])
        print(f"{token}\t{row}")
    
    print("\nMask combination is Boolean AND: combined = causal ∧ padding")

boolean_masking()
```

#### Output

```
Causal Mask (Boolean matrix):
Can token[row] attend to token[col]?

	The	cat	sat	on	mat
The	1	0	0	0	0
cat	1	1	0	0	0
sat	1	1	1	0	0
on	1	1	1	1	0
mat	1	1	1	1	1

Padding mask: [1 1 1 1 0]
(1 = real token, 0 = padding)

Combined mask (causal AND not-padding):
	The	cat	sat	on	mat
The	1	0	0	0	0
cat	1	1	0	0	0
sat	1	1	1	0	0
on	1	1	1	1	0
mat	1	1	1	1	0

Mask combination is Boolean AND: combined = causal ∧ padding
```

Every masking operation is Boolean algebra applied element-wise to matrices.

---

## Proof Structure in Chain-of-Thought

Formal proofs in discrete mathematics follow a structure: state assumptions, apply inference rules step by step, derive conclusions. Each step must follow logically from previous steps.

{{< term "chain-of-thought" "Chain-of-thought (CoT) prompting" >}} [(Wei et al., 2022)](#wei2022) mirrors this structure. Instead of asking for an answer directly, you ask the model to reason step by step:

**Direct prompt:**
> What is 17 × 24?

**Chain-of-thought prompt:**
> What is 17 × 24? Think step by step.

The step-by-step response resembles a proof:

1. **Given**: 17 × 24
2. **Decompose**: 17 × 24 = 17 × (20 + 4)
3. **Distribute**: = (17 × 20) + (17 × 4)
4. **Compute**: = 340 + 68
5. **Conclude**: = 408

This is the same structure as a mathematical proof:

1. **Premises**: What we're given
2. **Inference steps**: Apply rules to derive new facts
3. **Conclusion**: Final result

#### Python

```python
def proof_structure_in_cot():
    """
    Compare formal proof structure to chain-of-thought.
    """
    
    # Formal proof structure
    formal_proof = {
        "Given": "Prove that if n is even, then n² is even",
        "Assume": "Let n be even. Then n = 2k for some integer k.",
        "Derive": "n² = (2k)² = 4k² = 2(2k²)",
        "Conclude": "Since n² = 2(2k²) and 2k² is an integer, n² is even. QED"
    }
    
    # Chain-of-thought structure
    cot_response = {
        "Given": "Calculate the total cost of 3 items at $17.50 each with 8% tax",
        "Step 1": "First, find the subtotal: 3 × $17.50 = $52.50",
        "Step 2": "Next, calculate the tax: $52.50 × 0.08 = $4.20",
        "Conclude": "Total cost: $52.50 + $4.20 = $56.70"
    }
    
    print("Formal Proof Structure:")
    print("-" * 50)
    for step, content in formal_proof.items():
        print(f"  {step}: {content}")
    
    print("\n\nChain-of-Thought Structure:")
    print("-" * 50)
    for step, content in cot_response.items():
        print(f"  {step}: {content}")
    
    print("\n\nBoth follow: Premises → Inference Steps → Conclusion")

proof_structure_in_cot()
```

Why does CoT improve accuracy? One hypothesis: the model's training data contains many examples of step-by-step reasoning (textbooks, tutorials, Stack Overflow answers). By prompting for explicit steps, you're activating patterns the model learned from structured explanations rather than patterns it learned from direct Q&A.

The proof structure isn't just an analogy. It's a shared pattern that appears in training data and gets activated by step-by-step prompts.

---

## Modular Arithmetic in Positional Encoding

Modular arithmetic is central to discrete mathematics. The clock "wraps around" from 12 to 1. Cryptographic systems like RSA depend entirely on properties of modular exponentiation.

{{< term "positional-encoding" "Positional encoding" >}} in {{< term "transformer" "transformers" >}} uses similar periodic structures.

The original transformer paper ([Vaswani et al., 2017](#vaswani2017)) used sinusoidal positional encodings:

\\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
\\]
\\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
\\]

These are periodic functions. As position increases, the values cycle through patterns at different frequencies. Each dimension captures periodicity at a different scale, similar to how different moduli in modular arithmetic create different cycle lengths.

More recent work has explored alternatives like {{< term "alibi" "ALiBi" >}} [(Press et al., 2021)](#press2021), but the principle remains: periodic or relative structures encode position.

#### Python

```python
def positional_encoding_periodicity():
    """
    Show the periodic structure in positional encodings.
    """
    import numpy as np
    
    def positional_encoding(pos, d_model=16):
        """Compute positional encoding for a single position."""
        pe = np.zeros(d_model)
        for i in range(0, d_model, 2):
            denominator = 10000 ** (i / d_model)
            pe[i] = np.sin(pos / denominator)
            if i + 1 < d_model:
                pe[i + 1] = np.cos(pos / denominator)
        return pe
    
    # Show how different dimensions have different periods
    positions = range(20)
    
    print("Positional Encoding Values (selected dimensions):\n")
    print("Position\tDim 0\t\tDim 4\t\tDim 8")
    print("-" * 55)
    
    for pos in positions:
        pe = positional_encoding(pos)
        print(f"{pos}\t\t{pe[0]:+.3f}\t\t{pe[4]:+.3f}\t\t{pe[8]:+.3f}")
    
    print("\nDimension 0 cycles fastest (short period)")
    print("Higher dimensions cycle slower (longer period)")
    print("Like different moduli creating different cycle lengths")

positional_encoding_periodicity()
```

The connection to modular arithmetic is structural: both create periodic patterns that encode position or equivalence classes. In RSA, you compute \\(m^e \mod n\\). In positional encoding, you compute \\(\sin(pos / \text{scale})\\). Both use periodic functions to encode information in a bounded, cyclic way.

---

## Why These Connections Matter

Understanding the discrete mathematics inside LLMs isn't just intellectual curiosity. It has practical implications.

**Prompting strategies make more sense.** Chain-of-thought works because it activates proof-like patterns. Asking for specific formats works because you're constraining the output set. Understanding the underlying structure helps you design better prompts.

**Debugging becomes clearer.** When a model fails at a reasoning task, you can ask: which step in the implicit proof went wrong? Where did the set membership predicate misfire? The discrete math lens gives you a vocabulary for analyzing failures.

**Architecture choices make more sense.** Why sinusoidal positional encodings? Because periodic functions encode position without bound while maintaining relative distance information. The modular arithmetic structure is the answer.

**The foundations connect.** Discrete math, linear algebra, calculus, probability: these aren't separate subjects that happen to appear in ML. They're interconnected views of the same structures. Attention is both a weighted sum (linear algebra) and a predicate evaluation (discrete math) and a probability distribution (statistics).

---

## From Abstraction to Application

When I first encountered set theory and predicate logic in a discrete mathematics course, the applications weren't obvious. Venn diagrams and truth tables felt disconnected from writing software.

The connection became clear when I started digging into how LLMs actually work. The same structures keep appearing, just implemented in continuous approximations rather than crisp symbolic form:

This is not a set of literal equivalences, but a set of structural analogies that help explain how these systems behave.

| Discrete Math Lens                          | What It Helps Explain in LLMs          |
|---------------------------------------------|----------------------------------------|
| Predicate logic                             | Attention as soft relevance evaluation |
| Set theory                                  | Token selection and sampling           |
| Boolean algebra                             | Masking and constraints                |
| Proof structure                             | Chain-of-thought reasoning patterns    |
| Cyclic structures (like modular arithmetic) | Positional encoding                    |

The abstractions aren't abstract. They're the scaffolding that neural networks approximate at scale.

---

## Closing Thoughts

The conversation that started this post discussed how discrete mathematics connects to real software engineering. For AI systems, the answer is unambiguous: discrete math is embedded in the architecture.

This doesn't mean you need a discrete math course to use LLMs. But if you want to understand why they work, why certain prompting strategies succeed, and why certain failures occur, the discrete structures provide clarity. Attention isn't magic; it's predicate evaluation. Token selection isn't arbitrary; it's set construction. Chain-of-thought isn't a trick; it's proof structure.

The foundations were always there. Modern AI just implements them at scale.

If you've followed this series from [how LLMs read code](/posts/how-large-language-models-read-code) through [how they think](/posts/how-large-language-models-think), [how they learn](/posts/how-large-language-models-learn), and [how they handle context](/posts/how-large-language-models-handle-context-windows), you've seen the calculus, linear algebra, and probability. This post adds the discrete layer. The mathematics of language isn't one subject. It's all of them, working together.

---
