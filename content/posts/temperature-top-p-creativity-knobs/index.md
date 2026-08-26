+++
date = '2025-12-24T12:00:00-08:00'
draft = false
title = "Temperature and Top-P: The Creativity Knobs"
subtitle = "How sampling parameters shape AI personality"
categories = ["AI and the Mathematics of Language"]
tags = ["AI", "Anthropic", "LLM", "nucleus sampling", "OpenAI", "probability", "sampling", "softmax", "temperature", "top-p",]
author = "Tom Archer"
listThumb = "temperature-top-p-creativity-knobs.png"

hero = "temperature-top-p-creativity-knobs.png"
heroAlt = "A probability distribution being reshaped by temperature, with tokens spreading from a sharp peak to a flattened curve"
heroLabel = "Open full-size temperature top p creativity knobs"
heroCaption = "Turn it up and things get weird."

whatYoullLearn = [
    "How an LLM turns raw token scores into probabilities before choosing what comes next",
    "How temperature reshapes a probability distribution to make outputs more predictable or more varied",
    "How top-p sampling limits which tokens the model is allowed to consider",
    "Why top-p adapts to model confidence differently from a fixed top-k cutoff",
    "How temperature and top-p interact when both are applied to the same distribution",
    "How to choose sampling settings for factual, structured, professional, and creative tasks"
]

[[references]]
key = "goodfellow2016"
citation = "Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press."
url = "https://mitpress.mit.edu/9780262035613/deep-learning/"

[[references]]
key = "hinton2015"
citation = "Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*."
url = "https://arxiv.org/abs/1503.02531"

[[references]]
key = "holtzman2020"
citation = "Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The curious case of neural text degeneration. *Proceedings of the International Conference on Learning Representations (ICLR)*."
url = "https://arxiv.org/abs/1904.09751"

[[references]]
key = "jurafsky2023"
citation = "Jurafsky, D., & Martin, J. H. (2023). *Speech and language processing* (3rd ed. draft). Stanford University."
url = "https://web.stanford.edu/~jurafsky/slp3/"

[[references]]
key = "openai2023"
citation = "OpenAI. (2023). API reference: Chat completions."
url = "https://platform.openai.com/docs/api-reference/chat"

[[references]]
key = "peng2023"
citation = "Peng, B., Galley, M., He, P., Cheng, H., Xie, Y., Hu, Y., ... & Gao, J. (2023). Check your facts and try again: Improving large language models with external knowledge and automated feedback. *arXiv preprint arXiv:2302.12813*."
url = "https://arxiv.org/abs/2302.12813"

+++

Every API call to [ChatGPT](https://openai.com/api/), [Claude](https://claude.com/platform/api), or any other {{< term "large-language-model" "LLM" >}} includes two parameters most people either ignore or tweak randomly: {{< term "temperature" "temperature" >}} and {{< term "top-p" "top-p" >}}. The defaults work fine for casual use, so why bother understanding them? Because these two numbers fundamentally control how your model thinks. 

The temperature value determines whether the model plays it safe or takes creative risks while the top-p value decides how many options the model even considers. Together, these values shape the personality of every response you receive.

I've watched developers cargo-cult settings from others without understanding what they do. "Set temperature to 0.7 for creative writing" becomes tribal knowledge, passed down without explanation. Let's fix that by opening the hood and examining the mathematics that makes these knobs work.


This post explores the mathematical foundations of token sampling in large language models, showing exactly how temperature and top-p transform {{< term "probability-distribution" "probability distributions" >}} into actual text. You'll see the equations, run the code, and develop intuition for when to reach for which parameter.

> **TL;DR:**
> - LLMs don't output text directly; they output probability distributions over {{< term "token" "tokens" >}} 
> - Temperature divides the {{< term "logit" "logits" >}} before {{< term "softmax" "softmax" >}}, reshaping the probability curve
> - Lower temperature makes the model more deterministic; temperature near zero always picks the top choice
> - Higher temperature spreads probability across more tokens, approaching uniform randomness at extreme values
> - Top-p ({{< term "nucleus-sampling" "nucleus sampling" >}}) dynamically truncates the distribution, keeping only tokens that sum to probability \\(p\\)
> - Temperature affects the shape of probabilities; top-p affects how many tokens remain in consideration
> - For factual tasks: low temperature (0.1–0.3), top-p around 0.9
> - For creative tasks: higher temperature (0.7–1.0), top-p around 0.95
> - Using both simultaneously can produce unexpected interactions; understand them individually first
> - Temperature 0 isn't truly deterministic in all implementations due to floating-point issues

---

## How LLMs Generate Text

Before we touch the knobs, we need to understand what they're adjusting. Large language models don't generate text directly. They generate probability distributions over vocabularies (typically 50,000 to 100,000 tokens) and then sample from those distributions to select the next token ([Jurafsky & Martin, 2023](#jurafsky2023)).

Here's the pipeline:

1. **Input processing**: Your prompt gets tokenized into a sequence of integers
2. **Forward pass**: The transformer produces a vector of raw scores (logits) for every token in the {{< term "vocabulary" "vocabulary" >}}
3. **Softmax**: Logits get converted to probabilities that sum to 1
4. **{{< term "sampling" "Sampling" >}}**: A token is selected based on those probabilities
5. **Repeat**: The selected token gets appended, and we go back to step 2

Steps 3 and 4 are where temperature and top-p operate. They don't change what the model "knows"; they change how it decides.

---

## The Softmax Function: Probabilities from Scores

The transformer's final layer outputs *logits*, which are unbounded real numbers where higher means "more likely." To convert these to probabilities, we apply the **softmax** function ([Goodfellow et al., 2016](#goodfellow2016)):

\\[
P(token_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
\\]

Where \\(z_i\\) is the logit for token \\(i\\) and \\(V\\) is the vocabulary size.

The exponential function amplifies differences: if token \\(A\\) has logit 5.0 and token \\(B\\) has logit 4.0, their probabilities won't be in a 5:4 ratio. **The exponentials make \\(A\\) roughly 2.7 times more likely than \\(B\\).**

```python
# softmax.py

import numpy as np

def softmax(logits):
    """Standard softmax: convert logits to probabilities."""
    # Subtract max for numerical stability
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits)

def main():
    # Example logits for 5 tokens
    logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
    probs = softmax(logits)
    print(f"Probabilities: {probs.round(3)}")
    # Output: [0.429 0.26  0.158 0.096 0.058]
```

Notice how a logit of 2.0 doesn't give twice the probability of logit 1.0. It gives nearly three times the probability. **This exponential amplification is exactly what temperature modifies.**

---

## Temperature: Reshaping Confidence

Temperature is mathematically simple: divide the logits by a scalar before applying softmax ([Hinton et al., 2015](#hinton2015)):

\\[
P(token_i | T) = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}
\\]

That's it. One division. But watch what happens:

**When T < 1** (low temperature): Dividing by a fraction amplifies the differences between logits. A gap of 1.0 between two logits becomes 2.0 after dividing by T=0.5, or 5.0 after dividing by T=0.2. This makes the highest-probability token dominate even further.

**When T = 1**: Standard softmax. No modification.

**When T > 1** (high temperature): Dividing by a number greater than 1 compresses differences. Logit gaps shrink. The distribution flattens toward uniform.

**When T → 0**: The highest logit wins with probability approaching 1. Deterministic {{< term "greedy-decoding" "greedy decoding" >}}.

**When T → ∞**: All tokens approach equal probability. Pure randomness.

```python
# softmax_with_temperature.py

import numpy as np
import matplotlib.pyplot as plt

def softmax_with_temperature(logits, temperature):
    """Apply temperature scaling before softmax."""
    if temperature == 0:
        # Greedy: return one-hot for max logit
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1.0
        return result
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits)

def main():
    # Same logits, different temperatures
    logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    print("Token probabilities at different temperatures:")
    print("-" * 50)
    for T in temperatures:
        probs = softmax_with_temperature(logits, T)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        print(f"T={T:.1f}: {probs.round(3)} | Entropy: {entropy:.2f}")
```

**Output**:
```
Token probabilities at different temperatures:
--------------------------------------------------
T=0.1: [0.993 0.007 0.    0.    0.   ] | Entropy: 0.04
T=0.5: [0.636 0.234 0.086 0.032 0.012] | Entropy: 1.00
T=1.0: [0.429 0.26  0.158 0.096 0.058] | Entropy: 1.39
T=1.5: [0.349 0.25  0.179 0.129 0.092] | Entropy: 1.51
T=2.0: [0.31  0.241 0.188 0.146 0.114] | Entropy: 1.55
```

The **{{< term "entropy" "entropy" >}}** column, a measure of probability distribution spread, quantifies what we're seeing: low temperature concentrates probability (low entropy), high temperature spreads it (high entropy approaching the maximum of \\(ln(5) ≈ 1.61\\) for 5 tokens).

---

## Visualizing Temperature Effects

Let's see this graphically with a realistic vocabulary slice:

**Python:**

```python
# visualize_temperature_effects.py

import numpy as np
import matplotlib.pyplot as plt
from softmax_with_temperature import softmax_with_temperature

def visualize_temperature_effects():
    """Show how temperature reshapes probability distributions."""
    # Simulate logits for 20 tokens (sorted for visualization)
    np.random.seed(42)
    logits = np.sort(np.random.randn(20) * 2)[::-1]

    temperatures = [0.3, 0.7, 1.0, 1.5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, T in zip(axes, temperatures):
        probs = softmax_with_temperature(logits, T)
        bars = ax.bar(range(20), probs, color='steelblue', alpha=0.7)

        # Highlight top token
        bars[0].set_color('darkred')

        ax.set_title(f'Temperature = {T}', fontsize=14)
        ax.set_xlabel('Token rank')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, max(probs) * 1.1)

        # Annotate top probability
        ax.annotate(f'{probs[0]:.1%}', xy=(0, probs[0]),
                    xytext=(2, probs[0]), fontsize=10)

    plt.tight_layout()
    plt.savefig('temperature_comparison.png', dpi=150)
    plt.show()
```

**Output:**

At T=0.3, the top token claims nearly all the {{< term "probability-mass" "probability mass" >}}, so the model will almost always choose its first instinct. At T=1.5, probability spreads across many tokens, introducing genuine variety (and risk).

{{< lightbox
    src="visualize_temperature_effects.png"
    alt="Four bar charts comparing token probability distributions at temperatures 0.3, 0.7, 1.0, and 1.5, showing probability spreading across more tokens as temperature increases"
    label="Open full-size temperature effects visualization"
    caption="As temperature increases from 0.3 to 1.5, probability spreads across more candidate tokens, reducing the dominance of the highest-probability token and increasing output diversity."
>}}

---

## Top-P (Nucleus Sampling): Dynamic Truncation

While temperature reshapes the entire distribution, top-p takes a different approach: it truncates the distribution dynamically, keeping only the tokens needed to reach {{< term "cumulative-probability" "cumulative probability" >}} \\(p\\) ([Holtzman et al., 2020](#holtzman2020)).

The algorithm:
1. Sort tokens by probability (descending)
2. Compute cumulative sum
3. Find the smallest set where cumulative probability ≥ \\(p\\)
4. Zero out everything else
5. Renormalize

```python
# top_p_sampling

import numpy as np
from softmax_with_temperature import softmax_with_temperature

def top_p_sampling(logits, p, temperature=1.0):
    """Apply nucleus (top-p) sampling."""
    # First apply temperature
    probs = softmax_with_temperature(logits, temperature)

    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Find cumulative sum
    cumsum = np.cumsum(sorted_probs)

    # Find cutoff index (first index where cumsum >= p)
    cutoff_idx = np.searchsorted(cumsum, p) + 1

    # Create mask
    mask = np.zeros_like(probs)
    mask[sorted_indices[:cutoff_idx]] = 1

    # Apply mask and renormalize
    masked_probs = probs * mask
    return masked_probs / np.sum(masked_probs)

def main():
    # Example
    logits = np.array([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0])
    tokens = ['the', 'a', 'one', 'some', 'that', 'this', 'an', 'my']

    print("Original probabilities:")
    orig_probs = softmax_with_temperature(logits, 1.0)
    for tok, prob in zip(tokens, orig_probs):
        print(f"  {tok}: {prob:.3f}")

    print(f"\nAfter top-p=0.9:")
    nucleus_probs = top_p_sampling(logits, p=0.9, temperature=1.0)
    for tok, prob in zip(tokens, nucleus_probs):
        if prob > 0:
            print(f"  {tok}: {prob:.3f}")
```

**Output**:
```
Original probabilities:
  the: 0.437
  a: 0.265
  one: 0.161
  some: 0.059
  that: 0.036
  this: 0.022
  an: 0.013
  my: 0.008

After top-p=0.9:
  the: 0.474
  a: 0.287
  one: 0.174
  some: 0.064
```

Notice that top-p=0.9 kept 7 of 8 tokens here, but the key insight is that this cutoff is dynamic. If the model is confident (one token has 95% probability), top-p=0.9 might keep only that one token. If the model is uncertain, it might keep dozens.

This is the crucial difference from {{< term "top-k-sampling" "top-k sampling" >}}, which always keeps exactly \\(k\\) tokens regardless of the distribution shape.

---

## Why Nucleus Sampling Beats Top-K

The original top-p paper ([Holtzman et al., 2020](#holtzman2020)) demonstrated a key failure mode of fixed top-k sampling. Consider two scenarios:

**Scenario A**: Model is very confident
- Token 1: 92%
- Token 2: 5%
- Token 3: 2%
- Tokens 4-10: < 1% combined

With top-k=10, you're including 7+ tokens that together contribute less than 1% probability. They'll rarely be selected, but when they are, you get incoherent outputs.

**Scenario B**: Model is genuinely uncertain
- Tokens 1-5: 15% each
- Tokens 6-10: 5% each

With top-k=5, you're excluding tokens 6-10 that together represent 25% of the model's considered probability mass. You're artificially constraining legitimate options.

Top-p handles both gracefully:
- In Scenario A, p=0.9 keeps only tokens 1-2
- In Scenario B, p=0.9 keeps tokens 1-8

```python
# compare_topk_topp.py

import numpy as np
from softmax_with_temperature import softmax_with_temperature

def compare_topk_topp():
    """Demonstrate when top-p outperforms top-k."""

    # Scenario A: Confident model
    confident_logits = np.array([5.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])

    # Scenario B: Uncertain model
    uncertain_logits = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.3, 0.25, 0.2])

    for name, logits in [("Confident", confident_logits),
                         ("Uncertain", uncertain_logits)]:
        probs = softmax_with_temperature(logits, 1.0)
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]

        # Top-k=3
        topk_included = 3
        topk_mass = np.sum(sorted_probs[:topk_included])

        # Top-p=0.9
        cumsum = np.cumsum(sorted_probs)
        topp_included = np.searchsorted(cumsum, 0.9) + 1

        print(f"{name} model:")
        print(f"  Top-k=3: includes {topk_included} tokens, "
              f"captures {topk_mass:.1%} of mass")
        print(f"  Top-p=0.9: includes {topp_included} tokens, "
              f"captures 90% of mass")
        print()
```

**Output:**
```
Confident model:
  Top-k=3: includes 3 tokens, captures 99.0% of mass
  Top-p=0.9: includes 1 tokens, captures 90% of mass

Uncertain model:
  Top-k=3: includes 3 tokens, captures 48.0% of mass
  Top-p=0.9: includes 7 tokens, captures 90% of mass
```
---

## The Interaction Between Temperature and Top-P

Here's where things get subtle. When you use both parameters together, temperature applies first, then top-p filters the result. This means:

1. **High temperature + top-p**: Temperature flattens the distribution, so more tokens survive the top-p cutoff
2. **Low temperature + top-p**: Temperature sharpens the distribution, so fewer tokens survive

```python
# temp_topp_interaction.py

import numpy as np
from softmax_with_temperature import softmax_with_temperature

def temp_topp_interaction():
    """Show how temperature affects top-p token counts."""
    logits = np.random.randn(100) * 2  # 100 tokens
    logits = np.sort(logits)[::-1]  # Sort descending

    temperatures = [0.3, 0.5, 0.7, 1.0, 1.3]
    p = 0.9

    print(f"Tokens included in top-p={p} nucleus at different temperatures:")
    print("-" * 50)

    for T in temperatures:
        probs = softmax_with_temperature(logits, T)
        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)
        tokens_included = np.searchsorted(cumsum, p) + 1
        print(f"  T={T}: {tokens_included} tokens")
```

**Output**:
```
Tokens included in top-p=0.9 nucleus at different temperatures:
--------------------------------------------------
  T=0.3: 8 tokens
  T=0.5: 14 tokens
  T=0.7: 19 tokens
  T=1.0: 29 tokens
  T=1.3: 41 tokens
```

**This interaction is why many practitioners recommend adjusting one parameter at a time.** The OpenAI documentation suggests setting one to its default and only tuning the other ([OpenAI, 2023](#openai2023)).

---

## Practical Experiments with Real APIs

Let's see these parameters in action with actual API calls. The following experiments use both OpenAI and Anthropic APIs to demonstrate behavior across providers.

### API Temperature Ranges Differ

OpenAI accepts temperature values from 0.0 to 2.0, while Anthropic's API restricts temperature to 0.0–1.0. This means temperature=1.0 represents "maximum creativity" for Claude, whereas GPT models can go twice as high. The experiments below account for this by scaling Anthropic's temperature values proportionally.

### OpenAI Python Sample

**Python:**
```python
# openai_generate.py

from openai import OpenAI
import os

# Initialize client
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def openai_generate(prompt, temperature=1.0, top_p=1.0, n=5):
    """Generate n completions with OpenAI."""
    responses = []
    for _ in range(n):
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=50
        )
        responses.append(response.choices[0].message.content)
    return responses

def main():
    # Experiment: Same prompt, different temperatures
    prompt = (
        "Complete this sentence creatively: "
        "The robot looked at the sunset and felt"
    )

    print("=" * 60)
    print("TEMPERATURE EXPERIMENT (OpenAI)")
    print("=" * 60)

    for temp in [0.0, 0.5, 1.0, 1.5]:
        print(f"\nTemperature = {temp}")
        print("-" * 40)
        responses = openai_generate(prompt, temperature=temp, n=3)
        for i, r in enumerate(responses, 1):
            print(f"  {i}. {r[:70]}...")
```

**Output:**

```
============================================================
TEMPERATURE EXPERIMENT (OpenAI)
============================================================

Temperature = 0.0
----------------------------------------
  1. a surge of colors dance within its circuits, as if the vibrant hues of...
  2. a surge of colors swirling within its circuits, as if the vibrant hues...
  3. a surge of electric wonder, as if the vibrant hues of orange and pink ...

Temperature = 0.5
----------------------------------------
  1. The robot looked at the sunset and felt a strange flicker of something...
  2. a strange flicker of longing, as if the vibrant hues of orange and pin...
  3. a surge of emotions it could not compute, as the vibrant hues of orang...

Temperature = 1.0
----------------------------------------
  1. The robot looked at the sunset and felt an unexpected surge of warmth ...
  2. a surge of electric nostalgia, as if the vibrant hues of orange and pu...
  3. a strange flicker of warmth in its circuits, as if the vibrant hues of...

Temperature = 1.5
----------------------------------------
  1. a strange resonance within its circuits, as if the fading hues of oran...
  2. The robot looked at the sunset and felt a surge of iridescent data cou...
  3. a wistful yearning, sparking echoes of distant memories encoded in its...
```

### Anthropic Python Sample

**Python:**
```python
# anthropic_generate.py

from anthropic import Anthropic
import os

# Initialize client
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def anthropic_generate(prompt, temperature=1.0, top_p=1.0, n=5):
    """Generate n completions with Anthropic.

    Note: Anthropic's API accepts temperature in range [0, 1],
    unlike OpenAI's [0, 2]. Values are clamped accordingly.
    """
    # Clamp temperature to Anthropic's valid range
    temperature = max(0.0, min(1.0, temperature))

    responses = []
    for _ in range(n):
        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            system = (
                "You are a creative writing assistant. When asked to "
                "complete a sentence, respond with ONLY the completion "
                "- no preamble, no alternatives, no explanation. Just "
                "continue the text naturally."
            ),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=50
        )
        responses.append(response.content[0].text)
    return responses


def main():
    # Experiment: Same prompt, different temperatures
    prompt = (
        "Complete this sentence with a single continuation: "
        "The robot looked at the sunset and felt"
    )

    print("=" * 60)
    print("TEMPERATURE EXPERIMENT (Anthropic Claude)")
    print("=" * 60)

    for temp in [0.0, 0.3, 0.7, 1.0]:
        print(f"\nTemperature = {temp}")
        print("-" * 40)
        responses = anthropic_generate(prompt, temperature=temp, n=3)
        for i, r in enumerate(responses, 1):
            print(f"  {i}. {r[:80]}...")
```

**Output:**
Note: Since Anthropic's temperature range is 0.0–1.0, we test at 0.0, 0.3, 0.7, and 1.0.

```
============================================================
TEMPERATURE EXPERIMENT (Anthropic Claude)
============================================================

Temperature = 0.0
----------------------------------------
  1. a strange, inexplicable longing for something it could not understand, a whisper...
  2. a strange, inexplicable longing for something it could not understand, a whisper...
  3. a strange, inexplicable longing for something it could not understand, a whisper...

Temperature = 0.3
----------------------------------------
  1. a strange, inexplicable longing for something it could not understand, a whisper...
  2. a strange, unexpected warmth spreading through its circuits, as if something bey...
  3. a strange, inexplicable longing for something it could not understand, a whisper...

Temperature = 0.7
----------------------------------------
  1. a strange, inexplicable longing for something it could not understand, a whisper...
  2. a strange, unexpected warmth spreading through its circuits, almost like what hu...
  3. a strange, unexpected warmth spreading through its circuitry, almost like what h...

Temperature = 1.0
----------------------------------------
  1. a strange warmth spreading through its circuits, wondering if this was what huma...
  2. a strange, unfamiliar warmth spreading through its circuits, as if something bey...
  3. a strange, inexplicable longing for something it could not name, a whisper of em...
```

**What to observe:**
- At temperature 0, responses should be nearly identical (deterministic)
- At temperature 0.5, minor variations appear but core structure persists
- At temperature 1.0, genuine creativity emerges
- At temperature 1.5+ (OpenAI only), outputs become increasingly unpredictable
- Anthropic's maximum (1.0) produces roughly comparable variety to OpenAI's mid-range (~0.7–0.8)

---

## When to Use Which: A Decision Framework
Based on both the mathematics and empirical testing, here's a practical guide:

### Use Low Temperature (0.1–0.3) When:
- **Factual retrieval**: "What year was the Treaty of Westphalia signed?"
- **Code generation**: Syntax errors become more likely at high temperatures
- **Classification tasks**: You want the model's highest-confidence answer
- **Structured output**: JSON, XML, or other formats where deviation breaks parsing

### Use Medium Temperature (0.5–0.8) When:
- **Professional writing**: Emails, reports, documentation
- **Summarization**: Faithful but not robotic
- **Translation**: Accuracy matters but natural phrasing helps
- **Explanations**: Clear and engaging without hallucination risk

### Use High Temperature (0.9–1.0 for Anthropic, 0.9–1.5 for OpenAI) When:
- **Creative writing**: Fiction, poetry, brainstorming
- **Ideation**: Generating diverse options to choose from
- **Dialogue**: Conversational responses that feel natural
- **Exploration**: When you want to see what's possible

### Top-P Guidelines:
- **Start with 0.9**: A sensible default for most tasks
- **Reduce to 0.5–0.7**: For more focused outputs when using higher temperatures
- **Keep at 1.0**: When you want temperature to be the only control
- **Never use 0**: This would include zero tokens

---

## The Temperature 0 Myth

A common misconception: "Temperature 0 is deterministic." In theory, yes. The {{< term "argmax" "argmax" >}} of the logits always wins. In practice, {{< term "floating-point" "floating-point arithmetic" >}} introduces subtle variations, and different implementations handle the edge case differently ([Peng et al., 2023](#peng2023)).

Some providers implement "almost zero" (like 1e-8) rather than true zero. Some use a separate greedy decoding path. OpenAI's API accepts temperature=0 but may still show occasional variation in long outputs.

```python
# test_determinism.py

from openai_generate import openai_generate

def test_determinism(prompt, n_trials=10):
    """Test whether temperature=0 produces identical outputs."""
    responses = openai_generate(prompt, temperature=0, n=n_trials)
    unique_responses = set(responses)

    print(f"Unique responses at T=0: {len(unique_responses)} / {n_trials}")

    if len(unique_responses) == 1:
        print("Deterministic behavior confirmed!")
        print(f"  Response: {responses[0][:80]}...")
    else:
        print("Non-determinism detected!")
        for r in unique_responses:
            print(f"  - {r[:60]}...")


if __name__ == "__main__":
    # Simple prompt - likely deterministic
    print("Test 1: Simple math question")
    print("-" * 40)
    test_determinism("What is 2 + 2?")

    # Longer, more creative prompt - more likely to show variation
    print("\nTest 2: Creative prompt (longer output)")
    print("-" * 40)
    test_determinism(
        "Write a paragraph about a robot discovering emotions.",
        n_trials=5
    )
```

**Output:**
```
Test 1: Simple math question
----------------------------------------
Unique responses at T=0: 1 / 10
Deterministic behavior confirmed!
  Response: 2 + 2 equals 4....

Test 2: Creative prompt (longer output)
----------------------------------------
Unique responses at T=0: 2 / 5
Non-determinism detected!
  - In a dimly lit laboratory, a robot named AURA, designed for ...
  - In a dimly lit laboratory, a robot named AURA, designed for ...
```

For truly deterministic behavior, some APIs offer a separate `{{< term "random-seed" "seed" >}}` parameter. Always check your provider's documentation.

---

## Closing Thoughts

Temperature and top-p aren't magic. They're straightforward mathematical transformations with predictable effects. Temperature exponentially reshapes the probability distribution; top-p dynamically truncates it. Together, they give you fine-grained control over the {{< term "exploration-exploitation" "exploration-exploitation tradeoff" >}} that underlies all language generation.

The key insight is that these parameters don't change what the model knows. They change how the model decides. A low-temperature model isn't smarter; it's more committed to its first instinct. A high-temperature model isn't more creative; it's more willing to take risks.

Understanding this distinction helps you debug unexpected outputs. If your model keeps repeating itself, temperature might be too low. If it's generating nonsense, temperature might be too high. If it's ignoring plausible alternatives, top-p might be too restrictive. The mathematics tells you exactly where to look.

---

## Try It Yourself

[Download the full code on GitHub](https://github.com/TomArcher/technical-blog-examples/tree/main/temperature-top-p-creativity-knobs)

---
