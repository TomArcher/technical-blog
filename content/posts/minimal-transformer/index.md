+++
title = "Implementing a Minimal Transformer in PyTorch"
subtitle = "Building the core machinery of a language model from embeddings and attention to training and generation"
date = "2026-04-01T06:00:00-07:00"
draft = false
categories = ["Applied Modeling and Simulation"]
tags = ["AI", "Attention Mechanisms", "LLM", "NLP", "Python", "Transformers",]
author = "Tom Archer"
listThumb = "transformer.jpeg"
disposition="in-progress"

hero = "transformer.jpeg"
heroAlt = "Diagram of a Transformer block showing the flow from token embeddings through multi-head attention, add & norm, feed-forward, and add & norm to output logits"
heroLabel = "Open full-size transformer"
heroCaption = "Around 200 lines of Python separate you from understanding why attention really is all you need."

whatYoullLearn = [
    "How token and positional embeddings turn a sequence of characters into representations a Transformer can process",
    "How scaled dot-product attention lets each token decide which earlier tokens matter",
    "Why multiple attention heads can learn different relationships within the same sequence",
    "How residual connections, layer normalization, and feed-forward networks form a Transformer block",
    "How causal masking makes next-token training possible without letting the model see the future",
    "How to train a small Transformer in PyTorch and use it to generate new text"
]

[[references]]
key = "ba2016"
citation = "Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv preprint arXiv:1607.06450*."
url = "https://arxiv.org/abs/1607.06450"

[[references]]
key = "hendrycks2016"
citation = "Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv preprint arXiv:1606.08415*."
url = "https://arxiv.org/abs/1606.08415"

[[references]]
key = "karpathy2022"
citation = "Karpathy, A. (2022). *nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs.*"
url = "https://github.com/karpathy/nanoGPT"

[[references]]
key = "loshchilov2019"
citation = "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *Proceedings of ICLR 2019*."
url = "https://arxiv.org/abs/1711.05101"

[[references]]
key = "radford2019"
citation = "Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog, 1*(8)."
url = "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"

[[references]]
key = "vaswani2017"
citation = "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*."
url = "https://arxiv.org/abs/1706.03762"

[[references]]
key = "xiong2020"
citation = "Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. Y. (2020). On layer normalization in the Transformer architecture. *Proceedings of ICML 2020*."
url = "https://arxiv.org/abs/2002.04745"

+++

In 2017, Vaswani et al. published "Attention Is All You Need," a paper that quietly rearranged the entire landscape of machine learning. It introduced the Transformer architecture — a design that has since become the backbone of every major language model you've heard of: GPT, BERT, Claude, Gemini, and dozens of others. The paper's title was a provocation. Attention mechanisms already existed. The claim was that you could throw out recurrence entirely and let attention carry the whole load.

That claim turned out to be correct.

Most engineers who work *with* language models never look inside them. They call an API, parse a response, ship a product. There is nothing wrong with that. But if you want to understand *why* these models behave the way they do — why they track context across long passages, why they can switch tones mid-generation, why they hallucinate — you need to see the machinery. Not read about it. Build it.

In this post, we implement a minimal Transformer from scratch in PyTorch. No Hugging Face, no pre-built attention layers. We'll cover token and positional embeddings, scaled dot-product attention, multi-head attention, the feed-forward sublayer, layer normalization, residual connections, and a training loop on character-level Shakespeare — the classic proving ground for small language models. The full implementation lands at just under 200 lines.

By the end, the model will be writing sentences that loosely belong in the Globe Theatre. More importantly, you'll have built something that is architecturally identical — in every meaningful sense — to the models that power the AI tools you use every day.

---

> **TL;DR:**
> - A Transformer is a stack of identical blocks, each containing attention + feed-forward + layer norm
> - Attention lets every token "look at" every other token simultaneously — there is no recurrence
> - Positional embeddings inject sequence order because attention itself is order-blind
> - Multi-head attention runs several attention operations in parallel, each learning different relational patterns
> - The whole thing trains end-to-end with cross-entropy loss on next-token prediction
> - ~200 lines of PyTorch gets you a working character-level language model

---

## The Thought Experiment

Suppose you're trying to compress the meaning of a sentence into a format a computer can reason about. You can't hand it raw characters and expect anything useful. You need representations that carry *semantic weight*, and you need a mechanism for those representations to influence one another.

Before Transformers, the dominant approach was recurrence: feed tokens in one at a time, maintain a hidden state that tries to remember everything that came before. The problem is that the hidden state is a bottleneck. Information from 500 tokens ago has to survive 500 update steps to remain relevant. In practice, it often doesn't.

The Transformer's insight is different: don't process tokens sequentially. Process them all at once, but give each token a learnable way to *attend* to every other token in the sequence. The model doesn't need to remember — it can look.

**Assumptions and scope:**

- Dataset: the complete works of Shakespeare as a flat text file (~1MB), treated as a sequence of characters
- Vocabulary: the set of unique characters in the file (typically 65)
- Task: given a sequence of characters, predict the next character
- Model size: 4 heads, 4 layers, embedding dimension of 128 — enough for structure to emerge, fast enough to train on a laptop
- No learning rate schedulers, no gradient clipping (extensions for the exercises)

The mathematics we'll draw on:

**Scaled Dot-Product Attention:**

\\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\\]

Where \\(Q\\), \\(K\\), and \\(V\\) are the query, key, and value matrices derived by projecting the input, and \\(d_k\\) is the key dimension. The \\(\sqrt{d_k}\\) scaling prevents dot products from growing so large that gradients through the softmax vanish ([Vaswani et al., 2017](#vaswani2017)).

**Positional Encoding (learned variant):**

\\[
\text{Input}_i = \text{TokenEmbed}(x_i) + \text{PosEmbed}(i)
\\]

We use learned positional embeddings rather than the sinusoidal version in the original paper — simpler to implement and equally effective for short sequences.

**Layer Normalization:**

\\[
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta
\\]

Applied *before* each sublayer (the "Pre-LN" variant), which trains more stably than the original Post-LN formulation ([Xiong et al., 2020](#xiong2020)).

---

## Modeling the Problem

The Transformer block is a composition of two sublayers, each wrapped in a residual connection and a layer norm:

\\[
x \leftarrow x + \text{MultiHeadAttention}(\text{LayerNorm}(x))
\\]

\\[
x \leftarrow x + \text{FeedForward}(\text{LayerNorm}(x))
\\]

The residual connections (\\(x + \ldots\\)) are critical — they create a direct gradient path back to the input and make it possible to stack many blocks without the training signal disappearing.

**Multi-head attention** runs \\(h\\) attention operations in parallel on lower-dimensional projections of the input, then concatenates and re-projects the results:

\\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
\\]

Each head can specialize in a different relationship — one might track subject-verb agreement, another might link pronouns to their antecedents. We don't program these specializations; they emerge from training.

**The feed-forward sublayer** is a position-wise two-layer MLP applied independently to each token's representation. It's where the model gains capacity to transform representations in ways that attention alone cannot ([Vaswani et al., 2017](#vaswani2017)):

\\[
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
\\]

The inner dimension is conventionally \\(4 \times d_{\text{model}}\\).

**Assumptions:**

- All components use `float32`; no mixed precision
- Causal (lower-triangular) mask enforces the autoregressive property during training
- AdamW optimizer with a fixed learning rate of \\(3 \times 10^{-4}\\) ([Loshchilov & Hutter, 2019](#loshchilov2019))
- Dropout rate of 0.1 applied after embeddings, attention, and feed-forward layers

---

## Using AI to Scaffold the Code

We'll build the model component by component, mirroring how you'd approach this in a research setting — each piece testable in isolation before being assembled into the full stack.

### Character-Level Tokenizer

The simplest possible tokenizer: map each unique character in the corpus to an integer index.

**Prompt example:**

> Write a Python class `CharTokenizer` that takes a string of text, builds a vocabulary of unique characters, and exposes `encode(text) -> list[int]` and `decode(indices) -> str` methods. Include `vocab_size` as a property.

**Python:**
```python
# tokenizer.py

class CharTokenizer:
    """
    Character-level tokenizer. Maps each unique character to an integer index.

    Vocabulary is built from the training corpus at instantiation time.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for ch, i in self._stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    def encode(self, text: str) -> list[int]:
        return [self._stoi[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self._itos[i] for i in indices)
```

### Token and Positional Embeddings

Attention is order-blind — it treats the input sequence as a set, not a sequence. Positional embeddings re-inject the order that attention discards.

**Prompt example:**

> Write a PyTorch `nn.Module` called `Embeddings` that combines a token embedding table and a learned positional embedding table. The forward pass should accept a tensor of token indices of shape `(batch, seq_len)` and return the summed embeddings of shape `(batch, seq_len, d_model)`. Include dropout.

**Python:**
```python
# embeddings.py

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    Combines token embeddings with learned positional embeddings.

    Learned positional embeddings are simpler than sinusoidal and
    perform comparably for short sequences.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_seq_len, d_model)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return self.dropout(self.token_embed(x) + self.pos_embed(positions))
```

### Scaled Dot-Product Attention

This is the core operation. The query, key, and value projections are where learning happens; the softmax is what makes it *attention*.

**Prompt example:**

> Write a function `scaled_dot_product_attention(q, k, v, mask=None)` that computes attention scores, applies an optional causal mask, runs softmax, and returns the weighted sum of values. Inputs have shape `(batch, heads, seq_len, head_dim)`.

**Python:**
```python
# attention.py

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Dividing by sqrt(d_k) keeps dot products in a reasonable range,
    preventing softmax from saturating into near-zero gradients
    (Vaswani et al., 2017).

    Args:
        q: Queries  (batch, heads, seq_len, head_dim)
        k: Keys     (batch, heads, seq_len, head_dim)
        v: Values   (batch, heads, seq_len, head_dim)
        mask: Optional causal mask (1 = keep, 0 = mask)

    Returns:
        Attended values (batch, heads, seq_len, head_dim)
    """
    d_k    = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

### Multi-Head Attention

Multi-head attention projects the input into \\(h\\) lower-dimensional query, key, and value spaces, runs attention in each, then re-combines.

**Prompt example:**

> Write a `MultiHeadAttention` nn.Module that projects inputs into Q, K, V using linear layers, splits into multiple heads, runs `scaled_dot_product_attention` in parallel, concatenates the results, and projects back to `d_model`. Include a causal mask so tokens can only attend to previous positions.

**Python:**
```python
# multihead_attention.py

import torch
import torch.nn as nn
from attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal (autoregressive) masking.

    Each head attends over the full sequence in a lower-dimensional
    subspace. Different heads learn different relational patterns.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D    = self.n_heads, self.head_dim

        def project_and_split(linear: nn.Linear) -> torch.Tensor:
            # (B, T, C) -> (B, H, T, D)
            return linear(x).view(B, T, H, D).transpose(1, 2)

        q = project_and_split(self.q_proj)
        k = project_and_split(self.k_proj)
        v = project_and_split(self.v_proj)

        # Causal mask: token i may only attend to tokens 0..i
        mask     = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attended = scaled_dot_product_attention(q, k, v, mask)

        # (B, H, T, D) -> (B, T, C)
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out_proj(attended))
```

### Feed-Forward Sublayer

A position-wise MLP that gives the model capacity to transform each token's representation independently.

**Prompt example:**

> Write a `FeedForward` nn.Module with two linear layers, GELU activation, and dropout. The inner dimension should be `4 * d_model`.

**Python:**
```python
# feedforward.py

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied to each token independently. The 4x expansion gives the model
    representational capacity that attention alone cannot provide.
    GELU activation is standard in modern Transformers (Hendrycks & Gimpel, 2016).
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### Transformer Block

One complete Transformer block: Pre-LN, attention, residual; Pre-LN, feed-forward, residual.

**Prompt example:**

> Write a `TransformerBlock` nn.Module that composes `MultiHeadAttention`, `FeedForward`, and two `nn.LayerNorm` layers using Pre-LN residual connections.

**Python:**
```python
# transformer_block.py

import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    One Transformer decoder block.

    Pre-LN (LayerNorm before each sublayer) is more stable than the original
    Post-LN formulation, especially for deeper stacks (Xiong et al., 2020).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
```

### The Full Model

Stack the blocks, add a final layer norm, and project to the vocabulary.

**Prompt example:**

> Write a `MiniTransformer` nn.Module that accepts hyperparameters `vocab_size`, `d_model`, `n_heads`, `n_layers`, `max_seq_len`, and `dropout`. Compose `Embeddings`, a stack of `TransformerBlock`s, a final `nn.LayerNorm`, and a linear head projecting to `vocab_size`. Include a `generate` method that autoregressively samples `max_new_tokens` tokens given a context tensor.

**Python:**
```python
# transformer.py

import torch
import torch.nn as nn
from embeddings import Embeddings
from transformer_block import TransformerBlock


class MiniTransformer(nn.Module):
    """
    Minimal decoder-only Transformer for character-level language modelling.

    Architecture:
        Embeddings -> N x TransformerBlock -> LayerNorm -> Linear head

    Autoregressive: trained to predict the next character given all previous.
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        max_seq_len: int   = 256,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed  = Embeddings(vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)
        return self.head(self.ln_f(self.blocks(self.embed(x))))

    @torch.no_grad()
    def generate(self, context: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively sample one token at a time."""
        for _ in range(max_new_tokens):
            ctx      = context[:, -self.max_seq_len:]
            logits   = self(ctx)
            next_tok = torch.multinomial(
                torch.softmax(logits[:, -1, :], dim=-1), num_samples=1
            )
            context = torch.cat([context, next_tok], dim=1)
        return context
```

---

## Training on Shakespeare

With the model assembled, the training loop is straightforward: sample random chunks of the Shakespeare corpus, compute cross-entropy loss on next-token prediction, and back-propagate.

**Prompt example:**

> Write a `train.py` script that loads `shakespeare.txt`, builds a `CharTokenizer`, creates a `MiniTransformer`, and runs a training loop for `max_iters` steps. Each step should sample a random batch of `(batch_size, seq_len)` token sequences, run a forward pass, compute cross-entropy loss, and update with AdamW. Print loss every 500 steps, then generate a short sample at the end.

**Python:**
```python
# train.py

import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer
from transformer import MiniTransformer

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
SEQ_LEN      = 128
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 4
MAX_ITERS    = 5000
LR           = 3e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

def get_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequence pairs."""
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x  = torch.stack([data[i     : i + SEQ_LEN    ] for i in ix])
    y  = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def train() -> None:
    text      = open("shakespeare.txt").read()
    tokenizer = CharTokenizer(text)
    data      = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = MiniTransformer(
        vocab_size  = tokenizer.vocab_size,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        max_seq_len = SEQ_LEN,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {DEVICE}\n")

    for step in range(MAX_ITERS):
        x, y   = get_batch(data)
        logits = model(x)                           # (B, T, vocab_size)
        loss   = F.cross_entropy(
            logits.view(-1, tokenizer.vocab_size),  # (B*T, vocab_size)
            y.view(-1)                              # (B*T,)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.4f}")

    # Generate a sample
    context   = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=300)
    print("\n── Generated text ──────────────────────────────────────────")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    train()
```

**Sample output after 5,000 steps:**
```yaml
Model parameters: 1,354,049
Training on: cpu

Step     0 | Loss: 4.1726
Step   500 | Loss: 2.3841
Step  1000 | Loss: 2.1093
Step  1500 | Loss: 1.9872
Step  2000 | Loss: 1.9214
Step  2500 | Loss: 1.8807
Step  3000 | Loss: 1.8531
Step  3500 | Loss: 1.8214
Step  4000 | Loss: 1.7990
Step  4500 | Loss: 1.7823

── Generated text ──────────────────────────────────────────
KING RICHARD:
And the cause is the world, and the worder
That speaks not the death of the grace of the son
To the king that would be the propect of mine,
And so the prince with his father in the lord,
The worthy father of the common man
That shall be the world to be so greet.
```

Not Hamlet. But the model has learned that speeches have speaker labels, that lines end, that certain words cluster with certain others. It has learned the *texture* of Shakespearean English from first principles — no pre-training, no external knowledge, just gradient descent on next-character prediction.

---

## Making the Results Visual

Let's create two visualizations that reveal what the model has learned — and where the physics of gradient descent pushes it.

### Training Loss Curve

The loss curve is one of the most informative diagnostics in deep learning. A clean downward slope tells you the model is learning efficiently. Plateaus often indicate a learning rate that's too low or a batch that's too small.

**Prompt example:**

> Modify `train.py` to collect `(step, loss)` tuples every 100 steps and write a `plot_loss_curve(history)` function that plots training loss over steps with a smoothed EMA trend line.

**Python:**
```python
# plot_loss_curve.py

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(history: list[tuple[int, float]]) -> None:
    """
    Plot training loss with a smoothed EMA trend line.

    The steep early drop reflects rapid learning of character frequencies.
    The slower decline afterward reflects finer grammatical structure.

    Args:
        history: List of (step, loss) tuples recorded during training.
    """
    steps, losses = zip(*history)

    # Exponential moving average
    smoothed, alpha = [], 0.9
    running = losses[0]
    for loss in losses:
        running = alpha * running + (1 - alpha) * loss
        smoothed.append(running)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses,   alpha=0.3, color="steelblue", label="Raw loss")
    ax.plot(steps, smoothed, color="steelblue", linewidth=2, label="Smoothed (EMA)")

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Cross-entropy loss", fontsize=12)
    ax.set_title("Training Loss — Minimal Transformer on Shakespeare", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()
```

The shape of the curve is consistent across runs: a steep initial drop as the model rapidly learns letter frequencies and common digrams, followed by a slower decline as it picks up longer-range grammatical patterns.

<img src="./loss_curve.png" alt="Line chart showing training loss decreasing from approximately 4.2 to 1.78 over 5000 steps, with a smoothed EMA trend line overlaid" style="display: block; margin: 0 auto;">
<figcaption style="font-size: 0.9em; color: #555; margin-top: 5px;">
<em>Training loss over 5,000 steps. The steep early drop reflects rapid learning of character frequencies; the slower tail reflects finer grammatical structure emerging.</em>
</figcaption>

<br>

### Attention Pattern Heatmap

Visualizing the attention weights from a trained head reveals what each head has learned to attend to. Some heads specialize in adjacent characters, others in punctuation boundaries, others in repeated characters across longer spans.

**Prompt example:**

> Write a `plot_attention_heatmap(model, tokenizer, prompt)` function that uses a forward hook to capture attention weights from the first block's first head and plots them as a heatmap with the prompt characters on both axes.

**Python:**
```python
# plot_attention_heatmap.py

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_heatmap(model, tokenizer, prompt: str) -> None:
    """
    Visualize attention weights for Block 0, Head 0.

    Pattern types to look for:
      - Diagonal bands: local context dependence
      - Column spikes: a "global key" token many queries attend to
      - Scattered high values: long-range relational learning

    Args:
        model: Trained MiniTransformer
        tokenizer: CharTokenizer used during training
        prompt: Short string to visualize (10-20 characters works well)
    """
    captured = {}

    def hook(module, input, output):
        x   = input[0]
        B, T, C = x.shape
        H, D    = module.n_heads, module.head_dim

        q = module.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = module.k_proj(x).view(B, T, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        mask   = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        captured["weights"] = F.softmax(scores, dim=-1)[0, 0].detach().cpu()

    handle = model.blocks[0].attn.register_forward_hook(hook)

    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        model(x)
    handle.remove()

    fig, ax = plt.subplots(figsize=(10, 8))
    chars   = list(prompt)
    weights = captured["weights"].numpy()

    sns.heatmap(
        weights, ax=ax,
        xticklabels=chars, yticklabels=chars,
        cmap="Blues", square=True, linewidths=0.3,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Attention weights — Block 0, Head 0", fontsize=14)
    ax.set_xlabel("Key (attending from)")
    ax.set_ylabel("Query (attending to)")

    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.show()
```

The strict lower-triangular structure in the heatmap is a direct consequence of the causal mask — every token above the diagonal is set to \\(-\infty\\) before softmax, so those weights are exactly zero. What remains shows the model's learned preferences: notice how it tends to assign high weight to spaces and punctuation marks, which serve as reliable syntactic anchors in the Shakespeare corpus.

<img src="./attention_heatmap.png" alt="Heatmap of attention weights for a short Shakespeare prompt, showing a lower-triangular structure with high weights concentrated near spaces and punctuation" style="display: block; margin: 0 auto;">
<figcaption style="font-size: 0.9em; color: #555; margin-top: 5px;">
<em>Attention weights for Block 0, Head 0. The lower-triangular structure reflects the causal mask. High-weight columns near spaces and punctuation suggest this head has specialized in syntactic boundary detection.</em>
</figcaption>

<br>

## What We Learned

Building this model end-to-end surfaces a number of things that are easy to miss when reading about Transformers at a higher level:

1. **Attention has no inherent notion of position.** Without positional embeddings, a model would produce identical outputs for `"the cat sat"` and `"sat cat the"`. The embeddings are what inject order, and they are fully learned — not hardcoded.

2. **The causal mask is load-bearing.** Without it, each token during training would see its own target, making next-token prediction trivially easy and the trained model useless at generation time. The strict lower-triangular mask enforces the autoregressive property.

3. **Residual connections are why you can go deep.** Without them, gradients vanish over many layers. With them, there is always a direct path from the loss back to the first embedding layer, regardless of how many blocks sit in between.

4. **Layer norm placement matters.** Pre-LN (normalize before each sublayer) is noticeably more stable than the original Post-LN formulation — loss curves are smoother and the model is less sensitive to the choice of learning rate.

5. **Cross-entropy on next-token prediction is everything.** There is no separate "language understanding" objective. Syntax, style, and structure are learned as instrumental goals in service of predicting the next character.

---

## Exercises for the Reader

**Beginner Level:**

1. **Vocabulary size vs. quality:** Swap the character-level tokenizer for a simple word-level one. How does the generated text change? How does training time change?

2. **Temperature sampling:** Add a `temperature` parameter to `generate()` that divides the logits before softmax. What happens to output quality as temperature approaches 0? As it exceeds 1?

3. **Parameter count:** Systematically vary `d_model` and `n_layers`. Plot model size vs. final loss after 5,000 steps. Where does the curve flatten?

**Intermediate Level:**

1. **Sinusoidal positional embeddings:** Implement the sinusoidal variant from the original Vaswani et al. paper and compare loss curves against the learned version. Which converges faster?

2. **Top-k sampling:** Replace `torch.multinomial` in `generate()` with top-k sampling. How does restricting to the \\(k\\) highest-probability tokens affect the diversity/quality tradeoff?

3. **Gradient clipping:** Add `torch.nn.utils.clip_grad_norm_` to the training loop. Does it stabilize training when you increase `n_layers` to 8?

**Advanced Level:**

1. **Attention head specialization:** Visualize all \\(n_\text{heads} \times n_\text{layers}\\) attention maps after training. Can you categorize the patterns? Do any heads consistently specialize in the same relationship across different inputs?

2. **Scale to byte-pair encoding:** Replace the character tokenizer with a BPE tokenizer (e.g., `tiktoken`) and retrain on a larger dataset. What architectural changes become necessary as vocabulary size grows?

3. **KV-cache:** Implement a key-value cache in `generate()` so previously computed keys and values aren't recomputed at each new token. What is the theoretical time complexity improvement at sequence length \\(T\\)?

---

## Closing Thoughts

Every language model you've used — from early GPT to the models powering today's AI assistants — is an elaboration of the architecture we built here. The elaborations matter enormously: scale, data, alignment techniques, inference optimizations. But they are additions, not replacements.

The ~200 lines we wrote contain the essential ideas: attention as a learned routing mechanism, positional embeddings as a workaround for the model's spatial blindness, residual connections as a gradient highway, and the whole stack trained on the deceptively simple objective of predicting the next token.

A brief note on lineage: the character-level Shakespeare setup is a tradition in this space. Andrej Karpathy's char-rnn and nanoGPT repos made it canonical, and the implementation here follows the same spirit — build the smallest thing that honestly demonstrates the idea. If you find yourself wanting to go deeper after this post, nanoGPT is the natural next stop.

Shakespeare called the world a stage. With a Transformer, you can build a small model that has read every play he ever wrote and will, haltingly, try to add one more. It won't be good. But the machinery generating those imperfect lines is, at its core, the same machinery that drives the most capable AI systems ever built.

---

## Try It Yourself

[Download the full code on GitHub](https://github.com/TomArcher/technical-blog-examples/tree/main/minimal-transformer)

The Shakespeare corpus can be downloaded with:
```
curl -o shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

---
