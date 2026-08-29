+++
author = "Tom Archer"

date = '2025-11-10T06:00:00-07:00'
draft = false

title = "How Large Language Models (LLMs) Handle Context Windows: The Memory That Isn't Memory"
subtitle = "Exploring why longer context doesn't mean better memory and what happens when conversations grow"

learningPaths = [
    "AI for Software Engineers",
    "How Large Language Models Work"
]

categories = ["AI and the Mathematics of Language"]
tags = ["AI", "attention mechanisms", "context windows", "LLM", "transformers"]

listThumb = "how-llms-handle-context-windows.png"

hero = "how-llms-handle-context-windows.png"
heroAlt = "Digital artwork showing a conversation thread fading into the distance, with attention weights visualized as glowing connections"
heroLabel = "Open full-size context windows illustration"
heroCaption = "Context windows create the illusion of memory through mathematical attention, not storage."

whatYoullLearn = [
    "Why an LLM's context window is not the same thing as memory",
    "How chat applications create continuity even though the underlying model is stateless",
    "How attention lets earlier parts of a conversation influence the next token",
    "Why longer conversations become increasingly expensive for a transformer to process",
    "Why information can become harder to use even while it remains inside the context window",
    "How truncation, summarization, retrieval, and KV caching help manage long conversations"
]

[[references]]
key = "alammar2018"
citation = "Alammar, J. (2018). The illustrated transformer."
url = "http://jalammar.github.io/illustrated-transformer/"

[[references]]
key = "dai2019"
citation = "Dai, Z., Yang, Z., Yang, Y., et al. (2019). Transformer-XL: Attentive language models beyond a fixed-length context. *Proceedings of ACL 2019*."
url = "https://arxiv.org/abs/1901.02860"

[[references]]
key = "gu2023"
citation = "Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint*."
url = "https://arxiv.org/abs/2312.00752"

[[references]]
key = "jiang2023"
citation = "Jiang, Z., et al. (2023). LLMLingua: Compressing prompts for accelerated inference of large language models. *arXiv preprint*."
url = "https://arxiv.org/abs/2310.05736"

[[references]]
key = "liu2023"
citation = "Liu, N. F., Lin, K., Hewitt, J., et al. (2023). Lost in the middle: How language models use long contexts. *arXiv preprint*."
url = "https://arxiv.org/abs/2307.03172"

[[references]]
key = "peng2023"
citation = "Peng, B., Alcaide, E., Anthony, Q., et al. (2023). RWKV: Reinventing RNNs for the transformer era. *arXiv preprint*."
url = "https://arxiv.org/abs/2305.13048"

[[references]]
key = "shaw2018"
citation = "Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. *Proceedings of NAACL-HLT 2018*."
url = "https://arxiv.org/abs/1803.02155"

[[references]]
key = "tay2022"
citation = "Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2022). Efficient transformers: A survey. *ACM Computing Surveys, 55*(6), 1-28."
url = "https://dl.acm.org/doi/10.1145/3530811"

[[references]]
key = "vaswani2017"
citation = "Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*."
url = "https://arxiv.org/abs/1706.03762"

+++

When you have a long conversation with a {{< term "large-language-model" "large language model" >}} (LLM) such as [ChatGPT](https://chatgpt.com/) or [Claude](https://claude.ai/new), it feels like the model remembers everything you've discussed. It references earlier points, maintains consistent context, and seems to "know" what you talked about pages ago.

But here's the uncomfortable truth: the model doesn't remember anything. It's not storing your conversation in memory the way a database would. Instead, it's **rereading the entire conversation from the beginning every single time you send a message.**

This post explores what context windows actually are, how they work mathematically, why they have limits, and what happens when those limits are reached. For a foundation in how models represent meaning, start with [How LLMs Think: Turning Meaning into Math](/posts/how-ai-turns-meaning-into-math/).

---

## The API vs. Web Interface: Where "Memory" Really Lives

Before diving into context windows, it's crucial to understand a fundamental architectural distinction that most users never see.

### **The Raw API Has No Memory**

When you call an LLM API directly, each request is completely independent:

```python
# Call 1
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "What country is Paris in?"
        }
    ]
)
# Response: "France"

# Call 2 (completely separate request)
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "Repeat the last country name"
        }
    ]
)
# Response: "I don't have context about a previous country name..."
```

The API is **{{< term "stateless" "stateless" >}}**. The model truly has no memory between calls. Each request starts from zero.

### **Web Interfaces Create the Illusion**

The ChatGPT or Claude **web interface** maintains conversation continuity by:

1. **Storing the conversation history** in the application layer (not in the model)
2. **Sending the entire history** with each new message
3. **Managing what fits** within the context window

Here's what actually happens behind the scenes:

```python
# Your first message
conversation = [
    {
        "role": "user",
        "content": "What country is Paris in?"
    }
]
response = api_call(conversation)
# Model responds: "France"

# Application stores the response
conversation.append(
    {
        "role": "assistant",
        "content": "France"
    }
)

# Your second message
conversation.append(
    {
        "role": "user",
        "content": "Repeat the last country"
    }
)

# The web interface sends ALL of this to the API:
response = api_call(conversation)
# [
#   {"role": "user", "content": "What country is Paris in?"},
#   {"role": "assistant", "content": "France"},
#   {"role": "user", "content": "Repeat the last country"}
# ]

# Now the model can answer "France" because it sees
# the full conversation

```

**The key insight:** The model itself stores nothing. The application manages conversation history and decides what to include in each API request. The context window is simply the maximum amount of conversation history that **can** be sent at once.

Think of it like talking to someone with amnesia, but you're reading them their diary first. They don't remember yesterday. They're just responding to what you showed them right now.

---

## What Is a Context Window?

A context window is **how many tokens the model can process at once**. Think of it as the model's field of vision where everything within that window can influence the next token it predicts.

Modern models have impressive context windows:

* **GPT-4**: 128,000 tokens (~96,000 words)
* **Claude 3**: 200,000 tokens (~150,000 words)
* **Gemini 1.5**: 1,000,000 tokens (~750,000 words)

These numbers sound enormous. An entire novel fits easily. But bigger isn't always better, and the way models "use" this space reveals fundamental constraints.

---

## The Illusion of Memory

Now that you understand the architecture, here's what actually happens during a web interface conversation:

**Turn 1:** You send a message. The application sends it to the API. The model processes it and responds.

**Turn 2:** You send another message. The application takes **your first message + the model's first response + your second message** and sends all of it to the API. The model processes everything to generate its second response.

**Turn 3:** The application sends **everything from turns 1 and 2 plus your new message** to the API. The model processes it all.

By turn 10, the model is processing thousands of tokens every single time, rereading the entire conversation thread from scratch. It's not looking up stored facts. It's **recomputing attention over everything you've ever said in that session**.

This is why "memory" in LLMs is an illusion. The model performs pattern matching across the visible conversation, not retrieval from storage. The web interface maintains the conversation history, but the model itself remembers nothing.

---

## How Attention Creates the Illusion

The mechanism that makes this work is the **{{< term "attention" "attention mechanism" >}}**, specifically **{{< term "self-attention" "self-attention" >}}**, as defined in the foundational paper "Attention Is All You Need" ([Vaswani et al., 2017](#vaswani2017)). When processing token \\(i\\) in the sequence, the model computes how much attention to pay to every previous token \\(j\\).

The attention score between two tokens is computed as:

\\[
\text{Attention}(Q, K, V) = \text{{{< term "softmax" "softmax" >}}}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\\]

Where:
- \\(Q\\) = {{< term "query" "query" >}} (what the current token is looking for)
- \\(K\\) = {{< term "key" "keys" >}} (what each previous token can offer)
- \\(V\\) = {{< term "value" "values" >}} (the actual content of those tokens)
- \\(d_k\\) = dimensionality of the key vectors (scaling factor)

For every new token the model generates, it computes attention across **every token in the context window**. If your conversation is 10,000 tokens long, that's 10,000 attention calculations per new token. For a visual explanation of how attention mechanisms work in practice, see [Alammar (2018)](#alammar2018).

---

## The Quadratic Cost Problem

Here's where the math gets brutal. As shown in the original Transformer paper ([Vaswani et al., 2017](#vaswani2017)), attention scales **quadratically** with sequence length.

If you double the conversation length:
- The model has to compute \\(2n\\) queries
- Against \\(2n\\) keys
- Resulting in \\((2n)^2 = 4n^2\\) operations

**Doubling the context requires 4× the computation.**

This is why context windows have hard limits. It's not just storage. It's computational cost. Processing a 128k token conversation requires computing roughly **16 billion attention scores** for a single response token.

<div style="float: right; width: 40%; margin: 0 0 1em 1em; padding: 0.5em; background-color: #f8f8f8; border: 1px solid #ddd; font-size: 0.9em;">
    <div style="text-align: center;"><strong>Why not just store previous computations?</strong><br><br></div>
    The model could cache some calculations, and modern implementations do use a <strong>{{< term "kv-cache" "KV cache" >}}</strong> (key-value cache) to avoid recomputing keys and values for tokens that haven't changed. But this only helps with computational cost. It doesn't solve the fundamental attention problem. The model still has to compute attention weights across the entire cached context for every new token.
</div>

---

## What Gets Lost: The Attention Dilution Problem

Even within the context window, not all tokens are treated equally. The *softmax* function in the attention mechanism creates a **{{< term "probability-distribution" "probability distribution" >}}** over all previous tokens. 

As the context grows:
- Attention gets **spread thinner** across more tokens
- Older tokens receive **exponentially less attention**
- The model effectively "forgets" details from early in the conversation

Think of it like a spotlight with fixed brightness. In a small room, everything is illuminated clearly. In a stadium, that same spotlight becomes a dim glow barely reaching the far corners.

This is why models often lose track of instructions or details mentioned 50,000 tokens ago, even though those tokens are technically still in the context window being sent by the application.

---

## Why This Isn't Like Human Memory

It's tempting to compare LLM context to human memory. After all, both seem to "remember" past conversations. But the mechanisms are fundamentally different.

**Human memory consolidates and abstracts.** You might forget the details of an argument with a friend from years ago, but you remember the emotional aftermath: "I don't like Bob." That's a **learned fact** stored independently of the original experience. You've compressed the memory, extracted its essence, and stored that abstraction.

**LLMs don't consolidate anything.** When information falls outside the context window, there's no residual memory, no learned preference, no compressed representation. The model has **complete amnesia**. It's not that it forgot the details but remembers the gist. It has **zero knowledge** that the conversation ever happened.

Think of it this way: Your brain transforms experiences into lasting neural patterns that persist independently. An LLM's "memory" is the text itself, being reread. Remove the text, and the memory vanishes completely.

This is why context windows are better compared to **{{< term "working-memory" "working memory" >}}** (like keeping a phone number in your head) rather than **long-term memory** (like knowing your childhood address). The moment you stop actively thinking about that phone number, it's gone unless you encoded it into long-term storage. LLMs have working memory (the context window) but no mechanism to encode conversations into long-term storage.

The weights trained into the model represent **general patterns from training data**, not memories of **your specific conversations**. Those weights are frozen. Your chat doesn't update them. This is why every new conversation starts from zero, and why **the model never truly "learns" from talking to you**.

---

## Positional Encodings: Teaching Order Without Counting

One of the stranger aspects of context windows is that **transformers have no inherent sense of token order**. The attention mechanism is permutation-invariant. It doesn't naturally know that "The cat chased the mouse" is different from "The mouse chased the cat."

To solve this, models use **{{< term "positional-encoding" "positional encodings" >}}** that inject order information into each token's representation. The original Transformer architecture ([Vaswani et al., 2017](#vaswani2017)) used **absolute positional encodings** with sinusoidal functions:

\\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\\]
\\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\\]

Where \\(pos\\) is the token's position, \\(i\\) is the dimension index, and \\(d_{model}\\) is the model's dimensionality. These sinusoidal functions create unique patterns for each position that the model learns to interpret.

**Relative positional encodings** ([Shaw et al., 2018](#shaw2018); [Dai et al., 2019](#dai2019)) instead encode the **distance** between tokens rather than absolute positions. This helps with longer contexts because the model learns "token A is 5 positions before token B" rather than "token A is at position 1,247."

But both approaches degrade at extreme distances. A token 100,000 positions away has such a different positional encoding that attention scores become unreliable.

---

## What Happens When Context Fills Up

When your conversation exceeds the context window, the web interface (not the model) must employ strategies to manage what gets sent:

### **1. Truncation (Sliding Window)**
The application drops the oldest tokens from what it sends to the API. The model processes only the most recent \\(N\\) tokens.

**Problem:** You lose the beginning of the conversation entirely. Instructions given in message 1 vanish if the conversation grows long enough. The application literally can't send them anymore because they don't fit in the context window.

### **2. Summarization**
Some systems automatically summarize older portions of the conversation and replace them with condensed versions in what gets sent to the API.

**Problem:** Summarization is lossy. Nuance, specific examples, and precise wording disappear. The model is now reasoning over *a summary of your conversation* rather than your actual words.

### **3. {{< term "retrieval-augmented-generation" "Retrieval-Augmented Generation (RAG)" >}}**
The application stores the full conversation in an external database and retrieves relevant chunks to include in the API call.

**Problem:** Retrieval is imperfect. The system has to guess which past messages are relevant. It might retrieve the wrong context or miss critical information.

---

## The KV Cache Optimization

Modern implementations use a clever optimization called **key-value caching**. Instead of recomputing the keys and values for every token in the context on every forward pass, the model caches them.

Here's the efficiency gain:

**Without KV cache:**
- Process 10,000 tokens: compute 10,000 queries, 10,000 keys, 10,000 values
- Generate token 1: compute attention over all 10,000
- Generate token 2: **recompute everything again** plus the new token

**With KV cache:**
- Process 10,000 tokens once: cache the 10,000 keys and values
- Generate token 1: compute 1 new query, reuse cached keys/values
- Generate token 2: compute 1 new query, reuse all cached keys/values plus token 1's cache

This dramatically reduces computational cost, but it increases **memory cost**. Storing 100,000 cached key-value pairs for a 70-billion-parameter model requires gigabytes of GPU memory.

There's always a trade-off: speed vs. memory vs. context length.

---

## Why Longer Context Doesn't Mean Better Performance

You'd think a 1 million token context window would be strictly better than a 100k token window. But research shows diminishing returns and sometimes **worse performance** with extreme context lengths:

**Attention dilution:** With more tokens competing for attention, each individual token receives less focus.

**Positional encoding degradation:** Encoding schemes that work well at 10k tokens break down at 500k tokens.

**"Lost in the middle" phenomenon:** Studies show models perform worst on information buried in the middle of very long contexts ([Liu et al., 2023](#liu2023)). They attend well to the beginning (recency bias inverted by training) and the end (recency bias), but lose track of the middle.

**Increased hallucination:** With more context to "manage," models sometimes confabulate connections between distant, unrelated parts of the conversation.

---

## Visualizing Attention Across Context

Attention weight distribution is easier to understand visually. For a comprehensive visual explanation of how attention mechanisms work, see [Alammar (2018)](#alammar2018).

Below is a simplified simulation showing how attention weights decay across growing context lengths:

### Python Version

Here's a conceptual Python example showing how attention weights distribute across a growing context:

```python
import numpy as np
import matplotlib.pyplot as plt


def attention_weights(context_length, query_position):
    positions = np.arange(context_length)
    recency = np.exp(-0.001 * (query_position - positions))
    distance_penalty = 1.0 / (
        1.0 + 0.00001 * (query_position - positions) ** 2
    )
    weights = recency * distance_penalty
    weights = weights / weights.sum()
    return weights


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, ctx_len in enumerate([1000, 10000, 100000]):
    query_pos = ctx_len - 1
    weights = attention_weights(ctx_len, query_pos)
    axes[idx].plot(weights)
    axes[idx].set_title(f'Context Length: {ctx_len:,} tokens')
    axes[idx].set_xlabel('Token Position')
    axes[idx].set_ylabel('Attention Weight')
    axes[idx].set_yscale('log')

plt.tight_layout()
plt.show()

```

**What the plot reveals:**

{{< lightbox
    src="visualizing-attention-across-context-python.png"
    alt="Three plots showing attention weight decay across context lengths of 1,000, 10,000, and 100,000 tokens"
    label="Open full-size attention weight decay plots"
    caption="As context grows, attention spreads across exponentially more tokens, creating a flatter, noisier distribution. The signal-to-noise ratio degrades."
>}}


### MATLAB Version

To reproduce the same curve in MATLAB, use the following script:

```matlab
function visualize_attention_decay()
    figure('Position', [100 100 1200 400]);
    context_lengths = [1000, 10000, 100000];

    for i = 1:length(context_lengths)
        ctx_len = context_lengths(i);
        query_pos = ctx_len;
        positions = 1:ctx_len;

        recency = exp(-0.001 * (query_pos - positions));
        distance_penalty = 1 ./ ...
            (1 + 0.00001 * (query_pos - positions) .^ 2);

        weights = recency .* distance_penalty;
        weights = weights / sum(weights);

        subplot(1, 3, i);
        semilogy(positions, weights, ...
            'LineWidth', 1.5);

        title(['Context Length: ', ...
            num2str(ctx_len)]);

        xlabel('Token Position');
        ylabel('Attention Weight (log scale)');
        grid on;
    end
end
```

This produces the same curve, showing that as the number of tokens grows, older tokens fade into near-zero significance.

{{< lightbox
    src="visualizing-attention-across-context-matlab.png"
    alt="Three MATLAB plots showing attention weight decay across context lengths of 1,000, 10,000, and 100,000 tokens"
    label="Open full-size MATLAB attention weight decay plots"
    caption="As context grows, attention spreads across exponentially more tokens, creating a flatter, noisier distribution. The signal-to-noise ratio degrades."
>}}

---

## Retrieval-Augmented Generation (RAG)

RAG bridges the gap between finite context and long-term knowledge. Instead of depending on the model's internal attention, RAG systems perform **{{< term "semantic-retrieval" "semantic retrieval" >}}** to inject only the most relevant information into each API call.

1. **Chunking** text or conversation history into semantically meaningful segments.
2. **{{< term "embedding" "Embedding" >}}** each chunk into a vector space.
3. **Retrieving** top-k relevant segments using {{< term "cosine-similarity" "cosine similarity" >}}.
4. **Composing** a dynamic prompt with the query and retrieved context.
5. **Generating** an answer grounded in that retrieved material.

This scales almost logarithmically with data size, though retrieval accuracy becomes the bottleneck. If the retrieval misses a key piece of context, the response quality collapses.

Recent work on context compression models and adaptive retrieval improves this by predicting relevance before querying ([Jiang et al., 2023](#jiang2023)). GPT-4's emerging memory features and Claude's "Artifacts" hint at this direction with persistent, embedded summaries that act as *latent conversation memory*.

---

## Context Windows in Practice

{{< lightbox
    src="plot.png"
    alt="Horizontal bar chart showing context window requirements across nine use cases, categorized as short context sufficient (green), long context valuable (blue), and long context problematic (orange), with token ranges from hundreds to millions on a logarithmic scale"
    label="Open full-size plot"
    caption="Context requirements vary by task. Short contexts work best for focused tasks like code completion, while massive contexts often perform worse than retrieval-based approaches for complex reasoning."
>}}

Different use cases have different context needs:

**Short context sufficient:**
- Code completion (hundreds of tokens)
- Customer service chatbots (thousands of tokens)
- Translation (thousands of tokens)

**Long context valuable:**
- Document analysis (tens of thousands of tokens)
- Long conversation threads (tens of thousands of tokens)
- Codebase understanding (hundreds of thousands of tokens)

**Long context problematic:**
- Complex reasoning across many documents (attention dilution)
- Extracting specific facts from massive context (retrieval more effective)
- Maintaining consistent state over very long interactions (external memory better)

The key insight: **long context is a tool, not a solution**. For many tasks, structured retrieval or explicit memory systems outperform simply cramming everything into the context window.

---

## The Future: Beyond Attention

Researchers are pushing beyond quadratic attention toward architectures that combine **efficiency, persistence, and abstraction**.

**1. Linearized and Hybrid Attention**
Models like **{{< term "mamba" "Mamba" >}}** ([Gu & Dao, 2023](#gu2023)) and **{{< term "rwkv" "RWKV" >}}** ([Peng et al., 2023](#peng2023)) merge recurrent updates with attention-style gating, keeping a compact "state trace" instead of recomputing pairwise attention. They process input almost linearly with sequence length, hinting at a path to true scalability.

**2. Modular Memory Architectures**
Upcoming designs experiment with distinct submodules for short-term reasoning, factual recall, and abstract summaries. Instead of one monolithic attention map, different "memory heads" specialize, which is closer to how human cognition distributes recall.

**3. Hierarchical Context**
Multi-scale models use local attention for fine detail and upper layers for long-span summaries, compressing sequences without losing coherence. The model reasons hierarchically, the way we summarize and then recall the gist.

**4. Neuro-symbolic Integration**
Long-term context may eventually blend neural pattern recognition with symbolic memory. Rather than rereading raw text, future systems could retrieve *structured {{< term "knowledge-graph" "knowledge graphs" >}}*, giving them durable, compositional recall.

The direction is clear: the next breakthroughs won't simply extend context. They'll **reshape what it means to have context at all**.

---

## Prompt Engineering and Context Windows

Knowing how attention fades helps you design prompts that survive long contexts. {{< term "prompt-engineering" "Prompt engineering" >}} is really attention sculpting: placing information where the model is most likely to "see" it.

Reinforce key details near the end. Recency bias makes the last tokens disproportionately influential. Group related instructions so directives are not scattered. Prefer concise summaries over raw dumps of conversation. Repeat critical goals near the output boundary to keep them in focus.

For example, if you're asking the model to analyze a long document and extract specific facts, place your extraction criteria at the *end* of the prompt, after the document text. This ensures the model "sees" your requirements with maximum attention when generating its response.

In practice, the tail of your prompt is prime real estate. The closer critical details are to the end, the more reliably they guide the response.

---

## GPT-4 vs. Claude: Long Context Behavior

Both GPT-4 (128k tokens) and Claude 3 (200k) handle extended sequences gracefully, but they show distinct personalities when context stretches toward the limit.

**Claude 3** tends to preserve *semantic coherence*. It paraphrases well and often retains conceptual meaning even when exact phrases drift out of focus. However, its precision drops past 150k tokens where quotes blur, and substitutions appear. Claude maintains tone consistency remarkably well, suggesting active summarization before truncation.

**GPT-4** leans toward *syntactic fidelity*. It reproduces exact code and phrasing more accurately and resists paraphrasing. Yet it's more brittle with unstructured input; when order or hierarchy is lost, GPT-4's logical scaffolding weakens faster. On complex reasoning tasks, though, it often outperforms Claude because its tighter attention preserves logical dependencies.

In short, Claude remembers **the idea**, GPT-4 remembers **the wording**. This difference is a reflection of how each platform balances summarization and exact recall.

---

## Closing Thoughts

Longer context windows don't create memory; they create *the appearance* of memory. They expand the stage on which attention performs but don't change the nature of the act. The real progress will come from systems that know what to retain, what to retrieve, and what to ignore.

After years of building AI tools for documentation at Microsoft, I've learned that this pattern isn't a flaw; it's a design choice. The model isn't forgetful; it's efficient. It reads everything you show it, decides what matters mathematically, and writes a response based on the clarity of what's in view right now.

Understanding that transforms how you work with LLMs. You restate. You summarize. You keep the spotlight pointed where you need it. Because when you realize that an LLM's memory is simply its present context, you stop wishing for something it doesn't have, and start mastering what it actually does.

---

**If you've followed this series from [how LLMs read code](/posts/how-large-language-models-read-code) through [how they think](/posts/how-large-language-models-think) and [how they learn](/posts/how-large-language-models-learn), you now understand the full picture: pattern recognition through probability, geometric meaning through linear algebra, improvement through calculus, and the illusion of memory through attention. What emerges is a system that's powerful precisely because of, not despite, its constraints.**

---

## Try It Yourself

[Download the full code on GitHub](https://github.com/TomArcher/technical-blog-examples/tree/main/how-large-language-models-handle-context-windows)

---
