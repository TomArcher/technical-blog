+++
date = '2025-10-06T09:00:00-07:00'
aliases = [
    "/posts/how-ai-reads-code/",
    "how-large-language-models-read-code",
]
draft = false
title = "How Large Language Models (LLMs) Read Code: Seeing Patterns Instead of Logic"
subtitle = "Exploring how large language models interpret code and what they miss"
categories = ["AI and the Mathematics of Language"]
tags = ["AI", "LLM", "probability", "tokenization",]
author = "Tom Archer"
listThumb = "how-large-language-models-read-code.png"

hero = "how-large-language-models-read-code.png"
heroAlt = "Digital artwork showing a small piece of code outside an AI silhouette with circuit lines and a glowing probability curve inside its head, symbolizing machine learning interpreting code through statistical modeling rather than logic."
heroLabel = "Open full-size how large language models read code"
heroCaption = "AI reads code as patterns, not instructions."

whatYoullLearn = [
    "How an LLM reads code differently from a compiler or a human developer",
    "Why models recognize programming patterns instead of executing the code they see",
    "How embeddings let an LLM associate code with similar structures and meanings",
    "Why comments, variable names, and familiar coding idioms can change a model's interpretation",
    "How statistically likely code can still be logically or operationally wrong",
    "Why combining generative AI with compilers and static analysis produces safer coding tools"
]

[[references]]
key = "allamanis2018"
citation = "Allamanis, M., Barr, E. T., Devanbu, P., & Sutton, C. (2018). A survey of machine learning for big code and naturalness. *ACM Computing Surveys, 51*(4), Article 81."
url = "https://doi.org/10.1145/3212695"

[[references]]
key = "chen2021"
citation = "Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. de O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., ... Zaremba, W. (2021). Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*."
url = "https://arxiv.org/abs/2107.03374"

[[references]]
key = "feng2020"
citation = "Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., & Zhou, M. (2020). CodeBERT: A pre-trained model for programming and natural languages. *Findings of the Association for Computational Linguistics: EMNLP 2020*, 1536-1547."
url = "https://arxiv.org/abs/2002.08155"

[[references]]
key = "kanade2020"
citation = "Kanade, A., Maniatis, P., Balakrishnan, G., & Shi, K. (2020). Learning and evaluating contextual embedding of source code. *Proceedings of the 37th International Conference on Machine Learning, 119*, 5110-5121."
url = "https://proceedings.mlr.press/v119/kanade20a.html"

[[references]]
key = "li2025"
citation = "Li, Y., Qi, S., Gao, C., Peng, Y., Lo, D., Xu, Z., & Lyu, M. R. (2025). A closer look into Transformer-based code intelligence through code transformation: Challenges and opportunities. *IEEE Transactions on Software Engineering*."
url = "https://arxiv.org/abs/2207.04285"

[[references]]
key = "vaswani2017"
citation = "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*."
url = "https://arxiv.org/abs/1706.03762"

+++

Developers are accustomed to thinking about code in terms of {{< term "syntax-semantics" "syntax and semantics" >}}, the how and the why. Syntax defines what is legal; semantics defines what it means. A compiler enforces syntax with ruthless precision and interprets semantics through symbol tables and execution logic. But a {{< term "large-language-model" "Large Language Model (LLM)" >}}, reads code the way a seasoned engineer reads poetry, recognizing rhythm, pattern, and context more than explicit rules. 


The difference may seem subtle, but it has vast consequences. Understanding the gap between human reasoning, compiler verification, and model prediction is key to using generative AI responsibly in programming environments.

---

## What Is an LLM

If you're new to the world of generative AI, it helps to start with a clear idea of what a large language model actually is. An LLM is an AI system trained on vast collections of text to recognize and reproduce the patterns of human language. It doesn't just store sentences; it learns relationships between words, ideas, and structures.

These models can do many things:

* Answer questions
* Write content
* Translate languages
* Summarize text
* Hold conversations
* Generate code

**Examples:** [ChatGPT (OpenAI)](https://chatgpt.com/), [Claude (Anthropic)](https://claude.ai/new), [Gemini (Google)](https://gemini.google.com/app), and [LLaMA (Meta)](https://www.llama.com/).

The "large" in large language model refers to the scale of {{< term "model-parameter" "parameters" >}} where billions of adjustable values tune how the model interprets and generates text.

At its core, an LLM is a probability engine. It predicts the next most likely word or {{< term "token" "token" >}} based on the context of what came before ([Vaswani et al., 2017](#vaswani2017)). That simple act, repeated across billions of examples during training, is what gives these models the ability to sound fluent, coherent, and contextually aware.

In other words, an LLM doesn't think about language; it **models** language itself.

---

## Syntax as Pattern, Not Rule

When a compiler reads a function like the following, it parses tokens, constructs an {{< term "abstract-syntax-tree" "abstract syntax tree (AST)" >}}, and transforms the result into intermediate bytecode. The semantics are precise: multiply the variable `x` by itself.

```python
def square(x): 
    return x * x
```

{{< lightbox
    src="how-large-language-models-read-code-flow.png"
    alt="Flowchart-style illustration showing how an AI model {{< term"
    label="Open full-size how large language models read code flow"
    caption="Patterns drive AI recognition; not syntax."
>}}

An LLM, by contrast, does not parse in the traditional sense. It tokenizes text into learned units and predicts tokens from their statistical context rather than executing the program as a compiler does ([Chen et al., 2021](#chen2021)). The model "understands" that `return x * x` likely follows `def square(x):` because it has seen this pattern thousands of times across training corpora, not because it knows what multiplication does.

In the language of probability, a compiler computes meaning deterministically; a model approximates it {{< term "stochastic" "stochastically" >}}.

---

## The Shape of Understanding

When a human reads code, we chunk it semantically. The line `for user in data:` evokes an internal schema: iteration, collection, filtering. The model does something analogous, but its mental map is geometric, not symbolic.

Consider the following prompt given to a model fine-tuned on code:

```
"def process_users(data):"
```

This becomes a dense vector in a high-dimensional {{< term "embedding" "embedding space" >}}. Nearby vectors might represent similar constructs like "process_orders(data)" or "handle_clients(list)."

These proximity relationships are the raw materials of AI understanding ([Kanade et al., 2020](#kanade2020)). The closer two snippets lie in vector space, the more the model perceives them as semantically related, even when the model has no explicit representation of what a user or an order is.

Embeddings compress source-code information into learned geometric representations ([Kanade et al., 2020](#kanade2020)). Code with similar structure, naming, and flow tends to cluster, which is why renaming a variable or removing a comment can subtly shift a model's interpretation.

---

## The Comment Paradox

To illustrate, try this small experiment using the OpenAI API:

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Explain what this function does:\n\n"
                "# Send a welcome email to all active users\n"
                "def process_users(data):\n"
                "    for user in data:\n"
                "        if user.is_active:\n"
                "            send_email(user)"
            ),
        }
    ],
    logprobs=True,
)
```

The model will usually reply that the function "sends a welcome email to all active users." Now remove the comment and run it again. The response will still be similar, but the {{< term "probability-distribution" "probability distribution" >}} shifts: the model's confidence in "welcome email" drops because the lexical hint vanished.

Comments not only help humans; models trained jointly on programming and natural languages can learn relationships between code and natural-language descriptions ([Feng et al., 2020](#feng2020)). Embeddings are sensitive to natural language cues because language and code share the same token vocabulary. That is why consistent commenting style, clear naming, and logical spacing often yield more accurate AI-assisted explanations and refactorings.

---

## When Syntax Misleads Semantics

Because models learn statistical regularities rather than executing code, generated or interpreted code can be plausible while still being functionally incorrect ([Chen et al., 2021](#chen2021)). A variable named `result` near `sum()` nudges the model to assume aggregation, even if the code computes a difference. The model's "understanding" is weighted toward linguistic bias.

Take this example:

```python
def calculate_difference(a, b):
    result = a + b
    return result
```

A human instantly spots the contradiction between the name and operation. A compiler does not care. An LLM, however, may explain this as "subtracts one number from another," proving that its semantic space privileges pattern frequency over operational truth.

Studies of code-focused Transformers have shown that changing identifiers while preserving the underlying program semantics can substantially degrade model performance. Li et al. found that identifier transformations reduced performance across code completion, code search, and code summarization tasks, demonstrating how strongly these models can depend on identifier semantics [Li et al., 2025](#li2025).

---

## The Statistical Mind

LLMs do not parse control flow; they predict control flow. When you type `for`, the model's top token candidates include `i`, `item`, and `user`. When it predicts `if user.is_active:`, it has learned a latent schema: "loop + conditional + method call" often ends in a side effect like `send_email(user)` or `update_status(user)`.

That is not understanding in the compiler sense; it is associative modeling. But this statistical machinery is astonishingly effective because code follows social, not natural, evolution. Developers imitate idioms, and statistical models of code exploit this regularity ([Allamanis et al., 2018](#allamanis2018)). Together, they form a feedback loop of probabilistic convention.

---

## From Tokens to Intent

To see how deep this patterning goes, look at a model's {{< term "log-probability" "log probabilities" >}} for a simple prompt:

```
"def is_palindrome(s): return s == s[::-1]"
```

The log probability for `s[::-1]` is extremely high because that slice notation is a canonical pattern in the training corpus for palindrome detection. (For a deeper look at what *log probability* means and why models use it, see [Inside the Mind of a Model: How AI Turns Meaning into Math](/posts/how-ai-turns-meaning-into-math/).)

Now, consider a less common variant of the same prompt:

```
"def is_palindrome(s): return s == ''.join(reversed(s))"
```

Here, the probability distribution shifts. Both are correct, but one feels "unnatural" to the model. AI reads code through learned statistical regularities rather than an authoritative execution semantics ([Allamanis et al., 2018](#allamanis2018)).

---

## The Compiler and the Poet

A compiler knows exactly what your code does and cares nothing about what you meant. A language model knows approximately what you meant and nothing about what your code does.

The compiler enforces the syntax of logic; the model enforces the logic of culture. The former transforms instructions into machine behavior. The latter transforms text into probabilities that resemble meaning. When these systems meet, such as in Copilot or GitHub's autocomplete, they complement each other beautifully. The compiler guarantees execution; the model suggests intention.

---

## The Power of Context Windows

One of the most underappreciated aspects of AI code comprehension is the size of its {{< term "context-window" "context window" >}}. The broader the context, the closer a model gets to true comprehension.

In human terms, a developer reading fifty lines can recall relationships across functions; a model with a 128k token window can recall dependencies across entire modules. This does not facilitate logical reasoning, but it does enable global pattern retention, which is crucial for tasks such as refactoring, summarization, or maintaining style consistency.

---

## Experimenting with Prompt Geometry

Developers can exploit the geometric nature of embeddings by rephrasing code-related prompts. For example, rather than asking:

> "Explain this code."

Ask:

> "What would this function's docstring likely say in a production environment?"

That subtle shift pushes the model's attention toward documentation-style patterns in embedding space, yielding more reliable summaries. Understanding this geometric reasoning, how nearby textual forms affect token probabilities, is becoming an essential literacy for AI-assisted programming.

---

## When Probabilities Meet Production

Models that read code can accelerate programming tasks, but they introduce risk if developers mistake probability for proof ([Chen et al., 2021](#chen2021)). A suggestion may be statistically likely but logically wrong.

In safety-critical domains such as finance, medicine, and infrastructure, LLMs should never operate without a deterministic verification layer. Tools that combine {{< term "static-analysis" "static analysis" >}} with generative suggestions, such as semantic linting or {{< term "differential-testing" "differential testing" >}}, provide a bridge between the stochastic intuition of AI and the formal rigor of compilers.

---

## Toward a Hybrid Intelligence

The real frontier lies in coupling deterministic parsers with probabilistic interpreters. Imagine an IDE where the compiler exposes ASTs and an LLM attaches commentary to each node, explaining likely intent, flagging anomalies, and predicting downstream effects.

Such systems would merge two epistemologies: the compiler's precision and the model's pattern sense. Humans would no longer alternate between "write mode" and "read mode" but collaborate with an entity capable of probabilistic empathy for code.

---

## Closing Thoughts

After many years of writing software, I have come to realize that code is as cultural as it is logical. Every function carries fingerprints of habits, mentors, and languages long gone. Large language models do not understand code the way we do; they remember it, in the collective statistical sense. They compress decades of programming idioms into geometry.

That is why an AI sometimes finishes your thought before you finish typing. It is not reading your mind; it is reading the echoes of every mind that came before you.

And that, in its strange, approximate way, is a kind of understanding.

If you'd like to see how these ideas translate into math and geometry, continue with [Inside the Mind of a Model: How AI Turns Meaning into Math](/posts/how-ai-turns-meaning-into-math/).

---