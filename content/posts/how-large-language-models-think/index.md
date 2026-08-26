+++
date = '2025-10-07T06:00:00-07:00'
aliases = [
    "/posts/how-ai-turns-meaning-into-math/",
    "how-large-language-models-think",
]
draft = false
title = "How Large Language Models (LLMs) Think: Turning Meaning into Math"
subtitle = "Exploring how large language models use linear algebra to create geometric meaning"
categories = ["AI and the Mathematics of Language"]
tags = ["AI", "embeddings", "geometry", "linear algebra", "LLM",]
author = "Tom Archer"
listThumb = "how-large-language-models-read-code-thumb.png"

hero = "how-large-language-models-read-code.png"
heroAlt = "Digital artwork illustrating how a large language model represents language as mathematical relationships in a high-dimensional vector space."
heroLabel = "Open full-size how large language models think"
heroCaption = "Meaning takes shape in mathematics long before it reaches words."

whatYoullLearn = [
    "How an LLM turns words and tokens into numerical vectors it can process",
    "How distance and direction in embedding space can represent relationships in meaning",
    "Why linear algebra and geometry are two ways of describing the same internal structure",
    "How matrix operations transform information as it moves through a model",
    "Why high-dimensional spaces can represent many subtle features of language at once",
    "How probability guides a model from its current context toward the next token"
]


[[references]]
key = "mikolov2013"
citation = "Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*."
url = "https://arxiv.org/abs/1301.3781"

[[references]]
key = "mikolov2013linguistic"
citation = "Mikolov, T., Yih, W.-t., & Zweig, G. (2013). Linguistic regularities in continuous space word representations. *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 746-751."
url = "https://aclanthology.org/N13-1090/"

[[references]]
key = "pennington2014"
citation = "Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1532-1543."
url = "https://aclanthology.org/D14-1162/"

[[references]]
key = "vaswani2017"
citation = "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*."
url = "https://arxiv.org/abs/1706.03762"

[[references]]
key = "ethayarajh2019"
citation = "Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 55-65."
url = "https://aclanthology.org/D19-1006/"
+++

When you enter a sentence into a {{< term "large-language-model" "Large Language Model (LLM)" >}} such as [ChatGPT](https://chatgpt.com/) or [Claude](https://claude.ai/new), the model does not process words as language. It represents them as numbers.

Each word, phrase, and code {{< term "token" "token" >}} becomes a vector — a list of real-valued coordinates within a {{< term "high-dimensional-space" "high-dimensional space" >}}. Relationships between meanings are captured not by grammar or logic but by geometry. The closer two vectors lie, the more similar their semantic roles appear to the model.

This is the mathematical foundation of large language models: linear algebra. {{< term "matrix-multiplication" "Matrix multiplication" >}}, {{< term "vector-projection" "vector projection" >}}, {{< term "cosine-similarity" "cosine similarity" >}}, and {{< term "normalization" "normalization" >}} define how the model navigates this vast space of meaning. What feels like understanding is actually the alignment of high-dimensional vectors governed by probability and geometry.


This post explains how those operations create what we perceive as context and comprehension. You will see how linear algebra forms the bridge between words and meaning and how tools such as Python or [MATLAB](https://www.mathworks.com/products/matlab.html) make this hidden structure visible. For a conceptual introduction to how models interpret code, start with [How AI Reads Code: What Large Language Models Actually Understand](/posts/how-ai-reads-code/).

---

## From Words to Numbers

Computers don't understand language. They understand numbers. So before a model can process text, it has to convert every token (a word or fragment) into a numerical vector (in this case, representing a list of floating-point values).

Here's a simplified Python example:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(["AI reads code", "AI writes code"])
print(X.toarray())
```

**Output:**

```
[[1 1 1 0]
 [1 0 1 1]]
```

Each word becomes part of a vector representation showing which tokens appear together. Modern models go far beyond this in that they use dense, continuous {{< term "embedding" "embedding" >}}s instead of simple counts, but the principle is the same: text becomes math ([Mikolov et al., 2013](#mikolov2013); [Pennington et al., 2014](#pennington2014)).

---

## The Geometry of Meaning

An *embedding* is not a dictionary lookup. It is a **coordinate** in a high-dimensional space with thousands of dimensions where mathematical proximity corresponds to semantic similarity.

In this space, "cat" sits close to "dog," "car" lies near "truck," and `process_users()` resides near `handle_clients()`. Proximity encodes relationship.

You can visualize a simplified version in MATLAB or Python. For instance, in MATLAB:

```matlab
words = ["king","queen","man","woman"];
vectors = [ ...
    0.8 0.6;   % king
    0.75 0.65; % queen
    0.6 0.3;   % man
    0.55 0.35  % woman
];
plot(vectors(:,1), vectors(:,2),'o');
text(vectors(:,1)+0.02, vectors(:,2), words);
```

This tiny example mimics one of the most famous demonstrations of linguistic regularities in word embeddings ([Mikolov et al., 2013](#mikolov2013linguistic)):
**king – man + woman ≈ queen.**

Even though these are just coordinates, the geometry encodes analogy. That's the quiet miracle of embeddings where relations emerge from math alone.

---

## Linear Algebra and the Geometry of Thought

When I first started deep-diving into how LLMs work under the hood, I struggled to reconcile the fact that LLMs rely on geometry when I remembered that in my level 300 and 400 math courses, working with vectors was done through linear algebra. The reason for this mental tension was that I had always treated geometry and linear algebra as separate domains with geometry being something visual and spatial, and linear algebra being something symbolic and procedural.

What I eventually realized is that they are two views of the same thing. The math behind a large language model is entirely linear algebra: multiplying matrices, taking {{< term "dot-product" "dot products" >}}, projecting vectors, computing {{< term "vector-norm" "norms" >}}. But what those operations create is a *geometric world.*

A vector is just a list of numbers, but when millions of those vectors interact through {{< term "linear-transformation" "linear transformation" >}}s, they define a space where distance, angle, and direction become meaningful. Similar words or code fragments cluster together; analogies become lines; transformations become rotations and translations in thousands of dimensions.

So the geometry is not literal; it's emergent. The network doesn't draw shapes; it performs math that *behaves* geometrically. That's why "geometry" is such an accurate metaphor for how LLMs represent meaning. The linear algebra is the physics, and the geometry is the language we use to understand it.

---

## Eigenvectors: The Hidden Axes of Meaning

Every linear transformation within an LLM, including the learned projections used by attention, can be described as a matrix acting on vectors ([Vaswani et al., 2017](#vaswani2017)). But not all directions in that space change equally. Some directions remain stable while others stretch or shrink. Those privileged directions are defined by {{< term "eigenvector" "eigenvectors" >}}, and the amount of stretching or compression along them is determined by their {{< term "eigenvalue" "eigenvalues" >}}.

In an embedding space, you can think of eigenvectors as the hidden axes along which meaning varies most strongly. One direction might capture gender, another might capture tense, another might reflect tone or formality. These axes are not programmed; they emerge from training as the model learns to organize information in ways that minimize error.

Mathematically, this relationship is expressed as: `Av = λv`.

Where:

- `A` is a transformation matrix
- `v` is an eigenvector
- `λ` (lambda) is the eigenvalue

The eigenvalue tells us how much the transformation scales that direction. In practical terms, LLMs contain thousands of such matrices, each shaping information flow in subtle but predictable ways.

When researchers analyze embeddings using techniques like {{< term "singular-value-decomposition" "singular value decomposition (SVD)" >}} or {{< term "principal-component-analysis" "principal component analysis (PCA)" >}}, they are effectively identifying these dominant eigenvectors, the directions that explain the most structure in meaning. This is why, even though the model's internal space has thousands of dimensions, a handful of them often capture broad semantic relationships.

In that sense, eigenvectors reveal the skeleton of understanding inside the model: the stable, interpretable directions that give geometric form to meaning itself.

---

## Meaning as Direction

In embedding space, directions between vectors can encode linguistic regularities ([Mikolov et al., 2013](#mikolov2013linguistic)). For example:

* Moving in one direction might represent **gender** (man → woman).
* Another might represent **tense** (run → ran).
* Another might represent **formality** (kid → child → youth).

These relationships exist because models learn to compress the messy richness of language into consistent geometric transformations.

In code models, directions might encode things like:

* `function → class` (abstraction)
* `print → return` (output mechanism)
* `public → private` (scope)

When an LLM predicts your next token, it's essentially following these invisible directions through vector space.

---

## Visualizing It with Probability

Every time a model predicts the next token, it's estimating how close the new vector should be to the current path. Think of it as walking through this landscape one step at a time, guided by probability rather than certainty.

Here's a conceptual MATLAB snippet that simulates that:

```matlab
% Random 2D embedding of 4 tokens
tokens = ["def", "for", "if", "return"];
E = rand(4,2);

% Simulated "context vector" representing the current prompt
context = mean(E([1,2],:));  % halfway between 'def' and 'for'

% Compute cosine similarities (likelihood of next token)
similarity = E * context' ./ (vecnorm(E,2,2) * norm(context));
[~, idx] = max(similarity);

fprintf("Most likely next token: %s\n", tokens(idx));
```

**Output:**

```
Most likely next token: return
```

The code above doesn't generate real text; it illustrates the logic. The {{< term "context-vector" "context vector" >}}, which represents everything you've communicated so far to the model, points somewhere in the embedding space. The model searches for the nearest vectors and selects the one most aligned. **That is prediction in geometric form**.

---

## Why Models Use Log Probabilities

If you've read *[How AI Reads Code](/posts/how-ai-reads-code)*, you might remember the phrase "{{< term "log-probability" "log probability" >}}." Log probability is a small mathematical trick with a big purpose.

When a model predicts the next token, it doesn't make a single guess. It assigns a probability distribution over the vocabulary, from which the next-token prediction is obtained ([Vaswani et al., 2017](#vaswani2017)). For example, in a code context for implementing a palindrome check, the model might assign `s[::-1]` a probability of 0.93 (almost certain), `reversed(s)` a probability of 0.05, and everything else close to zero.

Multiplying those probabilities across long sequences quickly drives the numbers into infinitesimal fractions, making them computationally unstable. To avoid that instability, models work in logarithmic space, where multiplication becomes addition and tiny values remain manageable.

The result is a set of log probabilities, negative numbers that are easier to compare and sum across many steps. Because the logarithm of a number between 0 and 1 is negative, a confident prediction such as 0.93 becomes a value close to zero (around -0.07), while an uncertain one such as 0.05 becomes a much smaller number (around -3.0). In log space, a less negative value means greater confidence.

In other words, log probabilities do not change what the model believes. They simply make belief computable.

---

## Why High Dimensions Matter

It is easy to imagine this in two dimensions, but modern language representations operate in high-dimensional spaces whose geometry can encode substantial contextual information ([Ethayarajh, 2019](#ethayarajh2019)).

In two dimensions, you can represent a few relationships, such as animals versus vehicles. In ten thousand dimensions, you can represent subtler unions: gender, tone, syntax, domain, even emotional valence.

High-dimensional representations allow models to encode complex linguistic information in geometric relationships ([Ethayarajh, 2019](#ethayarajh2019)). That is why they can generalize across tasks, languages, and even programming paradigms.

---

## The Shape of Understanding

When you ask an LLM to complete your sentence, it is not retrieving an answer; it is following a path through vector space.

Each token pulls the next one toward its neighborhood. Each neighborhood has local rules learned from millions of examples. Together, they form a topology of understanding, a map that balances memory and probability.

In that sense, an LLM does not think; it navigates. It is a statistical traveler in a landscape shaped by human language.

---

## Closing Thoughts

Understanding embeddings turns the mystery of AI into something tangible. **When you realize that "meaning" is just geometry, you start to see why models feel both powerful and fragile.** They are not reasoning in symbols or words; they are surfing through probability.

MATLAB and Python give us two lenses to explore that space: Python for the implementation, MATLAB for the math. Between the two, you can watch meaning become math in real time.

And once you have seen that happen, the phrase *"AI understands"* feels less magical and far more human.

Now that you have a basic overview of how LLMs think, if you'd like to see how LLMs learn, continue with [How Large Language Models (LLMs) Learn: Calculus and the Search for Understanding](/posts/how-large-language-models-learn/).

---