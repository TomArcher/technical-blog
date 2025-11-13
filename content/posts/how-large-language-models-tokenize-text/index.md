+++
date = '2025-11-11T06:00:00-07:00'
draft = false
title = "How Large Language Models (LLMs) Tokenize Text: Why Words Aren't What You Think"
subtitle = "Understanding how LLMs break language into pieces—and why it matters more than you realize"
categories = ["AI and the Mathematics of Language"]
tags = ["AI", "tokenization", "LLMs", "transformers", "NLP"]
author = "Tom Archer"
listThumb = "how-large-language-models-tokenize-text.png"
+++

<figure style="float: right; margin: 0 20px 10px 20px; width: 250px; text-align: center;">
  <img src="./how-large-language-models-tokenize-text.png"
       alt="Digital artwork showing text being broken into irregular puzzle pieces, with some pieces glowing to indicate tokens"
       width="250"
       style="display: block; margin: 0 auto;">
  <figcaption style="font-size: 0.9em; color: #555; margin-top: 5px;">
    <em>LLMs read tokens. Not words. A distinction with a technical—and potentially financial—difference.</em>
  </figcaption>
</figure>

When you type "I love programming" into ChatGPT, you might assume the model reads three words. It doesn't. It reads somewhere between three and seven tokens, depending on how the text is split.

When you ask Claude to count the letters in the word "strawberry," it often gets it wrong. The reason is simple. Claude never saw the word "strawberry" as a complete unit. It saw tokens like `"str"`, `"aw"`, `"berry"` and tried to reason about letters it couldn't directly access.

And when early GPT-3 users discovered that typing "SolidGoldMagikarp" caused the model to behave erratically - generating nonsense, refusing requests, or producing bizarre outputs - the culprit wasn't the model's training. It was a **glitch token**: a tokenization artifact that never appeared in training data, leaving the model with no learned representation for how to handle it ([Rumbelow & Watkins, 2023](#rumbelow2023)).

---

>*"To a language model, text isn't a stream of words. It's a sequence of tokens. The way those tokens are created determines what the model can and cannot understand."*

---

<!--more-->

This post explores how tokenization works, why it's not as simple as splitting on spaces, and what happens when tokenization goes wrong. If you're following my series on how LLMs work, this is the foundation that makes everything else possible—because before a model can [read](/posts/how-large-language-models-read-code), [think](/posts/how-large-language-models-think), or [learn](/posts/how-large-language-models-learn), it has to break language into pieces it can process.

---

## What Is Tokenization?

**Tokenization is the process of breaking text into units (tokens) that a language model can process.**

At first glance, this seems trivial: just split on spaces and punctuation, right? But natural language is messier than that:

- `"don't"` → Is this one word or two? (`"do"` + `"n't"`)
- `"COVID-19"` → Should the hyphen and numbers be separate?
- `"🚀"` → How do you handle emoji?
- `"Dr. Smith"` → Is the period part of the abbreviation or sentence-ending punctuation?
- `"New York"` → Two words, but one semantic unit

Different tokenization strategies make different tradeoffs between **vocabulary size**, **sequence length**, and **semantic coherence**.

---

## Why Not Just Use Words?

The obvious approach would be to treat every word as a token. English has roughly 170,000 words in common use. Build a vocabulary of those words, assign each an ID, and you're done.

**This breaks immediately:**

### **1. Vocabulary Explosion**
Natural language isn't a fixed set of words:

- New words appear constantly: `"selfie"`, `"cryptocurrency"`, `"doomscrolling"`
- Proper nouns are infinite: every person, place, company, product
- Technical terms proliferate: `"hyperparameter"`, `"backpropagation"`, `"gradient"`
- Typos and variations: `"looooove"`, `"sooooo"`, `"hahahaha"`

A word-level vocabulary would need **millions of entries** and still couldn't handle novel terms.

### **2. Morphological Complexity**
Words have forms:

- `"run"`, `"runs"`, `"running"`, `"ran"`, `"runner"`
- `"happy"`, `"happier"`, `"happiest"`, `"happiness"`, `"happily"`

Word-level tokenization treats these as completely unrelated tokens, even though they share a root. The model has to relearn the relationship from scratch.

### **3. Multilingual Challenges**
Not all languages have spaces:

- Chinese: 我爱编程 (no word boundaries)
- Japanese: 私はプログラミングが好きです (mixed scripts, no consistent spacing)
- German: `"Donaudampfschifffahrtsgesellschaftskapitän"` (compound words)

Word-level tokenization doesn't generalize across languages.

### **4. Out-of-Vocabulary Problem**
If you encounter a word not in your vocabulary, you're stuck. You can't process it, can't embed it, can't generate it. The model has a blind spot for anything it wasn't explicitly trained to recognize.

---

## Character-Level Tokenization: The Other Extreme

If words are too coarse, what about going fine-grained? Treat every character as a token:

`"I love programming"` → `["I", " ", "l", "o", "v", "e", " ", "p", "r", "o", "g", "r", "a", "m", "m", "i", "n", "g"]`

### **Advantages:**
- ✅ Tiny vocabulary (26 letters + punctuation + digits ≈ 100 tokens)
- ✅ No out-of-vocabulary problem (can represent any text)
- ✅ Works for all languages

### **Fatal Flaw: Sequence Length Explosion**
A 500-word document becomes **~2,500 characters**. Since transformers have quadratic attention cost (see my [context windows post](/posts/how-large-language-models-handle-context-windows)), processing character sequences is **computationally prohibitive**.

Worse, the model has to learn that `"c-a-t"` forms a semantic unit, that `"p-r-o-g-r-a-m-m-i-n-g"` shares meaning with `"p-r-o-g-r-a-m"`, and so on. You've pushed all the linguistic structure learning onto the model.

---

## Subword Tokenization: The Goldilocks Solution

Modern LLMs use **subword tokenization**: splitting text into units that are bigger than characters but smaller than words. The most common approach is **Byte Pair Encoding (BPE)**, originally developed for data compression ([Gage, 1994](#gage1994)) and later adapted for neural machine translation ([Sennrich et al., 2016](#sennrich2016)).

### **How BPE Works**

**Step 1: Start with characters**
Begin with a character-level vocabulary.

**Step 2: Count adjacent pairs**
Find the most frequent pair of tokens in your training data.

Example corpus:
```
"low lower lowest"
```

Character tokens: `["l", "o", "w", " ", "l", "o", "w", "e", "r", " ", ...]`

Most frequent pair: `"l"` + `"o"` (appears in "low", "lower", "lowest")

**Step 3: Merge the pair**
Replace `["l", "o"]` → `["lo"]` everywhere.

Corpus becomes:
```
["lo", "w", " ", "lo", "w", "e", "r", " ", "lo", "w", "e", "s", "t"]
```

**Step 4: Repeat**
Find the next most frequent pair, merge it, repeat until you reach your desired vocabulary size (typically 30,000–50,000 tokens).

After many iterations:
```
"low" → ["low"]
"lower" → ["low", "er"]
"lowest" → ["low", "est"]
```

Notice how the model learned that `"low"` is a common subword, and suffixes like `"er"` and `"est"` are reusable components.

---

## Why This Works

**1. Compact Vocabulary**
Instead of millions of words, you need ~50,000 subword tokens to cover most text.

**2. Shared Representations**
Words with common roots share tokens:
- `"program"` → `["program"]`
- `"programmer"` → `["program", "mer"]`
- `"programming"` → `["program", "ming"]`

The model learns that `"program"` is the root once, then reuses that knowledge.

**3. Handles Novel Words**
Encounter a new word? It breaks into known subwords:
- `"doomscrolling"` → `["doom", "scroll", "ing"]`
- `"cryptocurrency"` → `["cry", "pt", "o", "currency"]`

The model can process it even if it never saw the full word during training.

**4. Multilingual Support**
Subword tokenization adapts to any language:
- Chinese characters become tokens
- German compounds break at meaningful boundaries
- Emoji and special characters get their own tokens

---

## Tokenization in Practice

Let's see how GPT's tokenizer handles real text. GPT-2 and later models use byte-level BPE ([Radford et al., 2019](#radford2019)). To do so, we'll use [tiktoken](https://pypi.org/project/tiktoken/) - OpenAI's open-source Python library for tokenizing text using the same tokenizers that GPT models use. **It's the official way to count tokens before sending requests to the OpenAI API.**

### **Example 1: Simple Sentence**

The first example is a simple sentence ("I love programming") where all the words have been tokenized.

**Python:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

text = "I love programming"
tokens = enc.encode(text)
print(tokens)
# Output: [40, 3021, 15840]

print([enc.decode([t]) for t in tokens])
# Output: ['I', ' love', ' programming']
```

**Notice:**
- `"I"` is one token
- `" love"` includes the leading space (spaces are part of tokens)
- `" programming"` is a single token (common word)

**Total: 3 tokens despite containing 18 characters.**

---

### **Example 2: Rare Word**

In this example we show an extreme opposite by finding the amount of tokens for a word that is rarely used ("supercalifragilisticexpialidocious").

**Python:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

text = "I love supercalifragilisticexpialidocious"
tokens = enc.encode(text)
print([enc.decode([t]) for t in tokens])
# Output: ['I', ' love', ' super', 'cal', 'if', 'rag', 'il', 
# 'istic', 'exp', 'ial', 'id', 'ocious']
```

**The rare word breaks into 10 tokens** because it didn't appear frequently enough in training data to warrant its own token. The model still processes it successfully by combining the meanings of its subword components.

**Total: 12 tokens for just 3 words.**

---

### **Example 3: Code**

Now let's see how the tokens for a small code snippet would be counted.

**Python:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

text = "def hello_world():\n    print('Hello!')"
tokens = enc.encode(text)
print([enc.decode([t]) for t in tokens])
# Output: ['def', ' hello', '_world', '():', '\n', '    ', 
# 'print', "('", 'Hello', "!'", ')']
```

**Code tokenizes differently than prose:**
- Function names split at underscores
- Indentation (spaces) is a separate token
- Parentheses and quotes are individual tokens

**Total: 11 tokens instead of 4 words if you naively split on spaces.**

---

### **Example 4: Multilingual**

Finally, let's look at a multilingual example.

**Python:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

text = "I love 编程"  # "programming" in Chinese
tokens = enc.encode(text)
print([enc.decode([t]) for t in tokens])
# Output: ['I', ' love', ' ', '编', '程']
```

Chinese characters typically get **one token per character** (though common two-character words sometimes merge).

**Total: 5 tokens for 3 words (or 4 words if you count the Chinese characters as one word).**

---

## Why Tokenization Matters

### **1. Token Limits Are Not Word Limits**

GPT-4 has a **128,000 token context window**, but this is **not** 128,000 words. The rough conversion for English is approximately 1 token per 0.75 words, meaning 128k tokens translates to about 96,000 words. 

However, this ratio varies dramatically based on content type. Code tokenizes less efficiently. Our small, but realistic, code examples show that ratio as 1.5-2 tokens per "word." Rare or technical vocabulary breaks into multiple tokens. Non-English text often requires significantly more tokens for the same semantic content. You simply can't predict token count from word count without actually tokenizing the text.


---

### **2. Letter Counting Fails**

When you ask Claude "How many letters are in strawberry?" it might say 10 (correct), 11, or 9, inconsistently. This happens because the model never sees "strawberry" as a sequence of characters. Instead, it sees three tokens: "str", "aw", and "berry". To count letters, it must know that "str" has 3 letters, "aw" has 2, and "berry" has 5, then add them up. This requires **reasoning over tokenization artifacts** rather than directly perceiving the letters. Character-level tasks are inherently difficult because LLMs don't see characters. They see tokens.

**Python:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

text = "strawberry"
tokens = enc.encode(text)
print([enc.decode([t]) for t in tokens])
# Output: [496, 675, 15717]  (tokens: "str", "aw", "berry")
```

---

### **3. Glitch Tokens**

In 2023, researchers discovered **glitch tokens** in GPT-3 that caused erratic behavior ([Rumbelow & Watkins, 2023](#rumbelow2023)). The most infamous was "SolidGoldMagikarp," which in GPT-3's tokenizer encoded as a single token that existed in the vocabulary but never appeared in training data.

**Python:**
```python
import tiktoken

# In GPT-3's tokenizer (not cl100k_base):
# enc.encode("SolidGoldMagikarp")
# Output: [46277]  (single token in GPT-3)

# But in GPT-4's tokenizer:
enc = tiktoken.get_encoding("cl100k_base")

tokens = enc.encode("SolidGoldMagikarp")
print([enc.decode([t]) for t in tokens])
# Output: ['Solid', 'Gold', 'Mag', 'ik', 'arp']
```

This happened because the token came from Reddit usernames scraped during GPT-3's tokenizer training but filtered out during model training. The result was a token ID with no learned representation—just random noise in the embedding space. When users invoked this token in GPT-3, attention patterns broke and generation became incoherent. Other glitch tokens included " petertodd" and " davidjl" (more Reddit usernames) and "\x00" (the null byte). This mismatch between tokenizer training and model training created tokens that literally broke GPT-3's ability to think.

**Note:** These glitch tokens were fixed in GPT-4's tokenizer, which is why "SolidGoldMagikarp" now tokenizes normally into multiple subwords.

---

### **4. Tokenization Affects Performance**

Common words like "the" or "said" use single tokens, making them cheap to process. But rare words like "antidisestablishmentarianism" might require 6 tokens, consuming more context window and computational resources. This creates an inherent bias where the model is more efficient at processing common vocabulary, while technical or specialized terms are expensive. Since API costs depend on token count rather than word count, verbose or technical text costs more to process than simple prose using common words.

The following Python code snippet illustrates that common words are cheap while rare words are expensive:

**Python:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Common words token count
text = "the"
tokens = enc.encode(text)  # 1 token
print([enc.decode([t]) for t in tokens])

text = "said"
tokens = enc.encode(text)  # 1 token
print([enc.decode([t]) for t in tokens])

# Rare word token count
text = "antidisestablishmentarianism"
tokens = enc.encode(text)  # 6 tokens
print([enc.decode([t]) for t in tokens])
```

---

## Byte-Level BPE (What GPT Uses)

GPT-4 uses **Byte-level BPE**, operating on UTF-8 bytes rather than Unicode characters ([Radford et al., 2019](#radford2019)). This means the base vocabulary consists of 256 possible byte values, with learned merges on top.

The advantage is universality—byte-level BPE can represent any text in any language or encoding without modification. The vocabulary size is inherently bounded (256 base bytes plus learned merges), and no special "unknown" token is needed since any text can be decomposed into bytes.

The tradeoff is that common characters might split into multiple tokens when they require multiple UTF-8 bytes to encode. For example, "café" tokenizes into two tokens:

**Python:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

text = "café"
tokens = enc.encode(text)
print([enc.decode([t]) for t in tokens])
# Output: ['ca', 'fé']
```

The accented é gets grouped with the f to form "fé" as a single token, showing how byte-level BPE creates tokens based on frequency patterns in the training data rather than obvious character boundaries.

---

## Tokenization in Other Models

Different model families use different tokenization strategies, each with its own tradeoffs.

### **GPT-2/GPT-3/GPT-4:**
- Byte-level BPE ([Radford et al., 2019](#radford2019))
- ~50,000 token vocabulary

### **BERT:**
- WordPiece (similar to BPE)
- ~30,000 token vocabulary
- Adds special tokens: `[CLS]`, `[SEP]`, `[MASK]`

### **T5:**
- SentencePiece ([Kudo & Richardson, 2018](#kudo2018))
- ~32,000 token vocabulary

### **LLaMA/Mistral:**
- SentencePiece ([Kudo & Richardson, 2018](#kudo2018))
- ~32,000–64,000 tokens

The implication is that tokenizers are not interchangeable. A prompt that uses 100 tokens in GPT-4 might be 110 tokens in Claude or 95 tokens in LLaMA. This affects not just API costs but also how much information fits in each model’s context window.

---

## Visualizing Tokenization

Here's a Python script to visualize how text tokenizes (using OpenAI's tokenizer tool as reference, [OpenAI, 2023](#openai2023)):

**Python:**
```python
import tiktoken

def visualize_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    print(f"Text: {text}")
    print(f"Tokens: {len(tokens)}\n")
    
    for i, token in enumerate(tokens):
        decoded = enc.decode([token])
        print(f"Token {i}: '{decoded}' (ID: {token})")

# Examples
visualize_tokens("I love programming")
print("\n" + "="*50 + "\n")
visualize_tokens("The quick brown fox jumps over the lazy dog")
print("\n" + "="*50 + "\n")
visualize_tokens(
    "def factorial(n):\n    "
    "return 1 if n == 0 else n * factorial(n-1)"
)
```

**Output:**
```
Text: I love programming
Tokens: 3

Token 0: 'I' (ID: 40)
Token 1: ' love' (ID: 3021)
Token 2: ' programming' (ID: 15840)

==================================================

Text: The quick brown fox jumps over the lazy dog
Tokens: 9

Token 0: 'The' (ID: 791)
Token 1: ' quick' (ID: 4062)
Token 2: ' brown' (ID: 14198)
Token 3: ' fox' (ID: 39935)
Token 4: ' jumps' (ID: 35308)
Token 5: ' over' (ID: 927)
Token 6: ' the' (ID: 279)
Token 7: ' lazy' (ID: 16053)
Token 8: ' dog' (ID: 5679)

==================================================

Text: def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)
Tokens: 21

Token 0: 'def' (ID: 755)
Token 1: ' factorial' (ID: 54062)
Token 2: '(n' (ID: 1471)
Token 3: '):
' (ID: 997)
Token 4: '   ' (ID: 262)
Token 5: ' return' (ID: 471)
Token 6: ' ' (ID: 220)
Token 7: '1' (ID: 16)
Token 8: ' if' (ID: 422)
Token 9: ' n' (ID: 308)
Token 10: ' ==' (ID: 624)
Token 11: ' ' (ID: 220)
Token 12: '0' (ID: 15)
Token 13: ' else' (ID: 775)
Token 14: ' n' (ID: 308)
Token 15: ' *' (ID: 353)
Token 16: ' factorial' (ID: 54062)
Token 17: '(n' (ID: 1471)
Token 18: '-' (ID: 12)
Token 19: '1' (ID: 16)
Token 20: ')' (ID: 8)
```

---

## Practical Implications

### **1. Counting Tokens Before API Calls**

Since OpenAI charges by token rather than by word, it’s essential to count tokens before sending requests. A simple utility function can save significant costs:

**Python:**
```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

prompt = "Explain quantum computing in simple terms"
tokens = count_tokens(prompt)
print(f"This prompt costs {tokens} tokens")
```

---

### **2. Optimizing Prompts for Token Efficiency**

Small changes in phrasing can dramatically affect token count. The verbose request “Please provide a detailed explanation of the concept” uses 12 tokens, while the concise “Explain the concept” uses just 5 tokens—same meaning, 58% fewer tokens. Over thousands of API calls, this optimization translates to substantial cost savings.

**Verbose version (12 tokens):**
```
"Please provide a detailed explanation of the concept"
```

**Concise version (5 tokens):**
```
"Explain the concept"
```

---

### **3. Understanding Model Failures**

When an LLM fails at a seemingly simple task, tokenization is often the culprit. Tasks like counting letters, spelling words backward, or comparing character positions all suffer from tokenization mismatch. The model isn’t inherently limited; it’s blind to information below the token level. Understanding this helps set appropriate expectations and design better prompts.

---

## The Future of Tokenization

### **Problems with Current Approaches**

Current tokenization methods have several fundamental limitations that constrain model capabilities.

Tokenization is brittle, with glitch tokens causing unpredictable failures ([Rumbelow & Watkins, 2023](#rumbelow2023)). Similar words might split differently, and adding new vocabulary requires retraining the tokenizer from scratch. The system is also language-biased in that English tokenizes efficiently while other languages often require significantly more tokens for equivalent content. Code tokenizes particularly poorly, especially when mixing natural language with programming languages. Perhaps most fundamentally, tokenization hides character-level information, making spelling, rhyming, and phonetic tasks unnecessarily difficult.

### **Emerging Alternatives**

Several promising approaches could replace traditional tokenization in future models.

**Character-level models** are becoming feasible with improved architectures like Mamba and RWKV that handle long sequences more efficiently. These models could operate on raw bytes without any tokenization step, eliminating artifacts and providing perfect character-level access, though sequence length remains a challenge.

**Learned tokenization** represents another path forward. Instead of fixing token boundaries during preprocessing, models could learn optimal segmentation during training. Google's ByT5 demonstrates this approach, operating on raw bytes and learning internal segmentation dynamically ([Xue et al., 2022](#xue2022)).

**Multimodal tokenization** points toward unified representations across modalities. Just as GPT-4V tokenizes images into patches and audio models tokenize sound into frames, future models might develop universal tokenization schemes that work across text, images, audio, and other modalities seamlessly.

---

## Closing Thoughts

Tokenization is the invisible layer between human language and machine learning. It's not a detail—it's a fundamental design choice that shapes what models can and cannot do.

When you understand tokenization:
- You write better prompts (token-efficient, avoiding splits)
- You debug model failures (recognizing tokenization-induced blindness)
- You estimate costs accurately (tokens ≠ words)
- You anticipate edge cases (glitch tokens, rare words)

**The next time an LLM fails at a seemingly simple task, ask yourself: what did the model *actually see*?** The answer is usually: not what you think.

---

**If you've followed this series from [how LLMs read code](/posts/how-large-language-models-read-code) through [how they think](/posts/how-large-language-models-think), [how they learn](/posts/how-large-language-models-learn), and [how they handle context](/posts/how-large-language-models-handle-context-windows), you now understand the complete pipeline: text becomes tokens, tokens become vectors, vectors become predictions, and predictions become learning. What emerges is a system that's both more powerful and more limited than it appears—and understanding those limits is the key to using these tools effectively.**

---

## Try It Yourself

[Download the full code on GitHub](https://github.com/TomArcher/technical-blog-examples/tree/main/how-large-language-models-tokenize-text)

---

## Further Reading

If you want to dive deeper into tokenization and its implications:

<a id="gage1994"></a>
- Gage, P. (1994). [A New Algorithm for Data Compression](https://www.derczynski.com/papers/archive/BPE_Gage.pdf). *C Users Journal*, 12(2).  
  *The original Byte Pair Encoding paper from 1994, predating its use in NLP by two decades. Shows how compression algorithms became NLP tools.*

<a id="kudo2018"></a>
- Kudo, T., & Richardson, J. (2018). [SentencePiece: A simple and language independent approach to subword tokenization](https://arxiv.org/abs/1808.06226). *Proceedings of EMNLP 2018*.  
  *Introduces SentencePiece, a language-agnostic tokenization library used by T5, LLaMA, and many modern models.*

<a id="openai2023"></a>
- OpenAI. (2023). [Tokenizer Tool](https://platform.openai.com/tokenizer).  
  *Interactive web tool for visualizing how GPT models tokenize text. Essential for prompt engineering.*

<a id="radford2019"></a>
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). *OpenAI Technical Report*.  
  *The GPT-2 paper, which popularized byte-level BPE. See Section 2.2 for tokenization details.*

<a id="rumbelow2023"></a>
- Rumbelow, J., & Watkins, M. (2023). [SolidGoldMagikarp and other Glitch Tokens](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation). *LessWrong*.  
  *Community-driven investigation into glitch tokens in GPT-3. Fascinating deep dive into tokenization artifacts.*

<a id="sennrich2016"></a>
- Sennrich, R., Haddow, B., & Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). *Proceedings of ACL 2016*.  
  *The foundational paper introducing BPE for neural networks. Essential reading for understanding why subword tokenization works.*

<a id="xue2022"></a>
- Xue, L., Barua, A., Constant, N., et al. (2022). [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626). *Transactions of ACL 2022*.  
  *Google's exploration of byte-level models without explicit tokenization. Points toward future architectures.*

---
