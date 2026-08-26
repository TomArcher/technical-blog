+++
date = '2026-02-09T06:00:00-07:00'
draft = false
title = "How Large Language Models (LLMs) Know Things They Were Never Taught"
subtitle = "Web search, RAG, and the illusion of current knowledge"
categories = ["AI and the Mathematics of Language"]
tags = ["AI", "LLM", "RAG", "web search", "context windows"]
author = "Tom Archer"
listThumb = "how-large-language-models-know-things-they-were-never-taught-thumb.png"

hero = "how-large-language-models-know-things-they-were-never-taught.png"
heroAlt = "A neural network with frozen weights receiving current information from external sources"
heroLabel = "Open full-size how large language models know things they were never taught"
heroCaption = "The model didn't learn this. It just read it."

whatYoullLearn = [
    "Why an LLM can answer questions about events that happened after its training ended",
    "How web search gives a model current information without changing its weights",
    "How RAG retrieves relevant information from private or specialized document collections",
    "Why retrieved information is read during inference rather than learned by the model",
    "How tool use lets an LLM work with search engines, code, databases, and external APIs",
    "Why retrieval quality, source accuracy, and model reasoning all affect the final answer"
]
+++

When you ask an {{< term "large-language-model" "LLM" >}} *without* web search enabled a question like "What happened in the news this morning?", the LLM will respond by telling you that it doesn't have access to current events and suggest you check a more current news source such as Reuters or Google News.

Conversely, ask an LLM *with* web search enabled the same question, and you receive a detailed rundown of breaking stories, political controversies, and sports news from the past 24 hours.

Identical question. Same underlying technology. Completely different answers. The difference isn't that one model is smarter or more current than the other. The difference is whether web search was triggered.

But why does that matter? Both models were trained months ago. Their internal knowledge stopped updating the moment training ended. So how does flipping a switch allow one model to suddenly "know" what happened this morning? The answer reveals a fundamental distinction most users never consider: the difference between what a model *learned* and what a model *read*.

LLMs don't update their weights (the billions of numerical parameters that encode everything learned during training) when you chat with them. They don't learn from your conversations. But they can access external information and reason over it within their context window. This isn't learning; it's reading. And understanding that difference changes how you think about what these systems can and cannot do.


This post explores how modern LLMs access information beyond their training data through web search, {{< term "retrieval-augmented-generation" "retrieval-augmented generation (RAG)" >}}, and {{< term "tool-use" "tool use" >}}. You'll see how these mechanisms work, why they're fundamentally different from learning, and what that means for the reliability and limitations of AI systems. If you haven't read my post on [how LLMs handle context windows](/posts/how-large-language-models-handle-context-windows/), start there; everything in this post builds on that foundation.

> **TL;DR:**
> 
> * LLM weights are frozen after training; chatting with them doesn't update their knowledge
> * The _Web search_ mechanism injects retrieved text into the context window before generation
> * RAG (Retrieval-Augmented Generation) does the same thing with private document stores
> * The model reasons over the injected text using its frozen capabilities
> * This is pattern matching over dynamic context, not learning
> * Retrieval quality determines answer quality; the model can't fix bad retrieval
> * Citations and source attribution come from the retrieved text, not model memory
> * Understanding this distinction helps you use these tools more effectively

* * *

## The Frozen Weights Problem

In my post on [how LLMs learn](/posts/how-large-language-models-learn/), I explained how {{< term "gradient-descent" "gradient descent" >}} adjusts model weights during training. Each weight update encodes a tiny piece of learned pattern, and after billions of updates, the model captures the statistical structure of language.

But here's the critical point: **that process stops when training ends.**

When you interact with any LLM, you're using a model with completely fixed weights. The parameters that determine how the model processes text haven't changed since the training run finished. Your conversations don't update them. Your corrections don't update them. Nothing you do updates them.

This creates an obvious problem. The model's knowledge has a cutoff date. Anything that happened after training is invisible to the weights. Ask about an event from last week, and the model's internal knowledge simply doesn't include it.

```python
# This is what the model "knows" from training
training_knowledge = {
    "events_before_cutoff": True,   # Encoded in weights
    "events_after_cutoff": False,   # Not in weights
    "your_conversations": False,    # Never encoded
    "corrections_you_made": False   # Never encoded
}
```

So how do models answer questions about current events?

* * *

## Retrieval: Reading Instead of Knowing

The solution is elegantly simple: instead of updating the model's weights, inject relevant information into its context window.

Recall from my [context windows post](/posts/how-large-language-models-handle-context-windows/) that every API call is {{< term "stateless" "stateless" >}}. The model processes whatever text you send it, generates a response, and forgets everything. There's no persistent memory between calls.

This same mechanism enables retrieval. Before the model generates a response, an external system:

1. Takes your question
1. Searches for relevant information (web pages, documents, databases)
1. Retrieves the most relevant results
1. Injects those results into the context window
1. Sends the augmented prompt to the model

The model then reasons over both your question and the retrieved information, producing an answer grounded in text it just "read" rather than knowledge it "learned."

```python
# Simplified retrieval flow
def answer_with_retrieval(user_question, search_function, llm):
    # Step 1: Search for relevant information
    search_results = search_function(user_question)

    # Step 2: Build augmented prompt
    augmented_prompt = f"""
    Use the following information to answer the question.

    Retrieved Information:
    {search_results}

    Question: {user_question}

    Answer based on the retrieved information:
    """

    # Step 3: Model reasons over the augmented context
    response = llm.generate(augmented_prompt)

    return response
```

This is the core insight: **the model's capabilities are frozen, but its inputs are dynamic.** A model trained in 2023 can answer questions about 2025 events because those events are injected as text, not because the model learned about them.

* * *

## Web Search: Real-Time Information Retrieval

When you ask an LLM with web search enabled a question about current events, something like this happens behind the scenes:

1. **Query Generation:** The model (or a specialized component) converts your question into one or more search queries. "Who won the Super Bowl last week?" might become the search query `Super Bowl 2026 winner`.

1. **Web Search Execution:** An external search system (not the LLM itself) queries the web and retrieves relevant pages. This is standard information retrieval, similar to what happens when you use a search engine such as Google or DuckDuckGo.

1. **Content Extraction:** The retrieved web pages are processed to extract relevant text. Full HTML pages are too long and noisy, so systems extract article text, key facts, or relevant snippets.

1. **Context Injection:** The extracted content is formatted and injected into the prompt sent to the LLM. The model now "sees" current information as part of its input.

1. **Grounded Generation:** The LLM generates a response based on both your question and the retrieved content. Citations typically reference the retrieved sources.

```Python
    # Conceptual web search augmentation
    def web_search_augmented_response(user_query, llm):
        # Generate search queries from user question
        search_queries = generate_search_queries(user_query)
    
        # Execute web searches
        web_results = []
        for query in search_queries:
            results = web_search(query)
            web_results.extend(results)
    
        # Extract and deduplicate relevant content
        extracted_content = extract_relevant_text(web_results)
    
        # Format for injection
        context = format_search_results(extracted_content)
    
        # Build the augmented prompt
        prompt = f"""
        You have access to the following web search results:
    
        {context}
    
        Based on these search results, answer the following question.
        Cite your sources.
    
        Question: {user_query}
        """
    
        # Generate response
        return llm.generate(prompt)
```

The model itself never "searched" anything. It received search results as text and reasoned over them. The intelligence is in how the model processes and synthesizes the retrieved information, but the information itself came from outside.

* * *

## RAG: Retrieval-Augmented Generation

RAG applies the same principle to private or specialized document collections. Instead of searching the web, RAG systems search a curated knowledge base.

The canonical RAG pipeline ([Lewis et al., 2020](#lewis2020)) consists of:

1. **Document Chunking:** Large documents are split into smaller, semantically meaningful chunks. A 50-page PDF might become 200 chunks of roughly 500 tokens each.

1. **{{< term "embedding" "Embedding" >}} Generation:** Each chunk is converted to a dense vector (embedding) using a model like OpenAI's `text-embedding-ada-002` or open-source alternatives. These embeddings capture semantic meaning; similar content produces similar vectors.

1. **Vector Storage:** Embeddings are stored in a {{< term "vector-database" "vector database" >}} (Pinecone, Weaviate, Chroma, FAISS) that enables fast similarity search.

1. **Query Embedding:** When a user asks a question, the question is also embedded into the same vector space.

1. **Similarity Search:** The system finds the document chunks whose embeddings are most similar to the query embedding. This typically uses {{< term "cosine-similarity" "cosine similarity" >}} or {{< term "approximate-nearest-neighbor" "approximate nearest neighbor search" >}}.

1. **Context Construction:** The {{< term "top-k" "top-k" >}} most relevant chunks are retrieved and formatted as context for the LLM.

1. **Augmented Generation:** The LLM receives both the user's question and the retrieved context, generating a response grounded in the retrieved documents.

```python
    import numpy as np
    from typing import List, Tuple
    
    class SimpleRAG:
        def __init__(self, embedding_model, llm):
            self.embedding_model = embedding_model
            self.llm = llm
            self.documents = []
            self.embeddings = []
    
        def add_documents(self, documents: List[str]):
            """Chunk and embed documents."""
            for doc in documents:
                chunks = self.chunk_document(doc)
                for chunk in chunks:
                    embedding = self.embedding_model.embed(chunk)
                    self.documents.append(chunk)
                    self.embeddings.append(embedding)
    
            self.embeddings = np.array(self.embeddings)
    
        def chunk_document(self, doc: str, chunk_size: int = 500) -> List[str]:
            """Split document into chunks."""
            words = doc.split()
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
            return chunks
    
        def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
            """Find most relevant chunks for a query."""
            query_embedding = self.embedding_model.embed(query)
    
            # Cosine similarity
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * 
                np.linalg.norm(query_embedding)
            )
    
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
    
            return [(self.documents[i], similarities[i]) for i in top_indices]
    
        def query(self, user_question: str) -> str:
            """Answer a question using RAG."""
            # Retrieve relevant chunks
            retrieved = self.retrieve(user_question)
    
            # Format context
            context = "\n\n".join([
                f"[Relevance: {score:.2f}]\n{doc}" 
                for doc, score in retrieved
            ])
    
            # Build prompt
            prompt = f"""
            Use the following retrieved documents to answer the question.
            If the documents don't contain relevant information, say so.
    
            Retrieved Documents:
            {context}
    
            Question: {user_question}
    
            Answer:
            """
    
            return self.llm.generate(prompt)
```

* * *

## The Embedding Space Connection

RAG relies heavily on the embedding concepts I covered in [How LLMs Think](/posts/how-large-language-models-think/). The same geometric principles apply:

**Semantic Similarity as Distance:** Documents about similar topics cluster together in embedding space. When you search for "quarterly revenue projections," the retrieval system finds chunks that are geometrically close to that query, even if they use different words like "Q3 financial forecasts."

**The Curse of Vocabulary Mismatch:** Traditional keyword search fails when queries and documents use different terminology. Embedding-based retrieval handles this gracefully because semantically similar content maps to nearby vectors regardless of exact wording.

**Relevance vs. Correctness:** Here's a critical limitation: embedding similarity measures *relevance*, not *correctness*. A document chunk might be highly relevant to your question while containing outdated or incorrect information. The retrieval system has no way to verify factual accuracy; it only measures semantic similarity.

```python
# Embedding similarity doesn't guarantee correctness
query = "What is the population of Tokyo?"

# Both chunks might have high similarity scores
chunk_a = "Tokyo's population is approximately 14 million."  # Correct
chunk_b = "Tokyo's population exceeds 50 million people."    # Incorrect

# The retrieval system can't tell which is accurate
# Both are semantically relevant to the query
```

* * *

## What the Model Actually Does

Once relevant content is retrieved and injected, what does the model do with it? This is where the frozen capabilities come into play.

**Pattern Matching Over Retrieved Text:** The model applies the same {{< term "attention" "attention mechanisms" >}} and pattern recognition it uses for any text. It doesn't "know" the retrieved content is special; it simply processes it as part of the input sequence.

**Synthesis and Reasoning:** The model can combine information from multiple retrieved chunks, identify contradictions, extract specific facts, and generate coherent summaries. These capabilities were learned during training; retrieval just provides new material to apply them to.

**Citation Generation:** When models cite sources, they're typically extracting attribution information from the retrieved text itself. The citation "according to the New York Times" appears because that phrase (or the source URL) was present in the retrieved content.

**No Fact Verification:** The model cannot independently verify whether retrieved information is accurate. If the retrieval returns outdated or incorrect content, the model will confidently present that information as fact. Garbage in, garbage out.

```python
# The model processes retrieved content like any other text
def model_processing(prompt_with_retrieved_content):
    """
    The model doesn't treat retrieved content specially.
    It applies the same attention and generation mechanisms.
    """

    # Tokenize everything
    tokens = tokenize(prompt_with_retrieved_content)

    # Apply attention across all tokens
    # Retrieved content participates in attention like any other text
    for layer in transformer_layers:
        tokens = layer.attention(tokens)
        tokens = layer.feedforward(tokens)

    # Generate response token by token
    response = []
    while not done:
        next_token = sample_next_token(tokens)
        response.append(next_token)
        tokens = append(tokens, next_token)

    return response
```

* * *

## Why This Isn't Learning

The distinction between retrieval and learning is fundamental:

| Aspect | Learning (Training) | Retrieval (Inference) |
| --- | --- | --- |
| Weights | Updated via gradient descent | Completely frozen |
| Duration | Persists permanently | Exists only for one request |
| Scope | Affects all future responses | Affects only current response |
| Cost | Enormous (millions of dollars) | Minimal (milliseconds) |
| Information | Compressed into parameters | Present as literal text |

When a model learns something during training, that knowledge becomes part of its weights. It's compressed, abstracted, and available for all future interactions. The model doesn't need to "look up" that information; it's encoded in the parameters.

When a model retrieves something at {{< term "inference" "inference" >}} time, that knowledge exists only as text in the current context window. It's not compressed or abstracted. The model reads it, reasons over it, and then forgets it completely when the conversation ends.

This is why retrieval-augmented models can "know" current events while still having a knowledge cutoff. The cutoff applies to the weights; retrieval has no cutoff because it accesses live information.

```python
# Learning vs Retrieval
class LearningVsRetrieval:
    def __init__(self):
        # Weights encode learned knowledge (frozen at inference)
        self.weights = train_on_data_before_cutoff()

        # Retrieval provides dynamic knowledge
        self.retrieval_system = SearchEngine()

    def answer_from_weights(self, question):
        """Limited to training data cutoff."""
        return self.forward_pass(question)  # Uses frozen weights

    def answer_with_retrieval(self, question):
        """Can access current information."""
        retrieved = self.retrieval_system.search(question)
        augmented = f"{retrieved}\n\nQuestion: {question}"
        return self.forward_pass(augmented)  # Same frozen weights
```

* * *

## Tool Use: Generalizing Beyond Text

Web search and RAG are specific instances of a broader capability: **tool use**. Modern LLMs can interact with external systems to gather information or perform actions.

The pattern is consistent:

1. The model generates a structured request (search query, API call, code to execute)
1. An external system executes the request
1. Results are returned to the model as text
1. The model reasons over the results

**Examples of Tool Use:**

* **Code Execution**: Model generates Python code, a sandbox executes it, output returns as text
* **Calculator**: Model requests a calculation, external system computes it precisely
* **Database Query**: Model generates SQL, database executes it, results return as text
* **API Calls**: Model generates API requests, external systems execute them, responses return as text

```python
# Tool use follows the same retrieval pattern
def tool_augmented_response(user_query, llm, tools):
    # Model decides which tool to use and how
    tool_call = llm.generate_tool_call(user_query, available_tools=tools)

    # External system executes the tool
    tool_result = execute_tool(tool_call)

    # Results injected back into context
    augmented_prompt = f"""
    Tool Used: {tool_call.tool_name}
    Tool Input: {tool_call.input}
    Tool Output: {tool_result}

    Original Question: {user_query}

    Based on the tool output, provide your answer:
    """

    return llm.generate(augmented_prompt)
```

The model's role is generating appropriate tool calls and reasoning over results. The actual computation or retrieval happens externally. This separation is important: the model can't execute code directly or access databases directly. It can only request that external systems do so.

* * *

## Limitations of Retrieval-Augmented Systems

Understanding how retrieval works reveals its limitations:

### Retrieval Quality Bounds Answer Quality

If the retrieval system returns irrelevant or incorrect documents, the model can't recover. It has no independent way to verify information. A confidently wrong source produces a confidently wrong answer.

### Context Window Constraints

Retrieved content consumes context window space. You can't inject an entire Wikipedia into the prompt. Systems must balance breadth (more sources) against depth (more content per source) within fixed token limits.

### Latency Costs

Every retrieval step adds latency. Web search takes time. Vector database queries take time. For real-time applications, these delays accumulate.

### Retrieval Doesn't Fix Reasoning Failures

If a model struggles with a particular type of reasoning (say, multi-step math), retrieval won't help. You can retrieve a textbook chapter on algebra, but the model still needs to apply that knowledge correctly.

### Hallucination Persists

Models can still {{< term "hallucination" "hallucinate" >}} even with retrieved content. They might misread the retrieved text, combine information incorrectly, or generate details not present in any source.

```python
# Retrieval doesn't prevent all failure modes
def retrieval_limitations():
    # Problem 1: Bad retrieval
    query = "Latest election results"
    retrieved = search(query)  # Returns outdated 2020 results
    # Model will confidently report wrong information

    # Problem 2: Correct retrieval, bad reasoning
    query = "What is 15% of the revenue mentioned in this report?"
    retrieved = "Q3 revenue was $4.2 million"  # Correct retrieval
    # Model might still compute 15% incorrectly

    # Problem 3: Hallucination despite retrieval
    query = "What did the CEO say about expansion?"
    retrieved = "The CEO discussed growth strategies"  # Vague
    # Model might invent specific quotes not in the source
```

* * *

## Practical Implications

Understanding the retrieval mechanism helps you use these systems more effectively:

### Trust Retrieved Information Cautiously

Just because a model cites a source doesn't mean it interpreted that source correctly. Verify important claims by checking the original source.

### Understand Citation Limitations

Citations reflect what was retrieved, not what the model independently verified. A citation to a reputable source doesn't guarantee the model's interpretation is accurate.

### Recognize Fresh vs. Learned Knowledge

When a model answers a question about current events, that knowledge is ephemeral. Ask the same question in a new conversation, and the model might retrieve different sources and give a different answer.

### Design Prompts Accordingly

For questions where retrieval is active, you can guide what gets searched by being specific. "According to recent news reports" vs. "According to scientific studies" might trigger different search strategies.

### Don't Expect Persistence

Information from retrieved sources doesn't persist between conversations. Each new chat starts fresh, requiring new retrieval for current information.

* * *

## The Architecture of Augmented LLMs

Modern AI assistants combine multiple components. The LLM itself is just one component. The {{< term "orchestration-layer" "orchestration layer" >}} decides when to retrieve, what to retrieve, and how to present retrieved information. This is where much of the system intelligence actually lives.

{{< lightbox
    src="augmented-llm-architecture.png"
    alt="Architecture diagram showing a user interface feeding an orchestration layer, which routes requests to web search, RAG, and external tools before passing augmented context to an LLM with frozen weights"
    label="Open full-size augmented LLM architecture diagram"
    caption="Modern AI assistants combine an orchestration layer with web search, retrieval-augmented generation, and external tools before passing augmented context to the underlying LLM."
>}}

* * *

## Closing Thoughts

The ability for LLMs to access current information through retrieval is genuinely useful. It extends frozen models far beyond their training data cutoffs and enables grounded responses that cite real sources.

But it's crucial to understand what's actually happening. The model isn't learning about current events. It isn't updating its knowledge. It's reading dynamically retrieved text and applying its frozen capabilities to that text.

This distinction matters for trust calibration. A model answering from its trained weights has seen that information millions of times across many sources. A model answering from retrieval is basing its response on whatever happened to be retrieved for that specific query. The reliability profiles are different.

It also matters for understanding failures. When a retrieval-augmented model gives a wrong answer about current events, the problem might be the retrieval (wrong sources), the interpretation (model misread the source), or both. Debugging requires understanding where in the pipeline the failure occurred.

After working with these systems extensively, I've come to see retrieval augmentation as giving models a form of open-book exam capability. They can look things up, but they can't learn during the exam. Their reasoning abilities are fixed; only their inputs are dynamic. Understanding that constraint is key to using these tools effectively.

* * *

**If you've followed this series from [how LLMs read code](/posts/how-large-language-models-read-code) through [how they think](/posts/how-large-language-models-think), [how they learn](/posts/how-large-language-models-learn), [how they handle context](/posts/how-large-language-models-handle-context-windows), [how they tokenize](/posts/how-large-language-models-tokenize-text), and [how sampling works](/posts/temperature-top-p-creativity-knobs), you now have the complete picture: from raw text to tokens to embeddings to attention to learning to generation to augmentation. These systems are remarkable for what they can do, and understanding their mechanisms helps you appreciate both their power and their limitations.**

* * *

## Further Reading

These resources provide deeper technical details on retrieval-augmented generation and related techniques.

<a id="guu2020"></a>Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. W. (2020). REALM: Retrieval-augmented language model pre-training. *Proceedings of the 37th International Conference on Machine Learning*. https://arxiv.org/abs/2002.08909

<a id="izacard2022"></a>Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., ... & Grave, E. (2022). Atlas: Few-shot learning with retrieval augmented language models. *arXiv preprint arXiv:2208.03299*. https://arxiv.org/abs/2208.03299

<a id="karpukhin2020"></a>Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*. https://arxiv.org/abs/2004.04906

<a id="lewis2020"></a>Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33. https://arxiv.org/abs/2005.11401

<a id="mialon2023"></a>Mialon, G., Dessì, R., Lomeli, M., Nalmpantis, C., Pasunuru, R., Raileanu, R., ... & Scialom, T. (2023). Augmented language models: A survey. *arXiv preprint arXiv:2302.07842*. https://arxiv.org/abs/2302.07842

<a id="schick2023"></a>Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., ... & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. *arXiv preprint arXiv:2302.04761*. https://arxiv.org/abs/2302.04761

* * *