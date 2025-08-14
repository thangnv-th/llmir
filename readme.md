# 🏆 Retrieval-Augmented Generation with Reward Optimization

This framework enhances traditional retrieval-augmented generation by introducing a **learned reward model** that supervises retrieval quality based on human-aligned feedback. Instead of treating retrieval as a static or heuristic process, Reward-RAG **optimizes the retriever to produce contextually helpful documents** that lead to better generated answers.

---

## 🎯 Objective

Move beyond top-k retrieval by training a **reward model** to rank or score retrieved documents based on their utility to the final output.

- Guide the retriever using feedback signals (human preference, score proxies)
- Fine-tune retrieval embeddings to match helpfulness signals
- Maintain plug-and-play compatibility with existing RAG pipelines

---

## 🔑 Key Features

- **Reward-driven supervision**: Uses a trainable reward function to evaluate document helpfulness
- **Retriever tuning**: Backpropagates reward feedback to update embedding space
- **Plug-and-play**: Works with any retriever (BM25, dense, hybrid)
- **Alignment-aware**: Encourages grounded and answer-relevant retrievals

---

## 🛠️ System Components

1. **Retriever**: Fetches top-k documents given a query
2. **Reward Model**: Scores each document given query + answer
3. **Gradient Feedback**: Updates retriever based on reward signal
4. **Generator (LLM)**: Generates answers from selected context

---

## 📦 Installation

```bash
cd reward-rag
pip install -r requirements.txt
```

Recommended setup includes: PyTorch, SentenceTransformers, OpenAI API, HuggingFace Transformers.

---
