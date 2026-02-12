# Product Requirements Document (PRD)
## Portfolio AI Chatbot

**Version:** 1.0  
**Author:** Imrane Naitembark  
**Last Updated:** February 2025

---

## 1. Overview

### 1.1 Product Summary
An AI-powered chatbot for the personal portfolio website that helps visitors learn about projects, skills, games, and contact information. Supports both real LLM (Ollama/OpenAI) and pure Python ML (TF-IDF, BM25) for intelligent responses.

### 1.2 Target Users
- Portfolio visitors exploring the site
- Recruiters and employers
- Fellow developers and collaborators

---

## 2. Goals & Objectives

| Goal | Description |
|------|-------------|
| **Engagement** | Provide an interactive way for visitors to explore the portfolio |
| **Information** | Answer questions about projects, skills, games, and contact |
| **Showcase** | Demonstrate AI/ML capabilities (TF-IDF, BM25, N-grams) |
| **Flexibility** | Support both offline ML and optional real AI (Ollama/OpenAI) |

---

## 3. Features

### 3.1 Core Features

| Feature | Priority | Description |
|---------|----------|-------------|
| **ML Response Matching** | P0 | TF-IDF + N-grams + BM25 ensemble for intelligent Q&A matching |
| **Knowledge Base** | P0 | 11 built-in topics (greetings, projects, skills, AI/ML, contact, etc.) |
| **Online Learning** | P0 | `/learn` command to teach new Q&A pairs at runtime |
| **Confidence Scoring** | P1 | Display match confidence (0.0–1.0) for each response |
| **Conversation History** | P1 | `/history` to view past exchanges |
| **Statistics** | P1 | `/stats` for ML metrics (match rate, vocabulary size) |

### 3.2 Real AI (Optional)

| Feature | Priority | Description |
|---------|----------|-------------|
| **Ollama Integration** | P1 | Free local LLM via `--ollama` flag (requires Ollama installed) |
| **OpenAI Integration** | P2 | GPT via `--openai` (requires `OPENAI_API_KEY`) |
| **Fallback to ML** | P0 | Automatic fallback to TF-IDF/BM25 when Real AI unavailable |

### 3.3 Commands

| Command | Description |
|---------|-------------|
| `/help` | Show command help |
| `/stats` | ML statistics and metrics |
| `/history` | Conversation history |
| `/learn` | Teach a new response |
| `/clear` | Clear conversation history |
| `/about` | About the chatbot |
| `/quit` | Exit |

---

## 4. Technical Requirements

### 4.1 Tech Stack
- **Language:** Python 3.8+
- **ML (built-in):** TF-IDF, N-grams, BM25, Cosine Similarity (pure Python, no deps)
- **Real AI (optional):** `ollama` package, `openai` package

### 4.2 Machine Learning Components
- **TF-IDF:** Term frequency–inverse document frequency vectorization
- **N-grams:** Unigrams + bigrams for richer semantic features
- **BM25:** Search engine ranking algorithm
- **Ensemble:** 70% Cosine Similarity + 30% BM25 for final score
- **Tokenization:** Lowercase, regex tokenization, stopword removal

### 4.3 Performance
- Responses within ~100ms for ML path (no network)
- Real AI latency depends on Ollama/OpenAI

---

## 5. User Stories

1. **As a visitor**, I want to ask about projects and get accurate answers.
2. **As a visitor**, I want to learn how to contact the portfolio owner.
3. **As a visitor**, I want to know what games are available.
4. **As a developer**, I want to see ML stats (vocabulary, confidence).
5. **As a developer**, I want to teach the bot new responses via `/learn`.
6. **As a user**, I want to use real AI (Ollama) when available for richer answers.

---

## 6. Non-Functional Requirements

- **No build step:** Run directly with `python chatbot.py`
- **Portability:** Works on Windows, macOS, Linux
- **Privacy:** ML mode sends no data externally
- **Extensibility:** Knowledge base and learned responses can grow over time

---

## 7. Future Considerations

- [ ] Load knowledge from external file (e.g., `knowledge.txt`)
- [ ] Save learned responses to disk for persistence
- [ ] Web interface (browser-based chat)
- [ ] Multi-language support
