# flowmcp Documentation

flowmcp is a lightweight RAG + MCP + LLM pipeline framework for Python and Java.

## Installation

```bash
pip install flowmcp[openai]
```

## Quick Start

```python
from flowmcp import Pipeline

pipeline = Pipeline(llm="openai/gpt-4o-mini", docs="./my-docs")
answer = pipeline.ask("What is the refund policy?")
print(answer)
```

## Supported LLM Providers

| Provider | Install | Example |
|---|---|---|
| OpenAI | `pip install flowmcp[openai]` | `openai/gpt-4o-mini` |
| Anthropic | `pip install flowmcp[anthropic]` | `anthropic/claude-3-5-sonnet` |
| Ollama (local) | `pip install flowmcp` | `ollama/llama3` |
| Gemini | `pip install flowmcp[gemini]` | `gemini/gemini-pro` |
| Mistral | `pip install flowmcp[mistral]` | `mistral/mistral-large` |
| Groq | `pip install flowmcp[groq]` | `groq/llama3-70b` |
| HuggingFace | `pip install flowmcp[huggingface]` | `huggingface/mistralai/Mixtral-8x7B` |

## Pipeline API

### `Pipeline(llm, docs=None, tools=None, embedding_model=None)`

| Parameter | Type | Description |
|---|---|---|
| `llm` | str or LLMBase | LLM provider string or instance |
| `docs` | str or list | Path to documents or raw text |
| `tools` | list | MCP tools (optional) |
| `embedding_model` | str | Embedding model name (optional) |

### `pipeline.ask(question)` → str

Runs the full RAG + LLM pipeline and returns an answer.

### `pipeline.aask(question)` → Coroutine[str]

Async version of `ask()`.

## RAG API

```python
from flowmcp import RAG

rag = RAG("./my-docs")
chunks = rag.retrieve("What is the refund policy?", top_k=3)
```

## CLI

```bash
flowmcp ask --llm openai/gpt-4o-mini --docs ./docs "What is this about?"
```

## Java SDK

```java
Pipeline pipeline = new Pipeline(llm, rag);
String answer = pipeline.ask("What is the refund policy?");
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute.
