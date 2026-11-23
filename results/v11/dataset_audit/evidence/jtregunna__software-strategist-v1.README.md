---
language: en
license: mit
tags:
  - code
  - software-engineering
  - reasoning
  - synthetic
  - instruction-tuning
  - language-agnostic
size_categories:
  - 10K<n<100K
task_categories:
  - text-generation
  - other
pretty_name: "Software Strategist v1 — Instruction-Tuning Dataset"
configs:
  - config_name: software-strategist-v1
    data_files:
      - split: train
        path: train.jsonl
---

# Software Fundamentals — Strategy Knowledge Base

A language-agnostic knowledge base of software engineering fundamentals, paired with a synthetic instruction-tuning dataset (~13,500 examples) for training small language models (SLMs) as **software engineering strategists**.

The trained model takes a description of a coding situation and routes it to relevant concepts, outputting synthesized strategic guidance as structured JSON.

## Dataset Summary

This dataset provides **~13,500 instruction-tuning examples** for training small language models (SLMs) as software engineering strategists. The training data is synthesized from a curated knowledge base of 518 language-agnostic software engineering concepts, each encoding the strategic reasoning behind when to apply a concept, when to avoid it, and what anti-patterns to watch for.

The knowledge base (`entries/`) serves as the source material — each of the 518 concept entries captures a single principle, not a snippet or tutorial, but the decision framework behind it. The training dataset (`software-strategist-v1.jsonl`) is generated from these entries, producing conversation-format examples that teach a model to route coding situations to relevant concepts.

**What makes this unique:**

- **Concept-level reasoning** — Each entry encodes a decision framework: when to use a concept, when not to, and the tradeoffs involved. No existing dataset teaches this kind of strategic reasoning at scale.
- **Anti-pattern recognition** — Every concept includes named anti-patterns with explanations of *why* they are bad, not just *that* they are bad.
- **"When not to use" guidance** — Explicitly teaching the boundaries of applicability, which is as important as teaching the concept itself.
- **Language-agnostic** — The same concept (e.g., "Circuit Breaker") applies whether the user is working in Python, Rust, JavaScript, Go, or any other language. The model learns patterns, not syntax.
- **Curated quality** — All 518 entries scored 14/14 on a 7-dimension quality rubric (see [Quality Assurance](#quality-assurance)).

## Quick Start

```python
from datasets import load_dataset

# Load the full training dataset
dataset = load_dataset("jtregunna/software-strategist-v1", split="train")

# Inspect a single example
example = dataset[0]
print(example["messages"][-1]["content"])
# {"concepts_applied": [...], "core_idea": "...", "key_principles": [...], "avoid": [...]}
```

A single training example is a conversation with three messages (system, user, assistant):

```json
{
  "messages": [
    {"role": "system", "content": "You are a software engineering strategist..."},
    {"role": "user", "content": "My API keeps crashing when the database is slow..."},
    {"role": "assistant", "content": "{\"concepts_applied\": [{\"id\": \"sf-error-classification-001\", \"name\": \"Error Classification\", \"weight\": \"primary\"}], \"core_idea\": \"Classify errors as transient (retryable) vs permanent (fail-fast)...\", \"key_principles\": [\"...\"], \"avoid\": [\"...\"]}"}
  ]
}
```

## Dataset Structure

The repository contains three types of files:

### Concept Entries (`entries/`)

518 self-contained JSON files (`sf-*.json`), each representing a single software engineering concept. Every entry follows a consistent schema (see [Schema](#schema)) and covers:

- `core_idea` — One-sentence concept definition
- `when_to_use` / `when_not_to_use` — Applicability conditions
- `anti_patterns` — Named anti-patterns with explanations
- `reasoning_chain` — Step-by-step decision logic
- `examples` — Concrete illustrative scenarios
- `related_concepts` — Cross-references to other entries

### Training Dataset (`software-strategist-v1.jsonl`)

~13,500 synthetic training examples in standard instruction-tuning format, generated from the 518 concept entries. The dataset is composed of approximately **60% single-concept**, **25% multi-concept**, and **15% null** examples. Each entry is a conversation with `messages` arrays.

The full dataset is available on [Hugging Face](https://huggingface.co/datasets/jtregunna/software-strategist-v1).

## Categories

The 518 concepts are organized across 19 categories:

### Core (7)

| Category | Scope |
|----------|-------|
| `design-patterns` | GoF patterns, SOLID, composition, structural/creational/behavioral patterns |
| `data-management` | Modeling, indexing, transactions, consistency, caching, state machines |
| `error-handling` | Classification, recovery, timeouts, retries, propagation, validation |
| `architecture` | Layered, microservices, DDD, 12-factor, API design, service mesh |
| `quality` | Code smells, testing strategies, mutation testing, observability, refactoring |
| `security` | AuthN/AuthZ, injection prevention, least privilege, secrets management |
| `algorithms` | Complexity analysis, search, sorting, graphs, dynamic programming |

### Craft (12)

| Category | Scope |
|----------|-------|
| `code-reading` | Understanding unfamiliar codebases, navigation patterns, grep-driven development |
| `legacy-code` | Tactic-based legacy modification, seam injection, characterization testing |
| `version-control` | Git workflows, branching strategies, merge vs. rebase, commit hygiene |
| `data-modeling` | ER diagrams, normalization, schema evolution, pagination, temporal modeling |
| `networking` | HTTP, REST, RPC, CORS, DNS, content negotiation, websockets |
| `build-dependencies` | Dependency management, version pinning, reproducible builds, monorepo |
| `debugging-methodology` | Systematic debugging, bisect, tracing, assumption checking, instrumenting |
| `code-review` | Review scope, feedback quality, checklist-driven review, cognitive load |
| `performance` | Profiling, caching, load testing, scaling, memory management |
| `scripting-cli` | Shell scripting, piping, environment management, process management |
| `documentation` | READMEs, inline comments, ADRs, changelog discipline, API docs |
| `concurrency` | Locks, producers/consumers, race conditions, event loops, actor model |

## Schema

Each concept entry follows a structured schema covering:

```
id, concept, category, subcategory, difficulty
core_idea, when_to_use, when_not_to_use
anti_patterns, reasoning_chain, examples
related_concepts, tags, task_types
```

See [SCHEMA.md](SCHEMA.md) for the complete field specification with examples and constraints.

Each training example (JSONL) uses the standard conversation format:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"concepts_applied\":[...],\"core_idea\":\"...\",\"key_principles\":[...],\"avoid\":[...] }"}
  ]
}
```

## Quality Assurance

All 518 concept entries are validated against a 7-dimension rubric (see [RUBRIC.md](RUBRIC.md)). Each dimension scores 0–2, with a passing threshold of ≥1 per dimension and ≥7/14 total.

| Dimension | What it measures |
|-----------|-----------------|
| Concept Clarity | Single, well-defined, language-agnostic concept |
| When-To-Use Accuracy | Precise, actionable applicability conditions |
| Anti-Pattern Quality | Named anti-patterns with clear explanations |
| Reasoning Chain Depth | Step-by-step decision logic with tradeoff analysis |
| Example Quality | Concrete, illustrative scenarios |
| Cross-Reference Integrity | Accurate links to related concepts |
| Task Relevance | Direct applicability to real coding situations |

All 518 entries score a perfect **14/14** on this rubric.

## Data Generation

The `tools/` directory contains scripts for validation and synthetic data generation:

- **`validate_entries.py`** — Validates all concept entries against the schema and rubric
- **`generate_synthetic.py`** — Generates synthetic training examples from the knowledge base, supporting multiple backends and fallback chains
- **`randomize_system_prompts.py`** — Variabilizes system prompts across generated examples to improve robustness

These tools are provided so users can regenerate or expand the dataset with their own LLM backends.

## Citation

```bibtex
@misc{tregunna2025softwarestrategist,
  title = {Software Fundamentals — Strategy Knowledge Base},
  author = {Tregunna, Jeremy},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/jtregunna/software-strategist-v1}},
  note = {MIT License}
}
```

## License

This dataset is licensed under the [MIT License](LICENSE).
