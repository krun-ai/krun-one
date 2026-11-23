---
language:
- en
license: apache-2.0
tags:
- function-calling
- tool-use
- synthetic
- agents
- training-data
size_categories:
- 10K<n<100K
task_categories:
- text-generation
---

# Function-Calling Dataset

High-quality synthetic dataset for training function-calling / tool-use capabilities in LLMs.

## Dataset Details
- **9723 examples** across 6 domains
- **30 unique function definitions** with JSON Schema parameters
- **5 complexity levels**: simple, moderate, complex, multi-tool, conversational
- Generated using Gemini 2.0 Flash with careful prompt engineering and validation

## Domains
| Domain | Description | Functions |
|--------|-------------|-----------|
| E-commerce | Shopping, cart, checkout | 5 |
| Calendar | Events, scheduling, free slots | 5 |
| DevOps | Deploy, logs, scaling, metrics | 5 |
| Finance | Accounts, transfers, invoices | 5 |
| CRM | Contacts, deals, pipeline | 5 |
| Data Analysis | SQL, charts, exports | 5 |

## Format
Each example is a JSON object with:
- `messages`: Array of conversation messages (system, user, assistant with function_call, function results)
- `metadata`: Domain, complexity level, available functions

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("Johin/function-calling-dataset")
```

## License
Apache 2.0 — use for any purpose including commercial training.
