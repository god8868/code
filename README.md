# AI Model Adapter

This repository contains a lightweight **AI model adapter** framework implemented in `model_adapter.py`. It abstracts different Large-Language-Model (LLM) providers behind the same Python interface so your application code remains unchanged when you swap providers.

## Supported back-ends

1. **OpenAI** ChatCompletion API – `openai` adapter (requires an API key)
2. **Hugging Face** local / hosted models – `huggingface` adapter (uses `transformers`)

You can extend the framework with your own providers by subclassing `BaseAdapter` and decorating with `@register_adapter("your_name")`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```python
from model_adapter import get_adapter

# Example 1 – OpenAI
openai_adapter = get_adapter(
    "openai",
    api_key="sk-...",   # or set OPENAI_API_KEY env var
    model="gpt-3.5-turbo",
)
print(openai_adapter.generate("Say hello in French."))

# Example 2 – Hugging Face (local gpt2)
hf_adapter = get_adapter("huggingface", model="gpt2", max_length=50)
print(hf_adapter.generate("Once upon a time"))
```

---

Feel free to open issues or contribute new adapters! :rocket:
