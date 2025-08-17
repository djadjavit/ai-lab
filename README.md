# AI Lab

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-Library-yellow.svg?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A small lab for experimenting with **local LLMs** using Hugging Face [Transformers](https://huggingface.co/docs/transformers).
> A set of simple examples for running local LLMs on your computer.

---

## Quick Start

```bash
# 1) Clone and enter the repo
git clone https://github.com/yourname/ai-lab.git
cd ai-lab

# 2) Create & activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -U pip
pip install -r requirements.txt  # or see "Dependencies" below
```

---

## Dependencies

If you don't use `requirements.txt`, install the essentials:
```bash
pip install transformers torch accelerate
```

**Model used:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

> Tip: CPU-only is fine for these tiny models, but enabling a GPU will speed things up.

---

## Scripts

### 1) `hello_llm.py`

First experiment with running a local LLM (Qwen2.5-0.5B-Instruct).

**Usage:**
```bash
# activate your venv first
python hello_llm.py

# CLI: run with a custom prompt
python hello_llm.py "Come up with 3 mini-project ideas with LLM"
```

---

### 2) `chat_llm.py`

Interactive CLI chat assistant powered by Qwen2.5-0.5B-Instruct.  
The script remembers the conversation history (`chat_history.json`) and streams responses token by token.

**Usage:**
```bash
python chat_llm.py
```

**Example session:**
```text
🤖 Local LLM chat. Type 'exit' to quit.

You: Come up with 2 mini-project ideas with an LLM
Assistant:
1. Build a simple chat assistant that can maintain a conversation and remember context.
2. Create an idea generator for learning projects in Python using a local model.

You: exit
👋 Bye!
```

---

## Troubleshooting

- **Slow / high RAM on CPU**: Use smaller context lengths or shorter prompts.  
- **CUDA not available**: Install a CUDA-enabled PyTorch build, or run on CPU; you can set `device_map="cpu"` in the script.  
- **Tokenizer/model download errors**: Check your internet connection and retry; Transformers will cache models under `~/.cache/huggingface/` by default.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.
