# hello_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import argparse
import sys, threading

def build_parser():
    p = argparse.ArgumentParser(description="Run a small local LLM (Qwen2.5-0.5B-Instruct).")
    p.add_argument("prompt", nargs="*", help="Your prompt. If empty, will ask interactively.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF model id (default: Qwen/Qwen2.5-0.5B-Instruct)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max generated tokens (default: 512)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (default: 0.8)")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="Top-p nucleus sampling (default: 0.95)")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable streaming; print the full answer at once.")
    return p

def main():
    args = build_parser().parse_args()
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        try:
            prompt = input("Enter your prompt: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    # Load model & tokenizer (CPU-safe defaults)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto"  # will use CPU if no GPU
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    # === Option A: full output at once (no streaming) ===
    if args.no_stream:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # If the model echoes the prompt, keep only the completion part:
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        print(text)
        return

    # === Option B: streaming output (default) ===
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        streamer=streamer,
    )

    th = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    th.start()
    try:
        for piece in streamer:
            sys.stdout.write(piece)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        th.join()

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
