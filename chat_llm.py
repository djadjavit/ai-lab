# chat_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch, sys, threading, json, os

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def main():
    print("🤖 Local LLM chat. Type 'exit' to quit, 'reset' to clear history.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # set safe padding/end tokens
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    history = load_history()  # [{role: "user"/"assistant", content: "..."}]

    while True:
        try:
            user = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break

        if user.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break
        if user.lower() == "reset":
            history = []
            save_history(history)
            print("🔄 History cleared.\n")
            continue
        if not user:
            continue

        # 1) update history
        history.append({"role": "user", "content": user})

        # 2) prepare messages for the chat template
        # you can add a system message if desired
        messages = [{"role": "system", "content": "You are a friendly assistant. Answer briefly and to the point."}]
        messages.extend(history)

        # 3) apply Qwen's chat template
        # this is an important step: it ensures correct special tokens and no echo
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # adds assistant prompt
            return_tensors="pt"
        )

        # 4) configure response streaming (without printing the prompt)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            streamer=streamer,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("Assistant: ", end="", flush=True)
        response = ""
        for piece in streamer:
            sys.stdout.write(piece)
            sys.stdout.flush()
            response += piece
        print("\n")

        history.append({"role": "assistant", "content": response})
        save_history(history)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
