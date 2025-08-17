from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch, sys

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # или "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,   # CPU безопасно
    device_map="auto"            # CPU/GPU выберется автоматически
)

prompt = "Ты — дружелюбный ассистент. Коротко представься и скажи одну идею для мини-проекта по LLM."
inputs = tokenizer(prompt, return_tensors="pt")
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

gen_kwargs = dict(**inputs, max_new_tokens=128, temperature=0.8, top_p=0.95, streamer=streamer)
thread = torch.cuda._lazy_init if False else None  # просто чтоб не ругался mypy :)
import threading
t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
t.start()
for piece in streamer:
    sys.stdout.write(piece); sys.stdout.flush()
t.join()
