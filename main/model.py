"""verify_backbone.py — confirm DeepSeek-R1-Distill-Qwen-1.5B loads on M3."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[device] {device}")

print(f"[load] tokenizer ...")
tok = AutoTokenizer.from_pretrained(MODEL)

print(f"[load] model (this downloads ~3GB on first run) ...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
print(f"[loaded] params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B  "
      f"hidden_size: {model.config.hidden_size}")

# One GSM8K-style sanity prompt
prompt = (
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes "
    "muffins with 4. She sells the rest at $2 each. How much does she make per day?"
)
messages = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(device)

print("[generate] ...")
out = model.generate(**ids, max_new_tokens=400, do_sample=False, pad_token_id=tok.eos_token_id)
print("=" * 70)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
print("=" * 70)
print("[done]")