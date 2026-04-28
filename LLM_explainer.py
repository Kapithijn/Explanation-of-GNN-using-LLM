from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Pick best device: MPS on Apple Silicon, else CPU
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
)
model.to(device)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]

# Qwen uses a chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([text], return_tensors="pt").to(device)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

# Decode only the new tokens
output_ids = generated[0][len(inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print("Model:", response)