#  pip install transformers accelerate bitsandbytes einops --upgrade
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from token_file import token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
# ---

# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# model_id = "microsoft/Phi-3-mini-4k-instruct"
model_id = "google/gemma-2-2b-it"
print(f"Loading model: {model_id}...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    token=token
    # trust_remote_code=True # for microsoft model
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Set pad token

text_to_analyze = "Aorta la inel, 10 milimetri, aorta la sinusurile varțalva, 5 milimetri, aorta așcendentă, păiți 11 milimetri. Ventrii colul drept, 10 milimetri, atriu stâng, 20 milimetri. Poți să lădicte și prescurtat, adică mă rogă orăta la inel și asta cu orăta, nu da după aia, a se. Paiși pe milimetri, vede, 10 milimetri. Noi, ca masa. Ne-au ziem, mulțumesc."

form_schema = """
{
  "subiect": "subiectul ingistrarii",
  "aorta ascendenta": "care este dimensiunea in mm a aortei ascendente",
  "aorta la sinusurile vartarda": "care este dimensiunea in mm a la sinusurile vartarda",
  "ventriculul drept ": "care este dimensiunea vertriculului drept",
  "atriu stâng ": "care este dimensiunea atriului stang?",
}
"""
if "google/gemma" in model_id:
    messages = [
        {
            "role": "user",
            "content": (
                "Sunteți un asistent expert în extragerea informațiilor. "
                "Analizați cu atenție textul și extrageți informațiile solicitate de utilizator. "
                "Trebuie să afișați informațiile * doar * într-un format JSON valid, bazat pe schema furnizată. "
                "Dacă nu se găsesc informații pentru o cheie, utilizați 'null'.\n\n"

                "Vă rugăm să extrageți informațiile din textul următor pe baza schemei JSON furnizate.\n\n"
                "TEXT:\n"
                f'"{text_to_analyze}"\n\n'
                "SCHEMA:\n"
                f"{form_schema}\n\n"
                "JSON OUTPUT:"
            )
        }
    ]
else:
    messages = [
        {
            "role": "system",
            "content": (
                "Sunteți un asistent expert în extragerea informațiilor."
                "Analizați cu atenție textul și extrageți informațiile solicitate de utilizator."
                "Trebuie să afișați informațiile * doar * într-un format JSON valid, bazat pe schema furnizată.Dacă nu se găsesc informații pentru o cheie, utilizați 'nul'"""
            )
        },
        {
            "role": "user",
            "content": f"""
    Vă rugăm să extrageți informațiile din textul următor pe baza schemei JSON furnizate.
    
    TEXT:
    "{text_to_analyze}"
    
    SCHEMA:
    {form_schema}
    
    JSON OUTPUT:
    """
        }
    ]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("--- PROMPT ---")
print(prompt)

print("\n--- Running Extraction ---")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
# print("input#s.input_ids.shape[1]",inputs.input_ids.shape[1],"outputs",outputs)
response_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

print("\n--- MODEL OUTPUT (Raw) ---")
print(response_text)