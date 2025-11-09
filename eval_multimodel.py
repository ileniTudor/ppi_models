import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, \
    AutoModelForImageTextToText
from PIL import Image
import os

from token_file import token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
print(f"Loading model: {model_id}...")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    token=token
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("Model and processor loaded successfully!")

photo_1 = "street.jpg"
photo_2 = "red_car.jpg"


print("Se încarcă imaginile de pe discul local...")

if not os.path.exists(photo_1):
    raise FileNotFoundError(f"Fișierul nu a fost găsit: {photo_1}")
if not os.path.exists(photo_2):
    raise FileNotFoundError(f"Fișierul nu a fost găsit: {photo_2}")
image_without = Image.open(photo_1)
image_with = Image.open(photo_2)
print("Imaginile au fost încărcate cu succes!")


def check_frame_for_object(image, object_description):
    """
    Verifică un singur cadru de imagine pentru un obiect descris.
    Folosește noul standard 'multimodal chat templating'.
    """

    # 1. Crearea promptului VQA
    prompt_text = f"Is there a {object_description} in the image? Answer with 'Yes' or 'No'."

    # 2. Formatarea promptului pentru model (stilul NOU)
    # Imaginea (obiectul PIL) și textul sunt acum într-o listă sub 'content'
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Trecem direct obiectul PIL
                {"type": "text", "text": prompt_text}
            ]
        },
    ]

    # 3. Procesarea folosind apply_chat_template
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception as e:
        print(f"Eroare la procesarea imaginii cu apply_chat_template: {e}")
        return "Eroare la procesare"

    # 4. Generarea răspunsului
    outputs = model.generate(**inputs, max_new_tokens=10)

    # 5. Decodificarea răspunsului
    # Găsim lungimea token-urilor de intrare pentru a le tăia
    start_index = inputs['input_ids'].shape[1]
    response_text = processor.decode(outputs[0][start_index:], skip_special_tokens=True).strip().lower()

    print(f"  > VLM Check: Prompt: '{prompt_text}' -> VLM Answer: '{response_text}'")

    # 6. Logica Aplicației
    if response_text.startswith("yes"):
        return "Am găsit obiectul: " + object_description
    else:
        return "Încă caut..."

search_query = "red car"

print("--- Testarea Cadrului 1 (FĂRĂ mașina roșie) ---")
result_1 = check_frame_for_object(image_without, search_query)
print(f"Rezultat final pentru utilizator: {result_1}\n")

print("--- Testarea Cadrului 2 (CU mașina roșie) ---")
result_2 = check_frame_for_object(image_with, search_query)
print(f"Rezultat final pentru utilizator: {result_2}\n")