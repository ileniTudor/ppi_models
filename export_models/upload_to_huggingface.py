from huggingface_hub import HfApi

from token_file import token
from transformers import T5Tokenizer, T5ForConditionalGeneration

api = HfApi(token=token)
model_id = "tudorileni/lanT5squad2"
model_name = "../flask_app_llm/lan-t5-base-SQuAD"

model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

model.push_to_hub(model_id, token=token)
tokenizer.push_to_hub(model_id, token=token)