import requests
import os

from token_file import token

# --- Configuration ---
MODEL_ID = "tudorileni/lanT5squad"
API_URL = f"https://vs7pz0y264201c15.us-east-1.aws.endpoints.huggingface.cloud"
# IMPORTANT: Replace "hf_YOUR_API_TOKEN" with your actual Hugging Face API token.
# You can generate one from your Hugging Face settings (Settings -> Access Tokens).
# Ensure it has at least 'read' access.
# It's best practice to load this from an environment variable or a secure config,
# but for a quick test, you can paste it directly.
HF_API_TOKEN = os.environ.get("HF_TOKEN", token) # Replace this or set env variable

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json" # Specify content type for JSON payload
}

def query_model_api(question: str, context: str):
    payload = {
        "inputs":"",
            "question": question,
            "context": context
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
    return response.json()

# --- Your example usage ---
context = "The United States Constitution was signed on September 17, 1787, by delegates to the Constitutional Convention in Philadelphia."
question = "When was the US constitution signed?"

try:
    # Make the API call
    output = query_model_api(question, context)

    # The structure of the output might vary slightly based on the exact pipeline HF uses
    # for T5-based Q&A, but usually it returns a dictionary with 'answer' or 'generated_text'.
    # For a T5 Q&A model, it typically returns a dictionary like {'answer': '...', 'score': ..., 'start': ..., 'end': ...}
    # Let's inspect the output:
    print("Full API Response:", output)

    # Accessing the answer:
    if isinstance(output, dict) and "answer" in output:
        predicted_answer = output["answer"]
    elif isinstance(output, list) and output and isinstance(output[0], dict) and "answer" in output[0]:
         predicted_answer = output[0]["answer"]
    else:
        predicted_answer = "Could not parse answer from response."

    print(f"\nQuestion: {question}")
    print(f"Context: {context}")
    print(f"Predicted Answer: {predicted_answer}")

except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
except Exception as e:
    print(f"An error occurred: {e}")