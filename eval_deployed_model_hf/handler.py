import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


# "question":"Which country contains 60% of the rainforest?",
# "context":"-"
# What time period did the Industrial Revolution span?

class EndpointHandler:
    def __init__(self, path=""):
        # Determine device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on device: {self.device}")

        # Path to the model is provided by the Inference Endpoint container
        # The model files (config.json, pytorch_model.bin, tokenizer_config.json, etc.)
        # will be available in the 'path' directory.
        model_path = path if path else "tudorileni/lanT5squad"  # Fallback if path is empty (unlikely in IE)

        # Load the tokenizer and model
        # For a private model, ensure the HF_TOKEN environment variable is set
        # in the Inference Endpoint's configuration.
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, use_auth_token=os.environ.get("HF_TOKEN"))
        self.model = T5ForConditionalGeneration.from_pretrained(model_path,
                                                                use_auth_token=os.environ.get("HF_TOKEN")).to(
            self.device)
        self.model.eval()  # Set model to evaluation mode

        print("Model and tokenizer loaded successfully.")

    def __call__(self, inputs: dict) -> dict:
        """
        Process the incoming request.
        'inputs' will be the JSON payload sent to the API.
        For a Q&A model, we expect {"question": "...", "context": "..."}.
        """
        question = inputs.get("question")
        context = inputs.get("context")

        if not question or not context:
            raise ValueError("Both 'question' and 'context' must be provided in the input.")

        # Prepare input in T5's Q&A format
        input_text = f"question: {question} context: {context}"

        # Tokenize input
        # Note: If inputs have different lengths, you might need padding=True, truncation=True
        # For simple Q&A, this should be fine.
        encoded_input = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,  # Ensure truncation if too long
            truncation=True
        ).to(self.device)

        # Generate answer with your desired parameters
        output_ids = self.model.generate(
            encoded_input["input_ids"],
            max_length=50,  # Use your desired max_length
            num_beams=4,  # Use your desired num_beams
            early_stopping=True  # Use your desired early_stopping
        )

        predicted_answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {"answer": predicted_answer}