from flask import Flask, jsonify, request, render_template, make_response
from flask_cors import CORS  # This is the magic
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import time

start_time = time.time()
model_name = "./lan-t5-base-SQuAD"  # Replace with your model's path or name
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to('cuda')
tokenizer = T5Tokenizer.from_pretrained(model_name)
load_time = time.time() - start_time
print(f"Load time: {load_time:.4f} seconds")

os.environ["FLASK_DEBUG"] = "1"

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route('/')
def home():
    return render_template('home.html')  # Render home.html

@app.route("/predict", methods=["GET", "POST"])
def index():

    data = request.json
    context = data["a" + str(0)]
    input_text = data["a" + str(1)]

    print("context", context)
    print("input_text", input_text)


    start_time = time.time()
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to('cuda')
    tokenizer_time = time.time() - start_time

    start_time = time.time() 
    output_ids = model.generate(inputs["input_ids"], max_length=100, num_beams=4, early_stopping=True)
    generate_time = time.time() - start_time

    start_time = time.time()
    predicted_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    decode_time = time.time() - start_time

    print(f"Tokenizer time: {tokenizer_time:.4f} seconds")
    print(f"Generate time: {generate_time:.4f} seconds") 
    print(f"Decode time: {decode_time:.4f} seconds")
    print(f"Total time: {tokenizer_time + generate_time + decode_time:.4f} seconds")

    print("predicted_answer",predicted_answer)
    response = jsonify({"predicted_answer": predicted_answer})
    response.status_code = 200
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
