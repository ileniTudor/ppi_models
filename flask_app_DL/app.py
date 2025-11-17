import io
import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


onnx_path = "best_model_resnet18.onnx"
ort_session = ort.InferenceSession(onnx_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):

            image = Image.open(io.BytesIO(file.read()))

            image = image.resize((224, 224))
            image_np = np.asarray(image).astype(np.float32) / 255.
            image_np = (image_np - mean) / std
            image_np = np.transpose(image_np, (2, 0, 1))
            image_np = np.expand_dims(image_np, axis=0)

            ort_inputs = {ort_session.get_inputs()[0].name: image_np}
            ort_outs = ort_session.run(None, ort_inputs)
            predicted_class = np.argmax(ort_outs[0])
            label = 'Dog' if predicted_class.item() == 1 else 'Cat'
            return jsonify({'label': label})

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
