from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model_path = "C:/Users/ma578/OneDrive/Documents/Motive/DataSets/IPHONE/project 4/MNIST/project/models/mnist_model.h5"
model = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255
    
    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_digit])
    
    return jsonify({
        'digit': int(predicted_digit),
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
