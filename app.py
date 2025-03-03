from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)  

# Load the TrashNet model
MODEL_PATH = "model/Trash_Detection.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocess the image for the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    return image

# API endpoint for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get the image file from the request
        file = request.files['image']
        image = Image.open(file.stream)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Perform inference
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Return the result as JSON
        return jsonify({
            "predicted_class": int(predicted_class),
            "predicted_label": predicted_label,
            "confidence": float(predictions[0][predicted_class]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
