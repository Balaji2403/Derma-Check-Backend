    # from flask import Flask, request, jsonify
    # from tensorflow.keras.models import load_model
    # from tensorflow.keras.preprocessing.image import img_to_array, load_img
    # import numpy as np
    # import os
    # from flask_cors import CORS


    # app = Flask(__name__)
    # CORS(app)  # Allow requests from React frontend

    # # Load the trained model
    # model = load_model('c:/Users/Balaji/Documents/Project/best_model.keras')

    # # Class names
    # class_names = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma', 'Actinic Keratosis', 
    #             'Dermatofibroma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Vascular lesion']

    # # Classification categories
    # malignant_classes = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma']
    # benign_classes = ['Actinic Keratosis', 'Dermatofibroma', 'Pigmented Benign Keratosis', 
    #                 'Seborrheic Keratosis', 'Vascular lesion']

    # # Image preprocessing and prediction function
    # def predict_image(image_path):
    #     img = load_img(image_path, target_size=(224, 224))
    #     img_array = img_to_array(img) / 255.0  # Normalize
    #     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    #     predictions = model.predict(img_array)
    #     predicted_class = class_names[np.argmax(predictions)]
    #     confidence = float(np.max(predictions))

    #     classification = "Malignant" if predicted_class in malignant_classes else "Benign"

    #     return {"class": predicted_class, "type": classification, "confidence": confidence}

    # @app.route('/predict', methods=['POST'])
    # def upload_image():
    #     if 'file' not in request.files:
    #         return jsonify({"error": "No file provided"}), 400

    #     file = request.files['file']
    #     file_path = "temp.jpg"
    #     file.save(file_path)

    #     result = predict_image(file_path)
    #     os.remove(file_path)  # Clean up temp file

    #     return jsonify(result)

    # if __name__ == '__main__':
    #     app.run(debug=True)

# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# import os
# from flask_cors import CORS

# app = Flask(__name__)

# # Allow only your Netlify frontend to access the API
# CORS(app, resources={r"/*": {"origins": "https://derma-check.netlify.app"}})

# # Set model path
# MODEL_PATH = os.path.join(os.getcwd(), "best_model.keras")

# # Load the trained model
# if os.path.exists(MODEL_PATH):
#     model = load_model(MODEL_PATH)
# else:
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# # Class names
# class_names = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma', 'Actinic Keratosis', 
#                'Dermatofibroma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Vascular lesion']

# # Classification categories
# malignant_classes = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma']
# benign_classes = ['Actinic Keratosis', 'Dermatofibroma', 'Pigmented Benign Keratosis', 
#                   'Seborrheic Keratosis', 'Vascular lesion']

# # Image preprocessing and prediction function
# def predict_image(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     predictions = model.predict(img_array)
#     predicted_class = class_names[np.argmax(predictions)]
#     confidence = float(np.max(predictions))

#     classification = "Malignant" if predicted_class in malignant_classes else "Benign"

#     return {"class": predicted_class, "type": classification, "confidence": confidence}

# @app.route('/predict', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files['file']
#     file_path = "temp.jpg"
#     file.save(file_path)

#     result = predict_image(file_path)
#     os.remove(file_path)  # Clean up temp file

#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)  # Remove debug=True for production



import os
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://derma-check.netlify.app"}})

# Google Drive file ID of the model
MODEL_FILE_ID = "1KrAzcqKnZPBBfLkHEGoL2eJRCFh6vwYL"
MODEL_PATH = "best_model.keras"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
        response = requests.get(url, stream=True)

        with open(MODEL_PATH, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Model download complete!")

# Ensure model is downloaded before loading
download_model()
model = load_model(MODEL_PATH)

# Class names
class_names = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma', 'Actinic Keratosis', 
               'Dermatofibroma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Vascular lesion']

# Classification categories
malignant_classes = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma']
benign_classes = ['Actinic Keratosis', 'Dermatofibroma', 'Pigmented Benign Keratosis', 
                  'Seborrheic Keratosis', 'Vascular lesion']

# Image preprocessing and prediction function
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    classification = "Malignant" if predicted_class in malignant_classes else "Benign"

    return {"class": predicted_class, "type": classification, "confidence": confidence}

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_path = "temp.jpg"
    file.save(file_path)

    result = predict_image(file_path)
    os.remove(file_path)  # Clean up temp file

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

