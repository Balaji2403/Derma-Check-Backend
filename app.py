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



# import os
# import requests
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "https://derma-check.netlify.app"}})

# # Google Drive file ID of the model
# MODEL_FILE_ID = "1KrAzcqKnZPBBfLkHEGoL2eJRCFh6vwYL"
# MODEL_PATH = "best_model.keras"

# # Function to download model from Google Drive
# def download_model():
#     if not os.path.exists(MODEL_PATH):
#         print("Downloading model from Google Drive...")
#         url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
#         response = requests.get(url, stream=True)

#         with open(MODEL_PATH, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)

#         print("Model download complete!")

# # Ensure model is downloaded before loading
# download_model()
# model = load_model(MODEL_PATH)

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
#     app.run(host="0.0.0.0", port=5000)

# from flask import Flask, request, jsonify
# import gdown
# import os
# import zipfile
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# from flask_cors import CORS

# app = Flask(__name__)

# # Allow only your Netlify frontend to access the API
# CORS(app, resources={r"/*": {"origins": "https://derma-check.netlify.app"}})

# # Google Drive file ID of your model
# GOOGLE_DRIVE_FILE_ID = "1SvznIpebERCB2LjAFep7AzxhNlz98hof"
# MODEL_ZIP_PATH = "best_model.zip"
# MODEL_DIR = "best_model"

# # Function to download and extract model
# def download_model():
#     if not os.path.exists(MODEL_ZIP_PATH):
#         print("Downloading model from Google Drive...")
#         gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_ZIP_PATH, quiet=False)
#         print("Model download complete!")

#     if not os.path.exists(MODEL_DIR):
#         print("Extracting model files...")
#         with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
#             zip_ref.extractall(MODEL_DIR)
#         print("Extraction complete!")

# # Load the model
# download_model()
# with open(os.path.join(MODEL_DIR, "config.json"), "r") as json_file:
#     model_json = json_file.read()

# model = model_from_json(model_json)
# model.load_weights(os.path.join(MODEL_DIR, "model.weights.h5"))

# # Class names
# class_names = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma', 'Actinic Keratosis',
#                'Dermatofibroma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Vascular lesion']

# malignant_classes = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma']
# benign_classes = ['Actinic Keratosis', 'Dermatofibroma', 'Pigmented Benign Keratosis',
#                   'Seborrheic Keratosis', 'Vascular lesion']

# # Prediction function
# def predict_image(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

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
#     os.remove(file_path)

#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)  # Production-ready


from flask import Flask, request, jsonify
import gdown
import os
import zipfile
import gc
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

# Allow only your Netlify frontend to access the API
CORS(app, resources={r"/*": {"origins": "https://derma-check.netlify.app"}})

# Google Drive file ID of your model
GOOGLE_DRIVE_FILE_ID = "1SvznIpebERCB2LjAFep7AzxhNlz98hof"
MODEL_ZIP_PATH = "best_model.zip"
MODEL_DIR = "best_model"
MODEL_CONFIG = os.path.join(MODEL_DIR, "config.json")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "model.weights.h5")

# Function to download and extract model if not present
def download_model():
    if not os.path.exists(MODEL_DIR):
        print("🚀 Downloading model from Google Drive...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_ZIP_PATH, quiet=False)
            print("✅ Model download complete!")

            print("📂 Extracting model files...")
            with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DIR)
            print("✅ Extraction complete!")

        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            exit(1)

# Load the model efficiently
def load_keras_model():
    global model
    if "model" not in globals():
        gc.collect()  # Free memory
        print("🔄 Loading model into memory...")
        with open(MODEL_CONFIG, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(MODEL_WEIGHTS)
        print("✅ Model successfully loaded!")
    return model

# Ensure model is downloaded and loaded at startup
download_model()
load_keras_model()

# Class names
class_names = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma', 'Actinic Keratosis',
               'Dermatofibroma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Vascular lesion']

malignant_classes = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma']
benign_classes = ['Actinic Keratosis', 'Dermatofibroma', 'Pigmented Benign Keratosis',
                  'Seborrheic Keratosis', 'Vascular lesion']

# Prediction function
def predict_image(image_path):
    model = load_keras_model()
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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

    try:
        result = predict_image(file_path)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500
    finally:
        os.remove(file_path)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
