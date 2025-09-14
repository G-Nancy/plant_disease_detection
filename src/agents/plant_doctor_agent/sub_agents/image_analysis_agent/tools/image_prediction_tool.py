# author: Nancy Goyal


"""Tool for image-based plant disease prediction."""

import os
import json
import numpy as np
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Setup (Load model and labels once) ---
# This part runs when the module is first imported.
model_dir = '/Users/gnancy/work/5h1va/plant-doctor/src/agents/plant_doctor_agent/sub_agents/image_analysis_agent/tools/'
model_path = os.path.join(model_dir, 'mobilenetv2_model.keras')
labels_path = os.path.join(model_dir, 'class_labels.json')

# Check if files exist
if not os.path.exists(model_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("Required model or labels file is missing. Please ensure 'save_model_and_labels.py' was run.")

# Load the model and labels
best_model = load_model(model_path)
with open(labels_path, 'r') as f:
    class_indices = json.load(f)
class_labels = sorted(class_indices, key=class_indices.get)
image_size = (224, 224)

# --- Agent Tool Function ---

def predict_disease_from_image(image_path: str) -> str:
    """
    Predicts the type of disease on a plant leaf from an image.

    Args:
        image_path: The full path to the image file to be analyzed.
                    The path should be accessible by the local filesystem.

    Returns:
        A string containing the predicted plant disease name.
    """
    if not os.path.exists(image_path):
        return f"Error: The image file was not found at {image_path}"

    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make the prediction
        y_pred = best_model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(y_pred, axis=1)[0]
        predicted_label = class_labels[predicted_class_index]
        
        return predicted_label

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"