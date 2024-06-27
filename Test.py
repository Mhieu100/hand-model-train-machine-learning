import numpy as np
from PIL import Image
import joblib

# Load the trained model and the scaler
model = joblib.load('hand_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict the hand type from an image
def predict_hand(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img).flatten()
    img_array = scaler.transform([img_array])  # Standardize features
    prediction = model.predict(img_array)
    return 'Right Hand' if prediction == 1 else 'Left Hand'

# Example predictions
new_image_paths = [
    r"data-angle/14595.png",
    r"data-angle/14409.png",
    r"data-angle/14369.png"
]

for image_path in new_image_paths:
    print(f"Prediction for {image_path}: {predict_hand(image_path)}")
