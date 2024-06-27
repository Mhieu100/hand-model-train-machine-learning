import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model

# Paths to the dataset
image_folder = r"data-angle/"
excel_path = r"hands.xlsx"

# Read Excel file
df = pd.read_excel(excel_path)

# Convert 'ID' column to string and append '.png'
df['ID'] = df['ID'].astype(str) + '.png'

# Prepare features and labels
features = []
labels = []

for index, row in df.iterrows():
    filename = row['ID']
    label = 1 if row['T/P'] == 'P' else 0
    file_path = os.path.join(image_folder, filename)

    if os.path.exists(file_path):
        img = Image.open(file_path).convert('L')
        img = img.resize((128, 128))
        img_array = np.array(img).flatten()
        features.append(img_array)
        labels.append(label)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['T', 'P'], zero_division=1))

# Save the model and the scaler
joblib.dump(model, 'hand_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved to 'hand_model.pkl' and 'scaler.pkl' respectively.")
