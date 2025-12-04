import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# === 1. Example questionnaire dataset ===
# You can replace this with your own CSV if available
data = {
    'hydration': [1, 2, 3, 2, 1, 3, 2, 4],      # 1=Low, 2=Medium, 3=High, etc.
    'diet': [2, 3, 1, 2, 4, 1, 3, 2],          # 1=Healthy, 4=Unhealthy
    'sleep': [3, 2, 4, 3, 2, 1, 3, 4],         # Hours category
    'stress': [5, 7, 2, 4, 8, 6, 3, 2],        # 1–10 scale
    'pollution': [1, 2, 1, 2, 1, 2, 1, 2],     # 1=Low, 2=High
    'climate': [1, 3, 2, 1, 3, 2, 1, 3],       # 1=Humid, 2=Dry, 3=Moderate
    'skincare': [1, 2, 1, 2, 1, 2, 1, 2],      # 1=No, 2=Yes
    'skin_type': ['oily', 'dry', 'normal', 'oily', 'dry', 'oily', 'normal', 'dry']
}

df = pd.DataFrame(data)

# === 2. Separate features & labels ===
X = df.drop(columns=['skin_type'])
y = df['skin_type']

# === 3. Encode labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 4. Train model ===
model = LogisticRegression(max_iter=1000)
model.fit(X, y_encoded)

# === 5. Save model & encoder ===
with open('models/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Model and label encoder saved in 'models/' folder")
