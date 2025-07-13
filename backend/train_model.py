import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load the dataset
df = pd.read_csv(r"C:\Users\Acer\Desktop\kavya A\Road Accident\backend\data\updated_data.csv")

# Drop missing values
df.dropna(inplace=True)

# Map categorical variables
weather_map = {'clear': 0, 'rainy': 1, 'foggy': 2, 'snowy': 3}
road_map = {'highway': 0, 'city': 1, 'rural': 2}

df['weather'] = df['weather'].map(weather_map)
df['road_type'] = df['road_type'].map(road_map)

# Verify required columns
required_cols = ['weather', 'road_type', 'vehicle_speed', 'traffic_density', 'accident']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"Dataset must include columns: {required_cols}")

# Define features and target
X = df[['weather', 'road_type', 'vehicle_speed', 'traffic_density']]
y = df['accident']

# Show class distribution before training
print("⚠️ Label Distribution:")
print(y.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with class_weight='balanced'
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
print("\n✅ Classification Report on Test Set:")
print(classification_report(y_test, model.predict(X_test)))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model trained and saved as 'model.pkl'")
