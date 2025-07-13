import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load sample data (replace with your own data)
data = load_iris()
X = data.data
y = data.target

# Train a model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model as model.pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("model.pkl file created!")
