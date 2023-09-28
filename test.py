import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the test data from the CSV file
test_data = pd.read_csv('./dataset/test.csv')

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess the test data
X_test = vectorizer.transform(test_data['text'])

# Predict using the model
y_pred = model.predict(X_test)

# Evaluate the model
test_accuracy = accuracy_score(test_data['label'], y_pred)
print(f"Test accuracy: {test_accuracy}")
