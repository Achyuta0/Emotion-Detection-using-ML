import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def Predict(data):

    # Load the new data to be predicted
    new_data = pd.DataFrame({'text': [str(data)]})

    # Preprocess the new data
    X_new = vectorizer.transform(new_data['text'])

    # Predict using the model
    predictions = model.predict(X_new)

    # Display the predictions
    for text, prediction in zip(new_data['text'], predictions):
        # print(f"Text: '{text}', Predicted Emotion (label): {prediction}")
        if prediction == 0: return "Emotion is sad."
        elif prediction == 1: return "Emotion is joy"
        elif prediction == 2: return "Emotion is love"
        elif prediction == 3: return "Emotion is anger"
        elif prediction == 1: return "Emotion is fear"
        