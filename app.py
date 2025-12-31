
from flask import Flask, request, jsonify
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data (only needs to be run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
try:
    model = joblib.load('random_forest_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    ps = PorterStemmer()
    print("Model and TF-IDF vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or TF-IDF vectorizer: {e}")
    # Exit or handle error appropriately if model/vectorizer cannot be loaded
    exit()

# Define the clean_text function (as used during training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http.?://[^\s]+[\s]?', '', text)
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Create a POST endpoint for sentiment prediction
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Invalid request: "text" key is missing or request is not JSON.'}), 400

    input_text = request.json['text']

    # Preprocess the input text
    cleaned_text = clean_text(input_text)

    # Transform the cleaned text using the loaded TF-IDF vectorizer
    # tfidf.transform expects an iterable, so pass cleaned_text in a list
    transformed_text = tfidf.transform([cleaned_text])

    # Predict the sentiment using the loaded model
    predicted_sentiment = model.predict(transformed_text)

    # Return the predicted sentiment as a JSON response
    return jsonify({'original_text': input_text, 'predicted_sentiment': predicted_sentiment[0]})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5000)
