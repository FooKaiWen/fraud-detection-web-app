from flask import Flask, request, jsonify, render_template
import contractions
import inflect
import pickle
import re
import string
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

p = inflect.engine()
lemmatizer = WordNetLemmatizer()

with open('fraud_detection_model.pkl', 'rb') as file:
    word_ratios_simplified_dict, suspicious_bigrams, model, tfidf_vectorizer = pickle.load(file)

def convert_numbers_to_words_inflect(text):
    return ' '.join(p.number_to_words(word) if word.isdigit() and len(word) < 10 else word for word in text.split())

def text_transform_custom(message):
    message = message.lower()
    message = re.sub(r'[\(\[].*?[\)\]]', "", message)
    message = convert_numbers_to_words_inflect(message)
    tokens = re.findall(r'\b\w+\b', message)
    y = [token for token in tokens if token.isalnum()]
    y = [contractions.fix(word) for word in y]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation and i not in ['uhhhhrmm', 'uh', 'um', 'uh-huh']]
    y = [lemmatizer.lemmatize(token) for token in y]
    return " ".join(y)

def explain_prediction(text):
    text_tfidf = tfidf_vectorizer.transform(text)
    prediction = model.predict(text_tfidf)[0]
    
    feature_indices = text_tfidf.nonzero()[1]
    feature_names = tfidf_vectorizer.get_feature_names_out()
    relevant_feature_names = [feature_names[i] for i in feature_indices]
    relevant_features_dict = {k: v for k, v in word_ratios_simplified_dict.items() if k in relevant_feature_names}
    sorted_relevant_features = dict(sorted(relevant_features_dict.items(), key=lambda item: item[1][1], reverse=True))

    return prediction, sorted_relevant_features

@app.route('/')
def root():
    return render_template('index.html', suspicious_bigrams=suspicious_bigrams)

@app.route('/api/predict'
           , methods=['POST']
           )
def api():
    try:
        if request.json is None:
            return jsonify({"error": "The body is invalid"}), 500
        
        data = request.json

        # Validate payload
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        # Preprocess the text
        raw_text = data['text']
        processed_text = [text_transform_custom(raw_text)]
        result, features = explain_prediction(processed_text)
        response = {
            "text": raw_text,
            "processed_text": processed_text,
            "prediction": bool(result),
            "relevant_words": features
        }
        return jsonify(response)
    except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':  
   app.run()  