from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set the NLTK data directory inside the virtual environment
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

nltk.download('punkt', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')

app = Flask(__name__)

# Load the Word2Vec model
model_filepath = "word2vec_model.bin"
model = Word2Vec.load(model_filepath)

# Preprocess the text data
stop_words = set(stopwords.words('english'))


def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens


# Calculate similarity score between two texts
def similarity_score(text1, text2):
    preprocessed_text1 = preprocess(text1)
    preprocessed_text2 = preprocess(text2)

    embedding1 = [model.wv[word] for word in preprocessed_text1 if word in model.wv]
    embedding2 = [model.wv[word] for word in preprocessed_text2 if word in model.wv]

    if embedding1 and embedding2:
        return cosine_similarity(embedding1, embedding2).mean()
    else:
        return 0.0


@app.route('/', methods=['POST'])
def calculate_similarity():
    data = request.get_json()
    text1 = data['text1']
    text2 = data['text2']
    similarity = similarity_score(text1, text2)
    similarity = round(float(similarity), 2)  # Convert to float
    return jsonify({"similarity score": similarity})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
