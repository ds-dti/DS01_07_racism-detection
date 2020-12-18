import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)


def clean(data):
  tweets = []
  for tweet in data:
      tweet = re.sub(r"'(?:\@|https?\://)\S+", "", tweet)
      tweet = re.sub('\n', '', tweet)
      tweet = re.sub('rt', '', tweet)
      tweet = re.sub("[^a-zA-Z^']", " ", tweet)
      tweet = re.sub(" {2,}", " ", tweet)
      tweet = tweet.strip()
      tweets.append(tweet)
  return tweets

def case_fold(data):
    return data.str.lower()

def token(data):
  return data.apply(nltk.word_tokenize)

def stop_words(data) :
  stop_words = set(stopwords.words('indonesian'))
  return data.apply(lambda x: [item for item in x if item not in stop_words])

def stem(data):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  return data.apply(lambda x: [ stemmer.stem(item) for item in x])


df_clean = pd.read_csv('data_clean.csv')
def tf_idf(data):
  vectorizer = TfidfVectorizer()
  return vectorizer, vectorizer.fit_transform(data)
df_clean['tweets'] = df_clean['tweets'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
vec,x = tf_idf(df_clean['tweets'])
label = np.array(df_clean['label'].values)

model = pickle.load(open('model_mlp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    model = pickle.load(open('model_mlp.pkl', 'rb'))
    sentence = request.form.values()
    sentence = [sentence]
    sentence = pd.DataFrame(data=sentence,columns=['text'], index=[0])
    sentence['text'] = clean(sentence['text'])
    sentence['text'] = case_fold(sentence['text'])
    sentence['text'] = token(sentence['text'])
    sentence['text'] = stop_words(sentence['text'])
    sentence['text'] = stem(sentence['text'])
    sentence['text'] = sentence['text'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    clean_sentence = vec.transform(sentence['text'])
    y_pred = model.predict(clean_sentence.toarray())
    label_pred = 'Racism' if np.round(y_pred[0]) else 'Non  Racism'

    output = label_pred

    return render_template('index.html', prediction_text='The tweet is detected :  {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict(data.toarray())

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)