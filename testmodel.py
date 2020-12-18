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

model_mlp_so = pickle.load(open('model_mlp_so.pkl', 'rb'))

def predict(sentence, model):

    # sentence = "Kamu jahat kaya cina komunis" #@param
    # sentence = str(request.form.values())
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

    return 'The tweet is detected :  {}'.format(output)

sent = input("input text : ")
pred = predict(sent, model_mlp_so)
print(pred)