import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('https://raw.githubusercontent.com/asthala/racism-detection/master/datasetfix.csv')
df.head()

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

df_clean = df.copy()
df_clean['tweets']= clean(df['tweets'])

def case_fold(data):
    return data.str.lower()

df_clean['tweets'] = case_fold(df_clean['tweets'])

def token(data):
  return data.apply(nltk.word_tokenize)

df_clean['tweets'] = token(df_clean['tweets'])

def stop_words(data) :
  stop_words = set(stopwords.words('indonesian'))
  return data.apply(lambda x: [item for item in x if item not in stop_words])

df_clean['tweets'] = stop_words(df_clean['tweets'])

def stem(data):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  return data.apply(lambda x: [ stemmer.stem(item) for item in x])

df_clean['tweets'] = stem(df_clean['tweets'])

print(df_clean['tweets'][:5])

df_clean.to_csv('data_clean.csv',index = False)


