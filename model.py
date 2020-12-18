import pandas as pd
import pickle
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


df_clean = pd.read_csv('data_clean.csv')


def tf_idf(data):
  vectorizer = TfidfVectorizer()
  return vectorizer, vectorizer.fit_transform(data)

df_clean['tweets'] = df_clean['tweets'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)


vec,x = tf_idf(df_clean['tweets'])
label = np.array(df_clean['label'].values)

X_train, X_test, y_train, y_test = train_test_split(x, label, test_size=0.25, random_state=1)

lb_make = LabelEncoder()
y_train = lb_make.fit_transform(y_train)
y_test = lb_make.fit_transform(y_test)

# k = StratifiedKFold(n_splits=10,shuffle=False)
# i=1
svm = SVC(probability=True)
svm.fit(X_train, y_train)

pickle.dump(svm, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

print("Training done")