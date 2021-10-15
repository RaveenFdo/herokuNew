import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('D:/Individual Research Project/Models/BattaramullaLiteNew PreprocessedNew.csv')

X = dataset[['PM10', 'NO2', 'SO2', 'CO']]
print(X.head())
from sklearn.preprocessing import LabelEncoder

y = dataset['PPM2.5']
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
y

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred)*100)

pickle.dump(clf, open("model.pkl", "wb"))

def predict_mpg(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    y_pred = model.predict(df)
    return y_pred


