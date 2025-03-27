import pandas as pd
import Hand_Tracking_Module as htm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("data.csv")
X = df.iloc[:,1:26]
y = df.iloc[:,-1]


model = RandomForestClassifier(100)
model.feature_names_in_ = [str(i) for i in range(1,26)]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
score = accuracy_score(y_test, y_predicted)
print(score)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)

f.close()