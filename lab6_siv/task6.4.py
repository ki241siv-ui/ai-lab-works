import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
df = pd.read_csv(url)

df = df.drop(columns=['insert_date'])
df = df.dropna()

le = LabelEncoder()
categorical_cols = ['origin', 'destination', 'train_type', 'train_class', 'fare']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df[['origin', 'destination', 'train_type', 'train_class', 'fare']]
y = le.fit_transform(df['fare'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))