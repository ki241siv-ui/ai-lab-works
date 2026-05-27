import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

outlook = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
wind = ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong']
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

le = LabelEncoder()
outlook_encoded = le.fit_transform(outlook)
humidity_encoded = le.fit_transform(humidity)
wind_encoded = le.fit_transform(wind)
label = le.fit_transform(play)

features = list(zip(outlook_encoded, humidity_encoded, wind_encoded))

model = GaussianNB()
model.fit(features, label)

predicted = model.predict([[0, 0, 0]]) 

print("Result:", "Yes" if predicted == 1 else "No")

probabilities = model.predict_proba([[0, 0, 0]])
print(f"Probabilities [No, Yes]: {probabilities[0]}")