import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

input_file = 'income_data.txt'
X, y = [], []
count_class1, count_class2 = 0, 0
max_datapoints = 500 

try:
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break
            if '?' in line:
                continue
            data = line[:-1].split(', ')
            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
                count_class1 += 1
            elif data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
                count_class2 += 1
except FileNotFoundError:
    print(f"Error: File {input_file} not found!")
    exit()

X = np.array(X)
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

kernels = [
    ('Polynomial', SVC(kernel='poly', degree=8)),
    ('Gaussian (RBF)', SVC(kernel='rbf')),
    ('Sigmoid', SVC(kernel='sigmoid'))
]

print(f"{'Kernel Type':<20} | {'F1 Score (%)':<15}")
print("-" * 40)

for name, clf in kernels:
    f1 = cross_val_score(clf, X, y, scoring='f1_weighted', cv=3)
    print(f"{name:<20} | {round(100 * f1.mean(), 2)}%")
