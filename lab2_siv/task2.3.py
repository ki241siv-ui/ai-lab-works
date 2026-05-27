import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

print("Форма датасету:", dataset.shape)
print("\nПерші 5 рядків:\n", dataset.head(5))
print("\nСтатистичне резюме:\n", dataset.describe())
print("\nРозподіл за класами:\n", dataset.groupby('class').size())

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.suptitle('Діаграми розмаху ознак')
plt.show()

dataset.hist()
plt.suptitle('Гістограми розподілу')
plt.show()

scatter_matrix(dataset)
plt.suptitle('Матриця діаграм розсіювання')
plt.show()

array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

model = SVC(gamma='auto')
model.fit(X_train, Y_train)

predictions = model.predict(X_validation)
print("\nТочність (Accuracy Score):", accuracy_score(Y_validation, predictions))
print("\nМатриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію:\n", classification_report(Y_validation, predictions))

X_new = [[5.0, 2.9, 1.0, 0.2]]
prediction = model.predict(X_new)
print(f"\nПрогноз для {X_new}: {prediction[0]}")
