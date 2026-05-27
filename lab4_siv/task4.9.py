import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")

m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, (m, 1))

lin_reg = LinearRegression()
plt.figure(figsize=(10, 5))
plot_learning_curves(lin_reg, X, y)
plt.title("Learning Curves (Linear Regression)")
plt.show()

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

plt.figure(figsize=(10, 5))
plot_learning_curves(polynomial_regression, X, y)
plt.title("Learning Curves (Polynomial degree=10)")
plt.axis([0, 80, 0, 3])
plt.show()