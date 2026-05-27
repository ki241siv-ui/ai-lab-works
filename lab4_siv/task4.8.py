import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, (m, 1))

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

print("Linear Regression:")
print("Intercept:", lin_reg.intercept_)
print("Coefficients:", lin_reg.coef_)
print("R2 Score:", r2_score(y, y_pred_lin))

print("\nPolynomial Regression (degree=2):")
print("Intercept:", poly_reg.intercept_)
print("Coefficients:", poly_reg.coef_)
print("R2 Score:", r2_score(y, y_pred_poly))

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_pred_lin, color='red', label='Лінійна регресія')
plt.plot(X, y_pred_poly, color='green', linewidth=2, label='Поліноміальна регресія (ступінь 2)')
plt.title('Варіант 7: sin(X) + шум')
plt.xlabel('x1')
plt.ylabel('y')
plt.legend()
plt.show()