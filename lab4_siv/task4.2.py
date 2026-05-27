import numpy as np
import matplotlib.pyplot as plt

X = np.array([-12, 29, 0, 4, 6, 8])
Y = np.array([-3, 0, 1, 2, 9, 5])

beta_1, beta_0 = np.polyfit(X, Y, 1)

print(f"y = {beta_1:.4f}x + {beta_0:.4f}")

x_line = np.linspace(min(X), max(X), 100)
y_line = beta_1 * x_line + beta_0

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', s=100, label='Експериментальні точки')
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'y = {beta_1:.2f}x + {beta_0:.2f}')
plt.title('МНК: Апроксимація даних')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()