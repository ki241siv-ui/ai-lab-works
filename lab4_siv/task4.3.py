import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3, 1, 1.8, 1.9])

f_linear = interp1d(x, y, kind='linear')
f_cubic = interp1d(x, y, kind='cubic')

x_new = np.linspace(min(x), max(x), 100)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', s=100, label='Точки')
plt.plot(x_new, f_linear(x_new), '--', color='blue', label='Linear')
plt.plot(x_new, f_cubic(x_new), '-', color='green', linewidth=2, label='Cubic Spline')
plt.title('Інтерполяція')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()