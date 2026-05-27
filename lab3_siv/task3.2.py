import numpy as np
import matplotlib.pyplot as plt

def trimf(x, abc):
    a, b, c = abc
    res = np.zeros_like(x)
    if a != b:
        idx1 = np.logical_and(a < x, x <= b)
        res[idx1] = (x[idx1] - a) / (b - a)
    if b != c:
        idx2 = np.logical_and(b < x, x < c)
        res[idx2] = (c - x[idx2]) / (c - b)
    res[x == b] = 1.0
    return np.maximum(res, 0)
x_reg = np.linspace(-90, 90, 400)

reg_terms = {
    'large_left':  [-90, -90, -45],
    'small_left':  [-50, -25, 0],
    'off':         [-10, 0, 10],
    'small_right': [0, 25, 50],
    'large_right': [45, 90, 90]
}

in_temp_status = 'very_warm'
in_speed = 'positive'

mu_rule = 1.0 
activation = np.fmin(mu_rule, trimf(x_reg, reg_terms['large_left']))

if np.sum(activation) == 0:
    result = 0
else:
    result = np.sum(x_reg * activation) / np.sum(activation)

plt.figure(figsize=(10, 6))
colors = ['blue', 'cyan', 'green', 'orange', 'red']

for (name, params), color in zip(reg_terms.items(), colors):
    plt.plot(x_reg, trimf(x_reg, params), label=name, color=color, lw=2)

plt.fill_between(x_reg, 0, activation, color='gray', alpha=0.3)
plt.axvline(x=result, color='black', lw=3, label=f'Результат: {result:.2f}°')

plt.title('Керування кондиціонером: Регулятор', fontsize=14)
plt.xlabel('Кут повороту регулятора (вліво - холод, вправо - тепло)', fontsize=12)
plt.ylabel('Ступінь належності', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()
