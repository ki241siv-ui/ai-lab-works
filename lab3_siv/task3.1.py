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

x_temp = np.linspace(0, 100, 101)
x_press = np.linspace(0, 10, 101)
x_valve = np.linspace(-90, 90, 200)

temp_hot = trimf(x_temp, [85, 100, 100])
temp_not_hot = trimf(x_temp, [70, 80, 90])
temp_warm = trimf(x_temp, [45, 60, 75])
temp_cool = trimf(x_temp, [20, 35, 50])
temp_cold = trimf(x_temp, [0, 0, 25])

press_strong = trimf(x_press, [6, 10, 10])
press_not_strong = trimf(x_press, [3, 5, 7])
press_weak = trimf(x_press, [0, 0, 4])

v_terms = {
    'large_left': [-90, -90, -60],
    'medium_left': [-75, -45, -15],
    'small_left': [-30, -15, 0],
    'zero': [-10, 0, 10],
    'small_right': [0, 15, 30],
    'medium_right': [15, 45, 75],
    'large_right': [60, 90, 90]
}

in_temp = 10
in_press = 2

mu_t_cold = np.interp(in_temp, x_temp, temp_cold)
mu_p_weak = np.interp(in_press, x_press, press_weak)

active_rule8 = np.fmin(mu_t_cold, mu_p_weak)
out_large_right = np.fmin(active_rule8, trimf(x_valve, v_terms['large_right']))

if np.sum(out_large_right) == 0:
    result = 0
else:
    result = np.sum(x_valve * out_large_right) / np.sum(out_large_right)

plt.figure(figsize=(10, 6))
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

for (name, params), color in zip(v_terms.items(), colors):
    m_func = trimf(x_valve, params)
    plt.plot(x_valve, m_func, label=name, color=color, alpha=0.7)

plt.fill_between(x_valve, 0, out_large_right, color='pink', alpha=0.3)
plt.axvline(x=result, color='black', linewidth=3, label=f'Результат: {result:.2f}°')

plt.title('Кран гарячої води', fontsize=14)
plt.xlabel('Кут повороту', fontsize=12)
plt.ylabel('Ступінь належності', fontsize=12)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.show()
