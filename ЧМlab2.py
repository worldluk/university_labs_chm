import csv
import numpy as np
import matplotlib.pyplot as plt
import math

#БАЗОВІ ФУНКЦІЇ ДЛЯ МНОГОЧЛЕНІВ
def read_data(filename):
    """Зчитування експериментальних даних з CSV [cite: 45]"""
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)


#Метод 1: Розділені різниці (для будь-яких вузлів)
def divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (x[j:n] - x[0:n - j])
    return coef


def newton_polynomial(coef, x_data, x_val):
    n = len(coef)
    p = coef[-1]
    for k in range(2, n + 1):
        p = coef[-k] + (x_val - x_data[-k]) * p
    return p


#Метод 2: Скінченні різниці та Факторіальні многочлени (для рівновіддалених вузлів)
def forward_differences(y):
    """Обчислення скінченних різниць (дельта y) """
    m = len(y) - 1
    diffs = list(y)
    result = [diffs[0]]
    for order in range(1, m + 1):
        diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        result.append(diffs[0])
    return result


def falling_factorial(t, k):
    """Обчислення факторіального многочлена t^(k) [cite: 204]"""
    if k == 0:
        return 1
    prod = 1
    for i in range(k):
        prod *= (t - i)
    return prod


def factorial_polynomial(diffs, t_val):
    """Побудова полінома через факторіальні многочлени [cite: 205]"""
    n = len(diffs)
    P = 0
    for k in range(n):
        P += (diffs[k] / math.factorial(k)) * falling_factorial(t_val, k)
    return P


# =====================================================================
# ЧАСТИНА 2: ВИКОНАННЯ ЗАВДАННЯ ВАРІАНТУ 5 [cite: 76]
# =====================================================================

print("--- ВАРІАНТ 5: ПРОГНОЗУВАННЯ FPS ---")
x_nodes, y_nodes = read_data('data.csv')

# 1. Розділені різниці (Звичайний Ньютон)
coef_newton = divided_diff(x_nodes, y_nodes)
fps_1000_newton = newton_polynomial(coef_newton, x_nodes, 1000)

# 2. Факторіальні многочлени.
# Оскільки x_nodes не є рівновіддаленими (100, 200, 400...), ми вводимо заміну t = log2(x/100).
# Тоді t_nodes будуть рівновіддаленими: 0, 1, 2, 3, 4.
t_nodes = np.log2(x_nodes / 100)
diffs_forward = forward_differences(y_nodes)

# Прогноз для 1000 об'єктів через t
t_1000 = math.log2(1000 / 100)  # t для x=1000
fps_1000_factorial = factorial_polynomial(diffs_forward, t_1000)

print(f"Прогноз FPS для 1000 об'єктів (Метод Ньютона): {fps_1000_newton:.2f}")
print(f"Прогноз FPS для 1000 об'єктів (Факторіальні):  {fps_1000_factorial:.2f}")

# Шукаємо межу для FPS >= 60 [cite: 76]
x_dense = np.linspace(100, 1600, 1500)
y_dense = [newton_polynomial(coef_newton, x_nodes, xi) for xi in x_dense]

max_objects = 100
for i, fps in enumerate(y_dense):
    if fps < 60:
        max_objects = x_dense[i - 1]
        break

print(f"Максимальна кількість об'єктів для FPS >= 60: ~{int(max_objects)}")

# Побудова графіка для FPS
plt.figure(figsize=(10, 5))
plt.plot(x_nodes, y_nodes, 'ro', label="Експериментальні дані (Вузли)", markersize=8)
plt.plot(x_dense, y_dense, 'b-', label="Інтерполяційний многочлен")
plt.plot(1000, fps_1000_newton, 'gs', label=f"Прогноз (1000 об'єктів, {fps_1000_newton:.1f} FPS)", markersize=8)
plt.axhline(y=60, color='gray', linestyle='--', label="Межа 60 FPS")
plt.axvline(x=max_objects, color='orange', linestyle='--')
plt.title("Залежність FPS від кількості об'єктів (LOD)")
plt.xlabel("Кількість об'єктів (n)")
plt.ylabel("FPS")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================================
# ЧАСТИНА 3: ДОСЛІДНИЦЬКА ЧАСТИНА (ЕФЕКТ РУНГЕ ТА ПОХИБКИ) [cite: 78, 79]
# =====================================================================
# Для дослідження беремо класичну функцію Рунге: f(x) = 1 / (1 + x^2) на інтервалі [-5, 5]

def runge_function(x):
    return 1 / (1 + x ** 2)


x_true = np.linspace(-5, 5, 500)
y_true = runge_function(x_true)

nodes_list = [5, 10, 20]  # Досліджуємо різну кількість вузлів [cite: 78]

plt.figure(figsize=(15, 5))

for idx, n in enumerate(nodes_list):
    # Генеруємо n рівномірних вузлів
    x_n = np.linspace(-5, 5, n)
    y_n = runge_function(x_n)

    # Будуємо поліном Ньютона
    coef_n = divided_diff(x_n, y_n)
    y_approx = [newton_polynomial(coef_n, x_n, xi) for xi in x_true]

    # Обчислюємо похибку e(x) = |f(x) - N_n(x)| [cite: 41]
    error = np.abs(y_true - y_approx)

    # Графік інтерполяції
    plt.subplot(2, 3, idx + 1)
    plt.plot(x_true, y_true, 'k--', label="Справжня функція")
    plt.plot(x_true, y_approx, 'b-', label=f"Ньютон (n={n})")
    plt.plot(x_n, y_n, 'ro', label="Вузли")
    plt.title(f"Інтерполяція (n={n})")
    # Обмежуємо вісь Y, бо при n=20 коливання дуже великі
    plt.ylim(-0.5, 1.5)
    plt.grid(True)

    # Графік похибок
    plt.subplot(2, 3, idx + 4)
    plt.plot(x_true, error, 'r-')
    plt.title(f"Похибка (n={n})")
    plt.yscale('log')  # Логарифмічна шкала для кращої видимості
    plt.grid(True)

plt.tight_layout()
plt.show()