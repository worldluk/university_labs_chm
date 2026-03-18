import csv
import numpy as np
import matplotlib.pyplot as plt
import math


# =====================================================================
# БАЗОВІ ФУНКЦІЇ ДЛЯ ІНТЕРПОЛЯЦІЇ
# =====================================================================

def read_data(filename):
    """Зчитування експериментальних даних з CSV"""
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)


def divided_diff(x, y):
    """Обчислення розділених різниць (для многочлена Ньютона)"""
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (x[j:n] - x[0:n - j])
    return coef


def newton_polynomial(coef, x_data, x_val):
    """Обчислення значення полінома Ньютона в точці (Схема Горнера)"""
    n = len(coef)
    p = coef[-1]
    for k in range(2, n + 1):
        p = coef[-k] + (x_val - x_data[-k]) * p
    return p


def forward_differences(y):
    """Обчислення скінченних різниць (для факторіальних многочленів)"""
    m = len(y) - 1
    diffs = list(y)
    result = [diffs[0]]
    for order in range(1, m + 1):
        diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        result.append(diffs[0])
    return result


def falling_factorial(t, k):
    """Обчислення факторіального многочлена t^(k)"""
    if k == 0: return 1
    prod = 1
    for i in range(k):
        prod *= (t - i)
    return prod


def factorial_polynomial(diffs, t_val):
    """Побудова полінома через факторіальні многочлени"""
    n = len(diffs)
    P = 0
    for k in range(n):
        P += (diffs[k] / math.factorial(k)) * falling_factorial(t_val, k)
    return P


# =====================================================================
# ХІД РОБОТИ (ОСНОВНЕ ЗАВДАННЯ - ВАРІАНТ 5)
# =====================================================================

print("--- ВАРІАНТ 5: ПРОГНОЗУВАННЯ FPS ---")

# 1. Зчитати дані з CSV-файлу
x_nodes, y_nodes = read_data('data.csv')

# 2. Побудувати таблицю розділених різниць
coef_newton = divided_diff(x_nodes, y_nodes)

# 3. Обчислити прогноз (методами Ньютона і факторіальними многочленами)
# Прогноз методом Ньютона для 1000 об'єктів
fps_1000_newton = newton_polynomial(coef_newton, x_nodes, 1000)

# Прогноз факторіальними многочленами
# Оскільки x_nodes зростають експоненційно, робимо заміну t = log2(x/100) для рівновіддалених вузлів
t_nodes = np.log2(x_nodes / 100)
diffs_forward = forward_differences(y_nodes)
t_1000 = math.log2(1000 / 100)
fps_1000_factorial = factorial_polynomial(diffs_forward, t_1000)

print(f"Прогноз FPS для 1000 об'єктів (Метод Ньютона): {fps_1000_newton:.2f}")
print(f"Прогноз FPS для 1000 об'єктів (Факторіальні):  {fps_1000_factorial:.2f}")

# 6. Зробити висновки щодо оптимізації (Пошук межі для FPS >= 60)
x_dense = np.linspace(100, 1600, 1500)
y_dense = [newton_polynomial(coef_newton, x_nodes, xi) for xi in x_dense]

max_objects = 100
for i, fps in enumerate(y_dense):
    if fps < 60:
        max_objects = x_dense[i - 1]
        break
print(f"Максимальна кількість об'єктів для FPS >= 60: ~{int(max_objects)}")

# 4. Побудувати графік FPS(n)
plt.figure(figsize=(10, 5))
plt.plot(x_nodes, y_nodes, 'ro', label="Експериментальні дані (Вузли)", markersize=8)
plt.plot(x_dense, y_dense, 'b-', label="Інтерполяційний многочлен Ньютона")
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
# ДОСЛІДНИЦЬКА ЧАСТИНА
# =====================================================================
print("\n--- ДОСЛІДНИЦЬКА ЧАСТИНА ---")


# 1 та 3. Вплив кроку (фіксований інтервал) і Аналіз ефекту Рунге
def runge_function(x):
    return 1 / (1 + x ** 2)


x_true = np.linspace(-5, 5, 500)
y_true = runge_function(x_true)
nodes_list = [5, 10, 20]

plt.figure(figsize=(15, 5))
for idx, n in enumerate(nodes_list):
    x_n = np.linspace(-5, 5, n)
    y_n = runge_function(x_n)

    coef_n = divided_diff(x_n, y_n)
    y_approx = [newton_polynomial(coef_n, x_n, xi) for xi in x_true]
    error = np.abs(y_true - y_approx)

    plt.subplot(2, 3, idx + 1)
    plt.plot(x_true, y_true, 'k--', label="Справжня функція")
    plt.plot(x_true, y_approx, 'b-', label=f"Ньютон (n={n})")
    plt.plot(x_n, y_n, 'ro', label="Вузли")
    plt.title(f"Інтерполяція (n={n}) - Ефект Рунге")
    plt.ylim(-0.5, 1.5)
    plt.grid(True)

    # 5 (з основного ходу) та 3 (з дослідницької): Побудова графіків похибок
    plt.subplot(2, 3, idx + 4)
    plt.plot(x_true, error, 'r-')
    plt.title(f"Похибка (n={n})")
    plt.yscale('log')
    plt.grid(True)

plt.tight_layout()
plt.show()

# 2. Дослідження впливу кількості вузлів, похибок (Фіксований крок, змінний інтервал)
print("\nДослідження: Фіксований крок, змінний інтервал (Функція sin(x))")


def test_func(x):
    return np.sin(x)


h = 0.5  # Фіксований крок
a = 0  # Початок інтервалу

for n in nodes_list:
    b = a + h * n  # Змінний кінець інтервалу залежно від кількості вузлів

    x_nodes_var = np.arange(a, b, h)
    y_nodes_var = test_func(x_nodes_var)

    x_test = np.linspace(a, b - h, 200)
    y_true_var = test_func(x_test)

    coef_var = divided_diff(x_nodes_var, y_nodes_var)
    y_newton_var = [newton_polynomial(coef_var, x_nodes_var, xi) for xi in x_test]

    max_error = np.max(np.abs(y_true_var - y_newton_var))
    print(f"Вузлів (n): {n:2d} | Інтервал: [{a:4.1f}, {b:4.1f}] | Максимальна похибка: {max_error:.2e}")