import numpy as np
import matplotlib.pyplot as plt



# ПУНКТ 1: Задане диференціальне рівняння та його точний розв'язок
# Рівняння: dy/dx = x - y, y(0) = 1
def f(x, y):
    return x - y


def exact_solution(x):
    return x - 1 + 2 * np.exp(-x)


# Загальні параметри задачі
x0, y0 = 0.0, 1.0
a, b = 0.0, 5.0
h_adams = 0.1  # Крок для методу Адамса
h_rk4 = 0.01  # Крок для методу РК4 (із завдання П.6: h = 10^-2)



# ПУНКТ 2: Метод прогнозу та корекції Адамса 2-го порядку
def adams_predictor_corrector(f, a, b, y0, h):
    N = int((b - a) / h) + 1
    x = np.linspace(a, b, N)
    y = np.zeros(N)
    y_pred = np.zeros(N)

    y[0] = y0
    y_pred[0] = y0

    # Для старту Адамса потрібна ще одна точка. Знайдемо її точним методом
    y[1] = exact_solution(x[1])
    y_pred[1] = y[1]

    for i in range(1, N - 1):
        f_n = f(x[i], y[i])
        f_n_minus_1 = f(x[i - 1], y[i - 1])

        # Прогноз
        y_pr = y[i] + (h / 2) * (3 * f_n - f_n_minus_1)
        y_pred[i + 1] = y_pr

        # Корекція
        f_n_plus_1_pr = f(x[i + 1], y_pr)
        y_cor = y[i] + (h / 2) * (f_n_plus_1_pr + f_n)

        y[i + 1] = y_cor

    return x, y, y_pred


# ПУНКТ 6: Метод Рунге-Кутта 4-го порядку (RK4)
def rk4_step(f, xi, yi, h):
    k1 = f(xi, yi)
    k2 = f(xi + h / 2, yi + h * k1 / 2)
    k3 = f(xi + h / 2, yi + h * k2 / 2)
    k4 = f(xi + h, yi + h * k3)
    return yi + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def runge_kutta_4(f, a, b, y0, h):
    N = int((b - a) / h) + 1
    x = np.linspace(a, b, N)
    y = np.zeros(N)
    y[0] = y0

    for i in range(N - 1):
        y[i + 1] = rk4_step(f, x[i], y[i], h)
    return x, y



# ПУНКТ 5 та 9: Автоматичний вибір кроку (реалізовано для РК4)
def adaptive_rk4(f, a, b, y0, tol=1e-5):
    x = [a]
    y = [y0]
    h_arr = [0.1]

    xi, yi, hi = a, y0, 0.1

    while xi < b:
        if xi + hi > b:
            hi = b - xi

        y_full = rk4_step(f, xi, yi, hi)
        y_half_1 = rk4_step(f, xi, yi, hi / 2)
        y_half_2 = rk4_step(f, xi + hi / 2, y_half_1, hi / 2)

        error = (16 / 15) * abs(y_full - y_half_2)

        if error <= tol:
            xi += hi
            yi = y_half_2
            x.append(xi)
            y.append(yi)
            h_arr.append(hi)

            if error < tol / 32:
                hi *= 2
        else:
            hi /= 2

    return np.array(x), np.array(y), np.array(h_arr)


# --- ВИКОНАННЯ ОБЧИСЛЕНЬ ---
# Адамс (П.2, П.3, П.4)
x_adams, y_adams, y_pr_adams = adams_predictor_corrector(f, a, b, y0, h_adams)
y_exact_adams = exact_solution(x_adams)
err_adams_exact = y_adams - y_exact_adams  # П.3
err_adams_est = y_adams - y_pr_adams  # П.4

# RK4 (П.6, П.7)
x_rk4, y_rk4 = runge_kutta_4(f, a, b, y0, h_rk4)
y_exact_rk4 = exact_solution(x_rk4)
err_rk4_exact = y_rk4 - y_exact_rk4  # П.7

# RK4 оцінка за Рунге (П.8)
y_rk4_half = np.zeros(len(x_rk4))
y_rk4_half[0] = y0
for i in range(len(x_rk4) - 1):
    temp_y = rk4_step(f, x_rk4[i], y_rk4_half[i], h_rk4 / 2)
    y_rk4_half[i + 1] = rk4_step(f, x_rk4[i] + h_rk4 / 2, temp_y, h_rk4 / 2)
err_rk4_runge = (16 / 15) * np.abs(y_rk4 - y_rk4_half)  # П.8

# Адаптивний RK4 (П.9)
x_adapt, y_adapt, h_adapt = adaptive_rk4(f, a, b, y0, tol=1e-6)


# ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
plt.figure(figsize=(16, 12))

# 1. Загальний графік розв'язків
plt.subplot(2, 2, 1)
plt.plot(x_rk4, y_exact_rk4, 'k-', label="Точний розв'язок", linewidth=2)
plt.plot(x_adams, y_adams, 'ro--', label=f"Адамс (h={h_adams})", markersize=4)
plt.plot(x_rk4, y_rk4, 'g-', label=f"RK4 (h={h_rk4})", alpha=0.7)
plt.title("Порівняння чисельних розв'язків із точним")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# 2. Похибки методу Адамса (П.3 та П.4)
plt.subplot(2, 2, 2)
plt.plot(x_adams, err_adams_exact, 'r-', label="Точна похибка (П.3)")
plt.plot(x_adams, err_adams_est, 'b--', label="Оцінка (y_cor - y_pr) (П.4)")
plt.title("Похибки методу Адамса 2-го порядку")
plt.xlabel("x")
plt.ylabel("Похибка")
plt.grid(True)
plt.legend()

# 3. Похибки методу РК4 (П.7 та П.8)
plt.subplot(2, 2, 3)
plt.plot(x_rk4, err_rk4_exact, 'g-', label="Точна похибка РК4 (П.7)")
plt.plot(x_rk4, err_rk4_runge, 'm--', label="Оцінка за Рунге (П.8)")
plt.title(f"Похибки методу Рунге-Кутта (h={h_rk4})")
plt.xlabel("x")
plt.ylabel("Похибка")
plt.grid(True)
plt.legend()

# 4. Автоматичний вибір кроку (П.9)
plt.subplot(2, 2, 4)
plt.step(x_adapt, h_adapt, 'c-', where='post', linewidth=2, label="Крок h(x)")
plt.title("Динаміка зміни кроку (Адаптивний РК4, tol=1e-6) (П.9)")
plt.xlabel("x")
plt.ylabel("Розмір кроку h")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()