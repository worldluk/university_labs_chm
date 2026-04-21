import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from typing import Callable, Tuple


# ФУНКЦІЯ 1: Підінтегральна функція f(x)
def f(x: float | np.ndarray) -> float | np.ndarray:
    """Аналітично задана функція навантаження на сервер."""
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)



# ФУНКЦІЯ 2: Складова формула Сімпсона
def simpson(func: Callable, a: float, b: float, N: int) -> float:
    """Обчислення означеного інтегралу за складовою формулою Сімпсона."""
    if N % 2 != 0:
        raise ValueError("Кількість розбиттів N має бути парним числом.")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)  # ТУТ ВИКЛИКАЄТЬСЯ ФУНКЦІЯ 1 ( f(x) )

    sum_odd = np.sum(y[1:-1:2])
    sum_even = np.sum(y[2:-1:2])

    return (h / 3) * (y[0] + 4 * sum_odd + 2 * sum_even + y[-1])



# ФУНКЦІЯ 3: Адаптивний алгоритм Сімпсона
def adaptive_simpson(func: Callable, a: float, b: float, delta: float, calls: int = 0) -> Tuple[float, int]:
    """Рекурсивний адаптивний алгоритм чисельного інтегрування [cite: 124-138]."""
    h = (b - a) / 2
    c = (a + b) / 2

    # Інтеграл на всьому поточному відрізку
    I1 = (h / 3) * (func(a) + 4 * func(c) + func(b))  # ТУТ ВИКЛИКАЄТЬСЯ ФУНКЦІЯ 1
    calls += 3

    # Інтеграли на половинках відрізка
    h2 = h / 2
    c1 = (a + c) / 2
    c2 = (c + b) / 2

    I_left = (h2 / 3) * (func(a) + 4 * func(c1) + func(c))
    I_right = (h2 / 3) * (func(c) + 4 * func(c2) + func(b))
    I2 = I_left + I_right
    calls += 6

    if abs(I1 - I2) <= delta:
        return I2, calls
    else:
        # ТУТ ФУНКЦІЯ 3 ВИКЛИКАЄ САМУ СЕБЕ (Рекурсія)
        I_L, calls_L = adaptive_simpson(func, a, c, delta / 2, 0)
        I_R, calls_R = adaptive_simpson(func, c, b, delta / 2, 0)
        return I_L + I_R, calls + calls_L + calls_R



# ЧАСТИНА 2: ОСНОВНА ЛОГІКА ТА ОБЧИСЛЕННЯ (Використання функцій)
# --- Еталонне значення ---
I_exact, _ = integrate.quad(f, 0, 24)

# --- Дані для графіка Сімпсона (Пункт 4) ---
N_values = np.arange(10, 1002, 2)
errors_simpson = []
for N in N_values:
    # ВИКЛИК ФУНКЦІЇ 2 (simpson) у циклі
    I_approx = simpson(f, 0, 24, N)
    errors_simpson.append(abs(I_approx - I_exact))

target_eps = 1e-12
idx_opt = np.where(np.array(errors_simpson) <= target_eps)[0][0]
N_opt = N_values[idx_opt]
eps_opt = errors_simpson[idx_opt]

# --- Дані для екстраполяції (Пункти 5-8) ---
N0 = 8
# ВИКЛИК ФУНКЦІЇ 2 (simpson) для трьох різних сіток
I_N0 = simpson(f, 0, 24, N0)
I_N0_2 = simpson(f, 0, 24, N0 // 2)
I_N0_4 = simpson(f, 0, 24, N0 // 4)

I_R = I_N0 + (I_N0 - I_N0_2) / 15  # Формула Рунге-Ромберга
I_E = (I_N0_2 ** 2 - I_N0 * I_N0_4) / (2 * I_N0_2 - (I_N0 + I_N0_4))  # Формула Ейткена

# --- Дані для адаптивного методу (Пункт 9) ---
deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
actual_errors = []
calls_list = []
for delta in deltas:
    # ВИКЛИК ФУНКЦІЇ 3 (adaptive_simpson)
    res_adapt, num_calls = adaptive_simpson(f, 0, 24, delta)
    err_adapt = abs(res_adapt - I_exact)

    actual_errors.append(err_adapt if err_adapt > 0 else np.nan)
    calls_list.append(num_calls)
    

# --- Вивід результатів у консоль ---
print(f"Еталонний інтеграл: {I_exact:.8f}")
print(f"N_opt для точності 10^-12: {N_opt}")
print(f"Похибка Рунге-Ромберга (N0=8): {abs(I_R - I_exact):.2e}")
print(f"Похибка Ейткена (N0=8): {abs(I_E - I_exact):.2e}")


# ЧАСТИНА 3: ГРАФІКИ (Всі візуалізації зібрані тут)
# --- ГРАФІК 1: Візуалізація підінтегральної функції ---
plt.figure(figsize=(10, 5))
x_plot = np.linspace(0, 24, 1000)
plt.plot(x_plot, f(x_plot), color='steelblue', label='f(x)')  # Знову використовуємо Функцію 1
plt.title('Графік функції навантаження на сервер (Пункт 1)')
plt.xlabel('Час, х (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# --- ГРАФІК 2: Залежність похибки Сімпсона від N ---
plt.figure(figsize=(10, 5))
plt.plot(N_values, errors_simpson, color='crimson', label='ε(N)')
plt.axhline(y=target_eps, color='black', linestyle=':', label='Задана точність 10⁻¹²')
plt.yscale('log')
plt.title('Залежність похибки складової формули Сімпсона від N (Пункт 4)')
plt.xlabel('Кількість розбиттів, N')
plt.ylabel('Абсолютна похибка (Лог. масштаб)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- ГРАФІК 3: Фактична похибка адаптивного методу ---
plt.figure(figsize=(10, 5))
plt.plot(deltas, actual_errors, marker='o', color='darkorange', label='Фактична похибка')
plt.plot(deltas, deltas, linestyle='--', color='gray', label='Лінія y = x')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.title('Адаптивний метод: Фактична похибка від Заданої δ (Пункт 9)')
plt.xlabel('Задана точність, δ')
plt.ylabel('Фактична похибка')
plt.grid(True, alpha=0.3, which="both", ls="--")
plt.legend()
plt.show()

# --- ГРАФІК 4: Трудомісткість адаптивного методу ---
plt.figure(figsize=(10, 5))
plt.plot(deltas, calls_list, marker='s', color='seagreen', label='Кількість викликів f(x)')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.title('Адаптивний метод: Трудомісткість від Заданої δ (Пункт 9)')
plt.xlabel('Задана точність, δ')
plt.ylabel('Кількість обчислень функції, шт.')
plt.grid(True, alpha=0.3, which="both", ls="--")
plt.legend()
plt.show()