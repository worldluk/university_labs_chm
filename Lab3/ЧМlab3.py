import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# 1. Завантаження даних
def load_data(filename="temperature.csv"):
    x, y = [], []
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Пропускаємо заголовок (Month, Temp) [cite: 100]
            for row in reader:
                x.append(float(row[0]))
                y.append(float(row[1]))
        return np.array(x), np.array(y)

# 2. Математичні функції МНК [cite: 81, 132-162]
def form_matrix(x, m):
    """Формування матриці системи (m+1 x m+1) [cite: 134-139]"""
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
    return A


def form_vector(x, y, m):
    """Формування вектора вільних членів [cite: 140-144]"""
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))
    return b


def gauss_solve(A, b):
    """Метод Гауса з вибором головного елемента по стовпцях [cite: 81, 145-154]"""
    n = len(b)
    # Прямий хід
    for k in range(n - 1):
        max_row = np.argmax(np.abs(A[k:n, k])) + k
        # Перестановка рядків
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        # Виключення
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]  # [cite: 148]
            A[i, k:] = A[i, k:] - factor * A[k, k:]  # [cite: 148]
            b[i] = b[i] - factor * b[k]  # [cite: 149]

    # Зворотній хід [cite: 150-154]
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]

    return x_sol


def polynomial(x, coef):
    """Обчислення значення полінома [cite: 155-159]"""
    y_poly = np.zeros_like(x, dtype=float)
    for i in range(len(coef)):
        y_poly += coef[i] * (x ** i)
    return y_poly


def variance(y_true, y_approx):
    """Обчислення дисперсії [cite: 160-162]"""
    return np.mean((y_true - y_approx) ** 2)


def calculate_error(f_x, phi_x):
    """Обчислення похибки ε(x) = |f(x) - φ(x)| [cite: 81]"""
    return np.abs(f_x - phi_x)

# 3. Основна логіка програми [cite: 77-98]
def main():
    x, y = load_data()
    n_nodes = len(x)

    max_degree = 10  # Досліджуємо від 1 до 10 [cite: 82]
    variances = []
    coefficients_list = []

    # Сітка для похибки з кроком h1 = (xn - x0) / (20 * n) [cite: 87]
    h1 = (x[-1] - x[0]) / (20 * n_nodes)
    x_fine = np.arange(x[0], x[-1] + h1, h1)

    # Інтерполюємо фактичні дані на дрібну сітку для порівняння похибки
    y_fine_actual = np.interp(x_fine, x, y)
    errors_fine_dict = {}

    # Обчислюємо МНК для кожного степеня m [cite: 82, 166-169]
    for m in range(1, max_degree + 1):
        A_mat = form_matrix(x, m)
        b_vec = form_vector(x, y, m)

        coef = gauss_solve(A_mat.copy(), b_vec.copy())
        coefficients_list.append(coef)

        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        variances.append(var)

        # Обчислюємо похибку на дрібній сітці [cite: 86]
        y_fine_approx = polynomial(x_fine, coef)
        errors_fine_dict[m] = calculate_error(y_fine_actual, y_fine_approx)

    # Вибір оптимального m (Уникнення ефекту Рунге / перенавчання) [cite: 31, 1048]
    # Замість абсолютного мінімуму шукаємо момент, де дисперсія перестає суттєво падати
    optimal_m = 1
    for i in range(1, len(variances)):
        # Якщо дисперсія впала менш ніж на 5% порівняно з попереднім кроком
        if (variances[i - 1] - variances[i]) / variances[i - 1] < 0.05:
            optimal_m = i  # Беремо попередній крок (індекс i відповідає степеню m)
            break

    # Запобіжник: згідно з псевдокодом, зазвичай m обмежують невеликим числом [cite: 165]
    if optimal_m > 5:
        optimal_m = 5

    optimal_index = optimal_m - 1
    best_coef = coefficients_list[optimal_index]

    print(f"--- РЕЗУЛЬТАТИ ---")
    print(f"Оптимальний степінь полінома: m = {optimal_m}")
    print(f"Дисперсія для m={optimal_m}: {variances[optimal_index]:.4f}")

    # Прогноз на наступні 3 місяці (екстраполяція)
    x_future = np.array([25, 26, 27])
    y_future = polynomial(x_future, best_coef)
    print(f"Прогноз температур на 25, 26, 27 місяці: {np.round(y_future, 2)}")

    # ==========================================
    # 4. Візуалізація [cite: 82, 88, 96, 97]
    # ==========================================
    plt.figure(figsize=(18, 5))

    # Графік 1: Дисперсія від степені m [cite: 82, 84]
    plt.subplot(1, 3, 1)
    plt.plot(range(1, max_degree + 1), variances, marker='o', color='b', linestyle='-')
    plt.axvline(x=optimal_m, color='r', linestyle='--', label=f'Оптимальне m={optimal_m}')
    plt.title("Залежність дисперсії від степені m")
    plt.xlabel("Степінь апроксимуючого многочлена (m)")
    plt.ylabel("Дисперсія (δ²)")
    plt.xticks(range(1, max_degree + 1))
    plt.legend()
    plt.grid(True)

    # Графік 2: Апроксимація та фактичні дані [cite: 96, 172-177]
    plt.subplot(1, 3, 2)
    plt.plot(x, y, 'ro', label="Фактичні дані")
    y_smooth = polynomial(x_fine, best_coef)
    plt.plot(x_fine, y_smooth, 'b-', label=f"Апроксимація (m={optimal_m})")
    plt.plot(x_future, y_future, 'go', marker='*', markersize=10, label="Прогноз (екстраполяція)")
    plt.title("Апроксимація та прогноз температур")
    plt.xlabel("Місяць")
    plt.ylabel("Температура (°C)")
    plt.legend()
    plt.grid(True)

    # Графік 3: Похибка апроксимації для m=1...10
    plt.subplot(1, 3, 3)
    for m in range(1, max_degree + 1):
        if m == optimal_m:
            plt.plot(x_fine, errors_fine_dict[m], label=f"m={m} (Опт)", linewidth=2.5, color='red')
        else:
            plt.plot(x_fine, errors_fine_dict[m], label=f"m={m}", alpha=0.4)

    plt.title("Похибка апроксимації ε(x)")
    plt.xlabel("Місяць")
    plt.ylabel("Похибка |f(x) - φ(x)|")
    # Щоб не захаращувати графік, обмежуємо вісь Y
    plt.ylim(0, np.max(errors_fine_dict[optimal_m]) * 2)
    plt.legend(fontsize='small', ncol=2, loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()