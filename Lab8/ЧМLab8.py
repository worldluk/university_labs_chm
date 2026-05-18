import numpy as np
import matplotlib.pyplot as plt
import cmath  # Потрібен для роботи з комплексними числами в методі парабол

# ЧАСТИНА 1: Базові функції та однокрокові методи
def F(x):
    """Трансцендентна функція: F(x) = x^2 - 3*sin(x) - 1"""
    return x ** 2 - 3 * np.sin(x) - 1


def dF(x):
    return 2 * x - 3 * np.cos(x)


def d2F(x):
    return 2 + 3 * np.sin(x)


def tabulate_to_file(a, b, h, filename="tabulation.txt"):
    """Крок 1: Табуляція значень функції"""
    x_vals = np.arange(a, b + h, h)
    y_vals = F(x_vals)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("x\t\tF(x)\n")
        f.write("-" * 25 + "\n")
        for x, y in zip(x_vals, y_vals):
            f.write(f"{x:.4f}\t\t{y:.4f}\n")


def check_stop(x_next, x_curr, eps):
    """Критерій зупинки"""
    return abs(F(x_next)) < eps and abs(x_next - x_curr) < eps


def simple_iteration(x0, eps=1e-10, tau=0.1):
    x = x0
    iters = 0
    while True:
        x_next = x + tau * F(x)
        iters += 1
        if check_stop(x_next, x, eps):
            return x_next, iters
        x = x_next


def newton_method(x0, eps=1e-10):
    x = x0
    iters = 0
    while True:
        x_next = x - F(x) / dF(x)
        iters += 1
        if check_stop(x_next, x, eps):
            return x_next, iters
        x = x_next


def chebyshev_method(x0, eps=1e-10):
    x = x0
    iters = 0
    while True:
        fx = F(x)
        dfx = dF(x)
        d2fx = d2F(x)
        x_next = x - fx / dfx - (fx ** 2 * d2fx) / (2 * dfx ** 3)
        iters += 1
        if check_stop(x_next, x, eps):
            return x_next, iters
        x = x_next


def chord_method(x0, x1, eps=1e-10):
    iters = 0
    while True:
        fx1 = F(x1)
        fx0 = F(x0)
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        iters += 1
        if check_stop(x_next, x1, eps):
            return x_next, iters
        x0, x1 = x1, x_next



# ЧАСТИНА 1.5: Багатокрокові ітераційні методи
def parabola_method(x_n2, x_n1, x_n, eps=1e-10):
    """Метод парабол (Мюллера)"""
    iters = 0
    while True:
        # Розділені різниці
        f_n_n1 = (F(x_n) - F(x_n1)) / (x_n - x_n1)
        f_n1_n2 = (F(x_n1) - F(x_n2)) / (x_n1 - x_n2)
        f_n_n1_n2 = (f_n_n1 - f_n1_n2) / (x_n - x_n2)

        # Коефіцієнти квадратного рівняння для дельти
        A = f_n_n1_n2
        B = (x_n - x_n1) * A + f_n_n1
        C = F(x_n)

        # Дискримінант (використовуємо cmath, бо корінь може бути від'ємним)
        D = cmath.sqrt(B ** 2 - 4 * A * C)

        # Знаходимо дві можливі дельти
        if abs(A) == 0:  # Захист від ділення на 0
            break

        delta1 = (-B + D) / (2 * A)
        delta2 = (-B - D) / (2 * A)

        # Вибираємо найменшу дельту за модулем
        delta = delta1 if abs(delta1) < abs(delta2) else delta2

        # Наступне наближення (беремо тільки дійсну частину)
        x_next = (x_n + delta).real

        iters += 1
        if check_stop(x_next, x_n, eps):
            return x_next, iters

        x_n2, x_n1, x_n = x_n1, x_n, x_next


def inverse_interpolation_method(x_n2, x_n1, x_n, eps=1e-10):
    """Метод зворотної інтерполяції (базується на поліномі Лагранжа)"""
    iters = 0
    while True:
        y_n2, y_n1, y_n = F(x_n2), F(x_n1), F(x_n)

        # Знаменники згідно з формулою зворотної інтерполяції Лагранжа
        d0 = (y_n2 - y_n1) * (y_n2 - y_n)
        d1 = (y_n1 - y_n2) * (y_n1 - y_n)
        d2 = (y_n - y_n2) * (y_n - y_n1)

        # Перевірка на ділення на 0 (якщо y стали однаковими)
        if abs(d0) < 1e-15 or abs(d1) < 1e-15 or abs(d2) < 1e-15:
            break

        # Обчислення нового x через суму трьох дробів
        term1 = (y_n1 * y_n / d0) * x_n2
        term2 = (y_n2 * y_n / d1) * x_n1
        term3 = (y_n2 * y_n1 / d2) * x_n

        x_next = term1 + term2 + term3
        iters += 1

        if check_stop(x_next, x_n, eps):
            return x_next, iters

        x_n2, x_n1, x_n = x_n1, x_n, x_next



# ЧАСТИНА 2: Алгебраїчні рівняння
def horner_newton(coeffs, x0, eps=1e-10):
    x = x0
    iters = 0
    while True:
        b = [coeffs[0]]
        for a in coeffs[1:]:
            b.append(a + x * b[-1])
        b0 = b[-1]

        c = [b[0]]
        for b_val in b[1:-1]:
            c.append(b_val + x * c[-1])
        c1 = c[-1]

        x_next = x - b0 / c1
        iters += 1

        if abs(b0) < eps and abs(x_next - x) < eps:
            return x_next, iters
        x = x_next


def lin_method(coeffs, p0, q0, eps=1e-10):
    p, q = p0, q0
    iters = 0
    n = len(coeffs) - 1

    while True:
        b = np.zeros(n + 1)
        b[0] = coeffs[0]
        b[1] = coeffs[1] - p * b[0]

        for i in range(2, n - 1):
            b[i] = coeffs[i] - p * b[i - 1] - q * b[i - 2]

        q1 = coeffs[-1] / b[-3]
        p1 = (coeffs[-2] * b[-3] - coeffs[-1] * b[-4]) / (b[-3] ** 2)

        iters += 1
        if abs(p1 - p) < eps and abs(q1 - q) < eps:
            alpha = -p1 / 2
            beta = np.sqrt(q1 - alpha ** 2)
            return complex(alpha, beta), complex(alpha, -beta), iters
        p, q = p1, q1



# ЧАСТИНА 3: Функції побудови графіків
def plot_final_graph(a, b, h, found_roots):
    x_vals = np.arange(a, b + h, h)
    y_vals = F(x_vals)

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, label="F(x) = x^2 - 3sin(x) - 1", color='blue')
    plt.axhline(0, color='black', linewidth=1)

    for i, root in enumerate(found_roots):
        plt.plot(root, F(root), 'ro', markersize=8, label=f'Корінь {i + 1}: x~={root:.3f}')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Локалізація та уточнення коренів")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.savefig("transcendental_plot.png")
    plt.show(block=False)


def plot_iterations_comparison(methods_data):
    names = list(methods_data.keys())
    iters = list(methods_data.values())

    plt.figure(figsize=(12, 6))
    colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FF9800', '#E91E63', '#00BCD4']
    bars = plt.bar(names, iters, color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, int(yval), ha='center', va='bottom', fontweight='bold')

    plt.title("Порівняння ефективності всіх 6 методів (Корінь 1)", fontsize=14, fontweight='bold')
    plt.ylabel("Кількість ітерацій (менше = краще)", fontsize=12)
    plt.xticks(rotation=15)  # Трохи нахиляємо текст, щоб влізли всі назви
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("methods_comparison.png")
    plt.show()



# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
if __name__ == "__main__":
    print("=== ЧАСТИНА 1: ТРАНСЦЕНДЕНТНЕ РІВНЯННЯ ===")
    tabulate_to_file(-2, 3, 0.1)

    # Визначаємо точки
    x_root1 = -0.5
    x_root2 = 2.0

    iters_compare = {}

    print(f"\n--- Дослідження кореня 1 (окіл {x_root1}) ---")

    root, iters = simple_iteration(x_root1, tau=0.1)
    iters_compare["Проста ітерація"] = iters
    print(f"Метод простої ітерації: корінь = {root:.10f}, ітерацій = {iters}")

    root, iters = newton_method(x_root1)
    iters_compare["Ньютона"] = iters
    print(f"Метод Ньютона: корінь = {root:.10f}, ітерацій = {iters}")

    root, iters = chebyshev_method(x_root1)
    iters_compare["Чебишева"] = iters
    print(f"Метод Чебишева: корінь = {root:.10f}, ітерацій = {iters}")

    # Для багатокрокових методів задаємо кілька сусідніх точок
    root, iters = chord_method(x_root1 - 0.2, x_root1)
    iters_compare["Хорд"] = iters
    print(f"Метод Хорд: корінь = {root:.10f}, ітерацій = {iters}")

    root, iters = parabola_method(x_root1 - 0.2, x_root1 - 0.1, x_root1)
    iters_compare["Парабол"] = iters
    print(f"Метод парабол: корінь = {root:.10f}, ітерацій = {iters}")

    root, iters = inverse_interpolation_method(x_root1 - 0.2, x_root1 - 0.1, x_root1)
    iters_compare["Звор. інтерполяція"] = iters
    print(f"Метод зворотної інтерп.: корінь = {root:.10f}, ітерацій = {iters}")

    print(f"\n--- Дослідження кореня 2 (окіл {x_root2}) ---")
    root2_newton, iters = newton_method(x_root2)
    print(f"Метод Ньютона: корінь = {root2_newton:.10f}, ітерацій = {iters}")

    print("\n=== ЧАСТИНА 2: АЛГЕБРАЇЧНЕ РІВНЯННЯ ===")
    coeffs_file = "coeffs.txt"
    with open(coeffs_file, "w") as f:
        f.write("1 -4 6 -4")

    with open(coeffs_file, "r") as f:
        coeffs = [float(x) for x in f.read().split()]

    real_root, iters_h = horner_newton(coeffs, x0=1.5)
    print(f"Метод Ньютона-Горнера: x = {real_root:.10f}, ітерацій = {iters_h}")

    c1, c2, iters_l = lin_method(coeffs, p0=-1, q0=1)
    print(f"Метод Ліна: x1 = {c1:.4f}, x2 = {c2:.4f}, ітерацій = {iters_l}")

    print("\nВсі обчислення завершено. Відкриваю графіки...")
    plot_final_graph(-2, 3, 0.1, found_roots=[root, root2_newton])
    plot_iterations_comparison(iters_compare)