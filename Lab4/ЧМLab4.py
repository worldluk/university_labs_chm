import math
import numpy as np
import matplotlib.pyplot as plt

# ЧАСТИНА 1: МАТЕМАТИЧНІ ФУНКЦІЇ
def M(t):
    """Модель вологості ґрунту: M(t) = 50 * e^(-0.1*t) + 5 * sin(t)"""
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)


def exact_derivative(t):
    """Точна аналітична похідна (швидкість висихання ґрунту)"""
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)


def forward_difference(func, x0, h):
    """Права різниця (крок вперед)"""
    return (func(x0 + h) - func(x0)) / h


def backward_difference(func, x0, h):
    """Ліва різниця (крок назад)"""
    return (func(x0) - func(x0 - h)) / h


def central_difference(func, x0, h):
    """Центральна різниця (найбільш збалансований метод)"""
    return (func(x0 + h) - func(x0 - h)) / (2 * h)


def runge_romberg(d_h, d_2h):
    """Уточнення методом Рунге-Ромберга (для формули 2-го порядку)"""
    return d_h + (d_h - d_2h) / 3


def aitken_method(d_h, d_2h, d_4h):
    """Уточнення методом Ейткена"""
    numerator = (d_2h ** 2) - (d_4h * d_h)
    denominator = 2 * d_2h - (d_4h + d_h)

    if abs(denominator) < 1e-15:
        return d_h
    return numerator / denominator


def aitken_accuracy_order(d_h, d_2h, d_4h):
    """Оцінка порядку точності p за Ейткеном"""
    numerator = abs(d_4h - d_2h)
    denominator = abs(d_2h - d_h)

    if denominator < 1e-15 or numerator < 1e-15:
        return 0.0
    return (1 / math.log(2)) * math.log(numerator / denominator)


# ЧАСТИНА 2: ПОБУДОВА ВСІХ ГРАФІКІВ НА ОДНОМУ ПОЛОТНІ
def plot_all_graphs(t0, exact_val, h_large=1.0):
    """Генерує єдине вікно з трьома підграфіками (subplots)"""

    # Створюємо полотно: 3 рядки, 1 колонка. Розмір 10 на 16 дюймів
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16))

    #Графік 1: Динаміка вологості та швидкості
    t_vals = np.linspace(0, 10, 400)
    M_vals = [M(t) for t in t_vals]
    dM_vals = [exact_derivative(t) for t in t_vals]

    ax1.plot(t_vals, M_vals, label="M(t) - Рівень вологості", color='#0033cc', linewidth=2)
    ax1.plot(t_vals, dM_vals, label="M'(t) - Швидкість зміни (Точна)", color='#cc0000', linewidth=2)
    ax1.set_xlabel("Час (t)")
    ax1.set_ylabel("Значення")
    ax1.set_title("1. Динаміка вологості ґрунту та швидкості її зміни")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    #Графік 2: Залежність похибки від кроку h
    h_values = np.logspace(-15, 1, 100)
    errors = []
    valid_h = []

    for h in h_values:
        approx_val = central_difference(M, t0, h)
        error = abs(approx_val - exact_val)
        if error > 0:
            errors.append(error)
            valid_h.append(h)

    ax2.loglog(valid_h, errors, marker='.', linestyle='-', color='purple')
    ax2.set_xlabel("Розмір кроку h (Логарифмічна шкала)")
    ax2.set_ylabel("Абсолютна похибка R")
    ax2.set_title("2. Пошук оптимального кроку h (Регуляризація)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    #Графік 3: Порівняння методів при грубому кроці
    t_vals_comp = np.linspace(0.1, 9.9, 400)
    exact_vals_comp = [exact_derivative(t) for t in t_vals_comp]
    forward_vals = [forward_difference(M, t, h_large) for t in t_vals_comp]
    backward_vals = [backward_difference(M, t, h_large) for t in t_vals_comp]
    central_vals = [central_difference(M, t, h_large) for t in t_vals_comp]

    ax3.plot(t_vals_comp, exact_vals_comp, label="Еталон (Точна похідна)", color='black', linewidth=2.5)
    ax3.plot(t_vals_comp, central_vals, label=f"Центральна різниця (h={h_large})", color='green', linestyle='--',
             linewidth=2)
    ax3.plot(t_vals_comp, forward_vals, label=f"Права різниця (Вперед)", color='red', linestyle='-.', alpha=0.7)
    ax3.plot(t_vals_comp, backward_vals, label=f"Ліва різниця (Назад)", color='blue', linestyle=':', alpha=0.7)

    ax3.set_xlabel("Час (t)")
    ax3.set_ylabel("Значення похідної")
    ax3.set_title(f"3. Чому центральна різниця краща? (Порівняння при h={h_large})")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.7)


    # Додаємо h_pad=4.0, щоб примусово зробити більший відступ по висоті між графіками
    plt.tight_layout(h_pad=4.0)

    # Зберігаємо та показуємо
    plt.savefig('all_graphs_combined.png', dpi=300)
    plt.show()

#ЧАСТИНА 3: ОСНОВНА ПРОГРАМА ТА РОЗРАХУНКИ
def main():
    t0 = 1.0
    exact_val = exact_derivative(t0)

    print("=" * 60)
    print(" ЛАБОРАТОРНА РОБОТА: ЧИСЕЛЬНЕ ДИФЕРЕНЦІЮВАННЯ ")
    print("=" * 60)

    # 1. Малюємо об'єднаний графік
    print("\n[!] Генерую графіки (закрийте вікно графіка, щоб програма продовжила розрахунки)...")
    plot_all_graphs(t0, exact_val, h_large=1.0)

    # 2. Розрахунки у консоль
    print(f"\n1. Точне значення похідної M'(1): {exact_val:.7f}")

    # Шукаємо оптимальне h
    h_test_values = np.logspace(-15, 1, 1000)
    min_error = float('inf')
    best_h = None

    for h_test in h_test_values:
        approx = central_difference(M, t0, h_test)
        current_error = abs(approx - exact_val)
        if current_error < min_error:
            min_error = current_error
            best_h = h_test

    print(f"\n2. Оптимальний крок h0: {best_h:.1e}")
    print(f"   Досягнута мінімальна похибка R0: {min_error:.2e}")

    # Фіксований крок за завданням
    h = 1e-3
    d_h = central_difference(M, t0, h)
    R1 = abs(d_h - exact_val)
    print(f"\n3. Базова апроксимація (при h = 0.001): {d_h:.7f}")
    print(f"   Похибка R1: {R1:.7e}")

    # Рунге-Ромберг
    d_2h = central_difference(M, t0, 2 * h)
    d_runge = runge_romberg(d_h, d_2h)
    R2 = abs(d_runge - exact_val)

    print("\n--- ПОКРАЩЕННЯ ТОЧНОСТІ ---")
    print(f"> Метод Рунге-Ромберга: {d_runge:.7f}")
    print(f"  Похибка R2: {R2:.7e}")

    # Ейткен
    d_4h = central_difference(M, t0, 4 * h)
    d_aitken = aitken_method(d_h, d_2h, d_4h)
    R3 = abs(d_aitken - exact_val)
    p_order = aitken_accuracy_order(d_h, d_2h, d_4h)

    print(f"\n> Метод Ейткена: {d_aitken:.7f}")
    print(f"  Похибка R3: {R3:.7e}")
    print(f"  Оцінений порядок точності p: {p_order:.2f}")
    print("=" * 60)


# Запуск програми
if __name__ == "__main__":
    main()