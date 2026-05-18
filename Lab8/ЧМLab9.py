import numpy as np
import matplotlib.pyplot as plt



# ЧАСТИНА 1: ФУНКЦІЇ (КОЛО + ГІПЕРБОЛА)
def f1(X):
    """Перше рівняння системи: x1^2 + x2^2 - 4 = 0 (Коло)"""
    return X[0] ** 2 + X[1] ** 2 - 4


def f2(X):
    """Друге рівняння системи: x1^2 - x2^2 - 1 = 0 (Гіпербола)"""
    return X[0] ** 2 - X[1] ** 2 - 1


def system_target_func(X):
    """Цільова функція: Ф(X) = f1(X)^2 + f2(X)^2"""
    return f1(X) ** 2 + f2(X) ** 2


# ==========================================
# ЧАСТИНА 2: АЛГОРИТМ ХУКА-ДЖИВСА
# ==========================================
def hooke_jeeves(func, x0, initial_step, eps1, eps2, q=2.0, p=2.0, max_iter=1000):
    X0 = np.array(x0, dtype=float)
    delta_X = np.array(initial_step, dtype=float)
    trajectory = [X0.copy()]
    steps_count = 0

    def exploratory_search(base_X, current_delta, allow_reduction=True):
        X_new = base_X.copy()
        delta_temp = current_delta.copy()
        for i in range(len(X_new)):
            while True:
                f_current = func(X_new)
                X_try = X_new.copy()
                X_try[i] += delta_temp[i]
                if func(X_try) < f_current:
                    X_new = X_try
                    break
                X_try = X_new.copy()
                X_try[i] -= delta_temp[i]
                if func(X_try) < f_current:
                    X_new = X_try
                    break
                if allow_reduction:
                    delta_temp[i] /= q
                    if delta_temp[i] < eps1:
                        break
                else:
                    break
        return X_new, delta_temp

    for _ in range(max_iter):
        steps_count += 1
        X1, delta_X = exploratory_search(X0, delta_X, allow_reduction=True)

        if np.linalg.norm(delta_X) < eps1 and abs(func(X1) - func(X0)) < eps2:
            break
        if np.array_equal(X1, X0):
            break

        X2_p = X1 + p * (X1 - X0)
        X2, _ = exploratory_search(X2_p, delta_X, allow_reduction=False)

        if func(X2) < func(X1):
            X0 = X1.copy()
            X1 = X2.copy()
        else:
            X0 = X1.copy()

        trajectory.append(X0.copy())
    return trajectory, steps_count



# ЧАСТИНА 3: ВІЗУАЛІЗАЦІЯ
# ЗМІНА 1: Додаємо новий аргумент steps_count у функцію
def plot_custom_trajectory(trajectory, steps_count):
    bounds = [-3, 3, -3, 3]
    x = np.linspace(bounds[0], bounds[1], 400)
    y = np.linspace(bounds[2], bounds[3], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = system_target_func([X[i, j], Y[i, j]])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    levels = [0.1, 0.5, 2.0, 5.0, 15.0, 40.0, 100.0]
    cp = ax.contour(X, Y, Z, levels=levels, colors='#6d597a', alpha=0.7, linewidths=1.5)
    ax.clabel(cp, inline=True, fontsize=9, fmt='%1.1f')

    Z_f1 = X ** 2 + Y ** 2 - 4
    Z_f2 = X ** 2 - Y ** 2 - 1

    ax.contour(X, Y, Z_f1, levels=[0], colors='blue', linewidths=2.5, linestyles='dashed')
    ax.contour(X, Y, Z_f2, levels=[0], colors='red', linewidths=2.5, linestyles='dashed')

    ax.plot([], [], color='blue', linestyle='dashed', linewidth=2.5, label='$x_1^2 + x_2^2 - 4 = 0$')
    ax.plot([], [], color='red', linestyle='dashed', linewidth=2.5, label='$x_1^2 - x_2^2 - 1 = 0$')

    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], color='black', marker='o', markersize=5, label='Траєкторія спуску')
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=9, label='Початкова точка X(0)')
    ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=14, label="Знайдений розв'язок X*")

    ax.set_title("Метод Хука-Дживса: Нова система (Коло та Гіпербола)", fontsize=14)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    # --- ЗМІНА 2: Додаємо текстове поле з кількістю кроків ---
    # boxstyle='round' робить кути заокругленими, facecolor - колір фону поля
    text_str = f"Кількість кроків: {steps_count}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

    # Координати 0.05, 0.95 - це лівий верхній кут графіка
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    # ---------------------------------------------------------

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    ax.grid(True)



# ЧАСТИНА 4: ВИКОНАННЯ ТА ЗАПИС У ФАЙЛ
if __name__ == "__main__":
    x0_sys = [-2.0, -2.0]
    step = [0.5, 0.5]
    eps1, eps2 = 1e-4, 1e-4

    traj_sys, steps_sys = hooke_jeeves(system_target_func, x0_sys, step, eps1, eps2)

    print("--- Розв'язок системи рівнянь ---")
    print(f"Кількість кроків: {steps_sys}")
    print(f"Знайдений розв'язок: x1 = {traj_sys[-1][0]:.4f}, x2 = {traj_sys[-1][1]:.4f}")

    with open("trajectory.txt", "w", encoding='utf-8') as f:
        f.write("Траєкторія спуску для системи рівнянь:\n")
        for i, point in enumerate(traj_sys):
            f.write(f"Крок {i}: x1 = {point[0]:.6f}, x2 = {point[1]:.6f}\n")
    print("Траєкторію успішно збережено у файл 'trajectory.txt'.")

    # ЗМІНА 3: Передаємо steps_sys у функцію малювання
    plot_custom_trajectory(traj_sys, steps_sys)
    plt.show()