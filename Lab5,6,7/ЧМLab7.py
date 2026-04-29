import numpy as np
import matplotlib.pyplot as plt
import os

# 1. ГЕНЕРАЦІЯ ТА ЗБЕРЕЖЕННЯ ДАНИХ
def generate_and_save_system(n=100, exact_val=2.5, file_A="matrix_A.txt", file_B="vector_B.txt"):
    """
    Генерує матрицю А з діагональним переважанням та вектор B.
    Зберігає їх у текстові файли.
    """
    # Генеруємо випадкову матрицю від 1 до 10
    A = np.random.rand(n, n) * 10

    # Робимо діагональне переважання: додаємо до діагоналі суму модулів рядка
    for i in range(n):
        A[i, i] += np.sum(np.abs(A[i, :])) * 1.5 +  5.0

    # Точний розв'язок (за умовою x_i = 2.5)
    X_exact = np.full(n, exact_val)

    # Обчислюємо вектор вільних членів B
    B = np.dot(A, X_exact)

    # Зберігаємо у файли
    np.savetxt(file_A, A, fmt='%.6f')
    np.savetxt(file_B, B, fmt='%.6f')

    print(f"Дані згенеровано та збережено у {file_A} та {file_B}")
    return A, B


 # 2. ДОПОМІЖНІ ФУНКЦІЇ ЗГІДНО З ЗАВДАННЯМ
def read_matrix(filename):
    return np.loadtxt(filename)


def read_vector(filename):
    return np.loadtxt(filename)


def matrix_vector_mult(A, X):
    # Обчислення добутку матриці на вектор
    return np.dot(A, X)


def vector_norm(V):
    # Максимальне значення по модулю серед елементів вектора
    return np.max(np.abs(V))


def matrix_norm(A):
    # Максимальна сума модулів елементів у рядку (норма ||A||_1)
    return np.max(np.sum(np.abs(A), axis=1))



# 3. ІТЕРАЦІЙНІ МЕТОДИ
def simple_iteration_method(A, B, x0, eps=1e-14, max_iter=2000):
    n = len(B)
    x_curr = x0.copy()

    # Знаходимо параметр tau: 0 < tau < 2 / ||A||
    tau = 1.0 / matrix_norm(A)

    error_history = []

    for k in range(max_iter):
        # x^(k+1) = x^(k) - tau * (A*x^(k) - b)
        Ax = matrix_vector_mult(A, x_curr)
        x_new = x_curr - tau * (Ax - B)

        # Перевірка умови зупинки: ||X^(k+1) - X^(k)|| < eps
        error = vector_norm(x_new - x_curr)
        error_history.append(error)

        if error < eps:
            return x_new, k + 1, error_history

        x_curr = x_new

    return x_curr, max_iter, error_history


def jacobi_method(A, B, x0, eps=1e-14, max_iter=2000):
    n = len(B)
    x_curr = x0.copy()
    error_history = []

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            # Сума всіх a_ij * x_j, де j != i
            # Швидко обчислюємо як повний добуток рядка мінус діагональний елемент
            s = np.dot(A[i, :], x_curr) - A[i, i] * x_curr[i]
            x_new[i] = (B[i] - s) / A[i, i]

        error = vector_norm(x_new - x_curr)
        error_history.append(error)

        if error < eps:
            return x_new, k + 1, error_history

        x_curr = x_new.copy()

    return x_curr, max_iter, error_history


def seidel_method(A, B, x0, eps=1e-14, max_iter=2000):
    n = len(B)
    x_curr = x0.copy()
    error_history = []

    for k in range(max_iter):
        x_old = x_curr.copy()
        for i in range(n):
            # Тут використовуємо x_curr, який містить вже оновлені елементи для j < i,
            # та старі елементи для j > i. Це суть методу Зейделя!
            s1 = np.dot(A[i, :i], x_curr[:i])  # Вже нові значення
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])  # Ще старі значення

            x_curr[i] = (B[i] - s1 - s2) / A[i, i]

        error = vector_norm(x_curr - x_old)
        error_history.append(error)

        if error < eps:
            return x_curr, k + 1, error_history

    return x_curr, max_iter, error_history



# 4. ГОЛОВНА ЛОГІКА ТА ВІЗУАЛІЗАЦІЯ
def main():
    n_size = 100
    eps0 = 1e-13

    # 1. Генерація та зчитування
    generate_and_save_system(n=n_size)
    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")

    # 2. Початкове наближення
    x0 = np.full(n_size, 1.0)

    # 3. Розв'язок методами
    print(f"\nШукаємо розв'язок з точністю {eps0}...")

    x_simp, iters_simp, err_simp = simple_iteration_method(A, B, x0, eps0)
    print(f"Метод простої ітерації: {iters_simp} ітерацій. Перший x: {x_simp[0]:.5f}")

    x_jac, iters_jac, err_jac = jacobi_method(A, B, x0, eps0)
    print(f"Метод Якобі: {iters_jac} ітерацій. Перший x: {x_jac[0]:.5f}")

    x_seid, iters_seid, err_seid = seidel_method(A, B, x0, eps0)
    print(f"Метод Гауса-Зейделя: {iters_seid} ітерацій. Перший x: {x_seid[0]:.5f}")

    # 4. Побудова графіків збіжності
    plt.figure(figsize=(10, 6))

    # Використовуємо логарифмічну шкалу для осі Y, оскільки похибка падає експоненційно
    plt.semilogy(err_simp, label='Проста ітерація', linewidth=2, color='blue')
    plt.semilogy(err_jac, label='Метод Якобі', linewidth=2, linestyle='--', color='orange')
    plt.semilogy(err_seid, label='Метод Гауса-Зейделя', linewidth=2, linestyle='-.', color='green')

    plt.axhline(y=eps0, color='r', linestyle=':', label='Задана точність (1e-14)')

    plt.title('Графік швидкості збіжності ітераційних методів')
    plt.xlabel('Номер ітерації')
    plt.ylabel('Похибка ||X^(k+1) - X^(k)|| (логарифмічна шкала)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()