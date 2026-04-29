import numpy as np
import matplotlib.pyplot as plt
import os

# ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ РОБОТИ З ФАЙЛАМИ
def save_matrix_to_file(matrix, filename):
    """Записує матрицю у текстовий файл."""
    np.savetxt(filename, matrix, fmt='%.6f', delimiter='\t')


def load_matrix_from_file(filename):
    """Зчитує матрицю з текстового файлу."""
    return np.loadtxt(filename, delimiter='\t')


def save_vector_to_file(vector, filename):
    """Записує вектор у текстовий файл."""
    np.savetxt(filename, vector, fmt='%.6f')


def load_vector_from_file(filename):
    """Зчитує вектор з текстового файлу."""
    return np.loadtxt(filename)

# МАТЕМАТИЧНІ ФУНКЦІЇ
def mat_vec_mult(A, X):
    """Обчислення добутку матриці на вектор."""
    n = len(A)
    B = np.zeros(n)
    for i in range(n):
        B[i] = sum(A[i, j] * X[j] for j in range(n))
    return B


def vector_norm(V):
    """Обчислення норми вектора (максимальний за модулем елемент)."""
    return np.max(np.abs(V))


def lu_decomposition(A):
    """Знаходження LU-розкладу матриці А."""
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Задаємо значення діагональних елементів матриці U рівними одиниці
    for i in range(n):
        U[i, i] = 1.0

    # Почергово знаходимо елементи стовпців L та рядків U
    for k in range(n):
        # Обчислення k-го стовпця матриці L
        for i in range(k, n):
            sum_lu = sum(L[i, j] * U[j, k] for j in range(k))
            L[i, k] = A[i, k] - sum_lu

        # Обчислення k-го рядка матриці U
        for j in range(k + 1, n):
            sum_lu = sum(L[k, p] * U[p, j] for p in range(k))
            U[k, j] = (A[k, j] - sum_lu) / L[k, k]

    return L, U


def solve_lu(L, U, B):
    """Розв'язок системи рівнянь AX=B за допомогою LU-розкладу."""
    n = len(B)
    Z = np.zeros(n)
    X = np.zeros(n)

    # Прямий хід: розв'язуємо LZ = B
    for i in range(n):
        sum_lz = sum(L[i, j] * Z[j] for j in range(i))
        Z[i] = (B[i] - sum_lz) / L[i, i]

    # Зворотний хід: розв'язуємо UX = Z
    for i in range(n - 1, -1, -1):
        sum_ux = sum(U[i, j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - sum_ux

    return X


def iterative_refinement(A, L, U, B, X0, eps_0=1e-14):
    """Ітераційний метод уточнення розв'язку СЛАР."""
    X_current = X0.copy()
    errors = []
    iterations = 0

    while True:
        # 1. Знаходимо вектор нев'язки: R = B - A*X_current
        R = B - mat_vec_mult(A, X_current)

        # 2. Розв'язуємо систему A*delta_X = R використовуючи готовий LU-розклад
        delta_X = solve_lu(L, U, R)

        # 3. Уточнюємо розв'язок
        X_current = X_current + delta_X

        # 4. Перевіряємо умови закінчення
        norm_delta_X = vector_norm(delta_X)
        errors.append(norm_delta_X)
        iterations += 1

        if norm_delta_X <= eps_0:
            break

        # Запобіжник від нескінченного циклу (хоча для коректних матриць збігається швидко)
        if iterations > 99:
            print("Досягнуто ліміт ітерацій!")
            break

    return X_current, iterations, errors

# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
def main():
    np.random.seed(42)  # Для відтворюваності результатів
    n = 100

    # 1. Генерація матриці А та вектора В
    print(f"Генеруємо матрицю A ({n}x{n}) та вектор B...")
    A = np.random.rand(n, n) * 10  # Випадкові числа від 0 до 10

    # Задаємо точний розв'язок x_j = 2.5
    X_true = np.full(n, 2.5)

    # Обчислюємо вектор вільних членів B
    B = mat_vec_mult(A, X_true)

    # Запис у файли
    save_matrix_to_file(A, "matrix_A.txt")
    save_vector_to_file(B, "vector_B.txt")
    print("Матрицю A та вектор B збережено у файли.\n")

    # 2. Зчитування (імітація реальної роботи за вимогами)
    A_loaded = load_matrix_from_file("matrix_A.txt")
    B_loaded = load_vector_from_file("vector_B.txt")

    # 3. Знаходження LU-розкладу
    print("Виконуємо LU-розклад...")
    L, U = lu_decomposition(A_loaded)

    # Запис LU-розкладу в файл (об'єднана матриця або окремо, запишемо окремо для зручності)
    save_matrix_to_file(L, "matrix_L.txt")
    save_matrix_to_file(U, "matrix_U.txt")

    print("Матриці L та U збережено у файли.\n")

    # 4. Розв'язок системи рівнянь AX=B
    print("Розв'язуємо систему за допомогою LU-розкладу...")
    X0 = solve_lu(L, U, B_loaded)

    # 5. Оцінка точності знайденого розв'язку (початкова похибка)
    initial_residual = B_loaded - mat_vec_mult(A_loaded, X0)
    eps_initial = vector_norm(initial_residual)
    print(f"Початкова точність (максимальна нев'язка): eps = {eps_initial:.6e}\n")

    # 6. Ітераційне уточнення розв'язку
    eps_0 = 1e-14
    print(f"Починаємо ітераційне уточнення (цільова точність: {eps_0})...")
    X_refined, iters, errors_history = iterative_refinement(A_loaded, L, U, B_loaded, X0, eps_0)

    print(f"Уточнення завершено за {iters} ітерацій!")
    print(f"Фінальна норма похибки (delta X): {errors_history[-1]:.6e}")

    # Перевірка з точним розв'язком
    print(f"Перші 5 елементів знайденого X: {X_refined[:5]}")


    # 7. Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors_history) + 1), errors_history, marker='o', linestyle='-', color='b')
    plt.yscale('log')  # Логарифмічний масштаб для осі Y, бо похибка падає експоненційно
    plt.grid(True, which="both", ls="--")
    plt.title("Збіжність ітераційного методу уточнення розв'язку СЛАР")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Норма вектора похибки ||ΔX|| (лог. масштаб)")
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=300)
    plt.show()
    print("\nГрафік збіжності побудовано та збережено як 'convergence_plot.png'.")


if __name__ == "__main__":
    main()