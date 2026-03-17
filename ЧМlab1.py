import requests
import numpy as np
import matplotlib.pyplot as plt

#Отримання та підготовка даних
def fetch_and_prepare_data():
    # Запит до Open-Elevation API
    url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
    response = requests.get(url)
    data = response.json()
    results = data["results"]

    n = len(results)
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = [p["elevation"] for p in results]

    # Функція розрахунку відстані за формулою Гаверсинуса
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Радіус Землі в метрах
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Обчислення кумулятивної відстані
    distances = [0]
    for i in range(1, n):
        d = haversine(*coords[i - 1], *coords[i])
        distances.append(distances[-1] + d)

    # Запис у текстовий файл
    with open("text.txt", "w") as f:
        f.write("Latitude | Longitude | Elevation (m) | Distance (m)\n")
        for i, point in enumerate(results):
            f.write(
                f"{point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f} | {distances[i]:.2f}\n")

    return np.array(distances), np.array(elevations)

#Побудова кубічного сплайна (Метод прогонки)
def cubic_spline_natural(x, y):
    n = len(x)
    h = np.diff(x)

    A = np.zeros(n)  # Піддіагональ
    B = np.zeros(n)  # Головна діагональ
    C = np.zeros(n)  # Наддіагональ
    D = np.zeros(n)  # Праві частини

    # Граничні умови для натурального сплайна (S'' = 0 на кінцях)
    B[0] = 1
    B[-1] = 1

    # Формування системи
    for i in range(1, n - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    print("=== Коефіцієнти системи (Діагоналі) ===")
    print("Піддіагональ A:", A)
    print("Головна діагональ B:", B)
    print("Наддіагональ C:", C)
    print("Праві частини D:", D)

    # Прямий хід методу прогонки
    for i in range(1, n):
        m = A[i] / B[i - 1]
        B[i] -= m * C[i - 1]
        D[i] -= m * D[i - 1]

    # Зворотний хід
    M = np.zeros(n)  # Це другі похідні у вузлах
    M[-1] = D[-1] / B[-1]
    for i in range(n - 2, -1, -1):
        M[i] = (D[i] - C[i] * M[i + 1]) / B[i]

    # Обчислення коефіцієнтів сплайна
    a = y[:-1]
    b = np.zeros(n - 1)
    c = M[:-1] / 2
    d = np.zeros(n - 1)

    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
        d[i] = (M[i + 1] - M[i]) / (6 * h[i])

    print("\n=== Коефіцієнти сплайна C_i ===")
    print(c)
    print("\n=== Всі коефіцієнти (a, b, c, d) ===")
    for i in range(n - 1):
        print(f"Інтервал {i}: a={a[i]:.2f}, b={b[i]:.4f}, c={c[i]:.6f}, d={d[i]:.6f}")

    return a, b, c, d, x

def spline_eval(xi, a, b, c, d, x_nodes):
    # Пошук потрібного інтервалу та обчислення значення
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= xi <= x_nodes[i + 1]:
            dx = xi - x_nodes[i]
            return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return None

#Аналіз та Візуалізація
def main():
    x_full, y_full = fetch_and_prepare_data()

    # 1. Побудова сплайна для всіх точок
    a, b, c, d, x_nodes = cubic_spline_natural(x_full, y_full)

    # Генерація точок для плавного графіка
    xx = np.linspace(x_full[0], x_full[-1], 1000)
    yy_full = np.array([spline_eval(xi, a, b, c, d, x_nodes) for xi in xx])

    # 2. Оцінка впливу кількості вузлів
    def test_nodes(k):
        indices = np.linspace(0, len(x_full) - 1, k, dtype=int)
        x_k, y_k = x_full[indices], y_full[indices]
        a_k, b_k, c_k, d_k, nodes_k = cubic_spline_natural(x_k, y_k)
        yy_k = np.array([spline_eval(xi, a_k, b_k, c_k, d_k, nodes_k) for xi in xx])
        error = np.abs(yy_k - yy_full)
        return yy_k, error

    yy_10, err_10 = test_nodes(10)
    yy_15, err_15 = test_nodes(15)
    yy_20, err_20 = test_nodes(20)

    # 3. Додаткові характеристики (Енергія)
    total_ascent = sum(max(y_full[i] - y_full[i - 1], 0) for i in range(1, len(y_full)))
    energy_joules = 80 * 9.81 * total_ascent  # Для маси 80 кг
    print(f"\n=== Додатково ===")
    print(f"Сумарний набір висоти: {total_ascent:.2f} м")
    print(f"Механічна робота на підйом (80 кг): {energy_joules / 1000:.2f} кДж")

    # 4. Побудова графіків
    plt.figure(figsize=(12, 8))

    # Графік 1: Маршрут
    plt.subplot(2, 2, 1)
    plt.plot(x_full, y_full, 'o', color='green', label='GPS вузли')
    plt.plot(xx, yy_full, label='Сплайн (усі вузли)')
    plt.title('Профіль висоти маршруту')
    plt.xlabel('Відстань (м)')
    plt.ylabel('Висота (м)')
    plt.legend()
    plt.grid(True)

    # Графік 2: Порівняння вузлів
    plt.subplot(2, 2, 2)
    plt.plot(xx, yy_full, label='Еталон')
    plt.plot(xx, yy_10, label='10 вузлів')
    plt.plot(xx, yy_15, label='15 вузлів')
    plt.plot(xx, yy_20, label='20 вузлів')
    plt.title('Вплив кількості вузлів')
    plt.legend()
    plt.grid(True)

    # Графік 3: Похибка
    plt.subplot(2, 2, 3)
    plt.plot(xx, err_10, label='10 вузлів')
    plt.plot(xx, err_15, label='15 вузлів')
    plt.plot(xx, err_20, label='20 вузлів')
    plt.title('Похибка апроксимації')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()