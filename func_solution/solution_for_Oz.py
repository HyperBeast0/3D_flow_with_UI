import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

nx, ny = 100, 100
# Инициализация поля скоростей и давления
u = np.zeros((ny, nx), dtype=np.float64)  # переменная скорости u
v = np.zeros((ny, nx), dtype=np.float64)  # переменная скорости v
p = np.zeros((ny, nx), dtype=np.float64)  # переменная давления p


def circle_calculate_slice_Oz_explicit(grid_x, grid_y, coord_x, coord_y, radius):
    # Константы
    U, V = 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    dt, dx, dy = 1, 1, 1
    dtOverDx, dtOverDy = dt / dx, dt / (2 * dy)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dx = dtOverDx
    dy = dtOverDy

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    X, Y = np.meshgrid(x, y)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_circle(x, y, cx=int(coord_x), cy=int(coord_y), r=int(radius)):
        return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2

    # Вычисление потока
    def solve_flow(u, v, p, dt, nt):
        for i in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            p_new = p.copy()

            # Поле скоростей не учитывая давление и плотность
            u[1:-1, 1:-1] = (
                    u_new[1:-1, 1:-1] - dt * (u_new[1:-1, 1:-1] * (u_new[1:-1, 1:-1] - u_new[1:-1, :-2]) / dx +
                                              v_new[1:-1, 1:-1] * (u_new[1:-1, 1:-1] - u_new[:-2, 1:-1]) / dy) +
                    kv * dt * ((u_new[1:-1, 2:] - 2 * u_new[1:-1, 1:-1] + u_new[1:-1, :-2]) / dx ** 2 +
                               (u_new[2:, 1:-1] - 2 * u_new[1:-1, 1:-1] + u_new[:-2, 1:-1]) / dy ** 2))

            v[1:-1, 1:-1] = (
                    v_new[1:-1, 1:-1] - dt * (u_new[1:-1, 1:-1] * (v_new[1:-1, 1:-1] - v_new[1:-1, :-2]) / dx +
                                              v_new[1:-1, 1:-1] * (v_new[1:-1, 1:-1] - v_new[:-2, 1:-1]) / dy) +
                    kv * dt * ((v_new[1:-1, 2:] - 2 * v_new[1:-1, 1:-1] + v_new[1:-1, :-2]) / dx ** 2 +
                               (v_new[2:, 1:-1] - 2 * v_new[1:-1, 1:-1] + v_new[:-2, 1:-1]) / dy ** 2))

            # Уравнение Пуассона
            Poisson = rho * (
                    (u_new[1:-1, 2:] - u_new[1:-1, :-2]) / (2 * dx) + (v_new[2:, 1:-1] - v_new[:-2, 1:-1]) / (
                    2 * dy)) / dt
            for _ in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1] = (((p[1:-1, 2:] + p[1:-1, :-2]) * dy ** 2 +
                                      (p[2:, 1:-1] + p[:-2, 1:-1]) * dx ** 2 -
                                      Poisson * dx ** 2 * dy ** 2) /
                                     (2 * (dx ** 2 + dy ** 2)))
                p_new[:, 0] = p_new[:, 1]  # dp/dx = 0 при x = 0
                p_new[:, -1] = p_new[:, -2]  # dp/dx = 0 при x = L
                p_new[0, :] = p_new[1, :]  # dp/dy = 0 при y = 0
                p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 при y = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

            # Применение граничных условий
            for i in range(0, nx):
                u[i, 0] = u[i, -1] = U # Граниченое условие слева и справа
                u[0, :] = 0  # Граниченое условие сверху
                u[-1, :] = 0  # Граниченое условие снизу

            v[:, 0] = v[:, -1] = v[0, :] = v[-1] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    if is_inside_circle(X[i, j], Y[i, j]):
                        u[i, j] = v[i, j] = 0

            return u, v, p

    # Вывод графика и анимация
    def plot_flow(u, v, ax):
        ax.clear()
        ax.streamplot(X, Y, u, v, color="blue")
        circle = plt.Circle((int(coord_x), int(coord_y)), radius, color='r', alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Поле скоростей возле круга. Вид сверху.')

    # Параметри для симуляции
    dt = 0.001  # Шаг по времени
    nt = 1  # Колличество шагов на кадр

    # Создание окна для вывода
    fig, ax = plt.subplots(figsize=(10, 5))

    # Анимация
    def animate(frame):
        global u, v, p
        u, v, p = solve_flow(u, v, p, dt, nt)
        plot_flow(u, v, ax)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)

    plt.show()

    plt.pause(0.001)

def circle_calculate_slice_Oz_implicit(grid_x, grid_y, coord_x, coord_y, radius):
    # Константы
    U, V = 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    dt, dx, dy = 1, 1, 1
    dtOverDx, dtOverDy = dt / dx, dt / (2 * dy)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dt_U = U / nx
    dx = dtOverDx
    dy = dtOverDy

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    X, Y = np.meshgrid(x, y)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_circle(x, y, cx=int(coord_x), cy=int(coord_y), r=int(radius)):
        return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2

    def solve_flow(u, v, p, dt, nt):
        for _ in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            p_new = p.copy()

            # Решение уравнений для u и v методом прогонки
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    # Прогонка для u
                    a = -kv * dt / dx ** 2
                    b = 1 + 2 * kv * dt / dx ** 2
                    c = -kv * dt / dx ** 2
                    d = np.zeros(ny)
                    d[j] = u_new[i, j] + dt * (
                            -np.round(u_new[i, j], 6) * (np.round(u_new[i, j], 6) - np.round(u_new[i - 1, j], 6)) / dx
                            - np.round(v_new[i, j], 6) * (np.round(u_new[i, j], 6) - np.round(u_new[i, j - 1], 6)) / dy
                    )
                    u[i, j] = solve_tridiagonal(a, b, c, d)[j]

                    # Прогонка для v
                    d = np.zeros(ny)
                    d[j] = v_new[i, j] + dt * (
                            -np.round(u_new[i, j], 6) * (np.round(v_new[i, j], 6) - np.round(v_new[i - 1, j], 6)) / dx
                            - np.round(v_new[i, j], 6) * (np.round(v_new[i, j], 6) - np.round(v_new[i, j - 1], 6)) / dy
                    )
                    v[i, j] = solve_tridiagonal(a, b, c, d)[j]

            # Уравнение Пуассона
            Poisson = rho * ((u_new[1:-1, 1:-1] - u_new[1:-1, :-2]) / (2 * dx)
                             + (v_new[2:, 1:-1] - v_new[:-2, 1:-1]) / (2 * dy)) / dt
            for i in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1] = (((p[1:-1, 2:] + p[1:-1, :-2]) * dy ** 2 +
                                      (p[2:, 1:-1] + p[:-2, 1:-1]) * dx ** 2 -
                                      Poisson * dx ** 2 * dy ** 2) /
                                     (2 * (dx ** 2 + dy ** 2)))

                p_new[:, 0] = p_new[:, 1]  # dp/dx = 0 при x = 0
                p_new[:, -1] = p_new[:, -2]  # dp/dx = 0 при x = L
                p_new[0, :] = p_new[1, :]  # dp/dy = 0 при y = 0
                p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 при y = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

            # Применение граничных условий
            for i in range(nx):
                for j in range(ny):
                    u[i, j] = U  # Граничное условие слева и справа
                    u[0, j] = U  # Граничное условие сверху
                    u[-1, j] = 0  # Граничное условие снизу

            for j in range(ny):
                u[:, j] = U  # Граничное условие по оси y

            v[:, 0] = v[:, -1] = v[0, :] = v[-1, :] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    if is_inside_circle(X[i, j], Y[i, j]):
                        u[i, j] = v[i, j] = 0

        return u, v, p

    def solve_tridiagonal(a, b, c, d):
        n = len(d)
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)

        c_prime[0] = c / b
        d_prime[0] = d[0] / b

        for i in range(1, n - 1):
            temp = b - a * c_prime[i - 1]
            c_prime[i] = c / temp
            d_prime[i] = (d[i] - a * d_prime[i - 1]) / temp

        d_prime[-1] = (d[-1] - a * d_prime[-2]) / (b - a * c_prime[-2])

        x = np.zeros(n)
        x[-1] = d_prime[-1]

        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    # Вывод графика и анимация
    def plot_flow(u, v, ax):
        ax.clear()
        ax.streamplot(X, Y, u, v, color="blue")
        circle = plt.Circle((int(coord_x), int(coord_y)), radius, color='r', alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Поле скоростей вокруг круга. Вид сверху.')

    # Параметри для симуляции
    dt = 0.001  # Шаг по времени
    nt = 1  # Колличество шагов на кадр

    # Создание окна для вывода
    fig, ax = plt.subplots(figsize=(10, 5))

    # Анимация
    def animate(frame):
        global u, v, p
        u, v, p = solve_flow(u, v, p, dt, nt)
        plot_flow(u, v, ax)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)

    plt.show()

    plt.pause(0.001)

def rectangle_calculate_slice_Oz_explicit(grid_x, grid_y, coord_x, coord_y, length, width):
    # Константы
    U, V = 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    dt, dx, dy = 1, 1, 1
    dtOverDx, dtOverDy = dt / dx, dt / (2 * dy)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dt_U = U / nx
    dx = dtOverDx
    dy = dtOverDy

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    X, Y = np.meshgrid(x, y)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_rectangle(x, y, cx=int(coord_x), cy=int(coord_y), lR=int(length), wR=int(width)):
        return abs(x - cx) < lR / 2 and abs(y - cy) < wR / 2

    # Вычисление потока
    def solve_flow(u, v, p, dt, nt):
        for _ in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            p_new = p.copy()

            # Поле скоростей не учитывая давление и плотность
            u[1:-1, 1:-1] = (
                    u_new[1:-1, 1:-1] - dt * (u_new[1:-1, 1:-1] * (u_new[1:-1, 1:-1] - u_new[1:-1, :-2]) / dx +
                                              v_new[1:-1, 1:-1] * (u_new[1:-1, 1:-1] - u_new[:-2, 1:-1]) / dy) +
                    kv * dt * ((u_new[1:-1, 2:] - 2 * u_new[1:-1, 1:-1] + u_new[1:-1, :-2]) / dx ** 2 +
                               (u_new[2:, 1:-1] - 2 * u_new[1:-1, 1:-1] + u_new[:-2, 1:-1]) / dy ** 2))

            v[1:-1, 1:-1] = (
                    v_new[1:-1, 1:-1] - dt * (u_new[1:-1, 1:-1] * (v_new[1:-1, 1:-1] - v_new[1:-1, :-2]) / dx +
                                              v_new[1:-1, 1:-1] * (v_new[1:-1, 1:-1] - v_new[:-2, 1:-1]) / dy) +
                    kv * dt * ((v_new[1:-1, 2:] - 2 * v_new[1:-1, 1:-1] + v_new[1:-1, :-2]) / dx ** 2 +
                               (v_new[2:, 1:-1] - 2 * v_new[1:-1, 1:-1] + v_new[:-2, 1:-1]) / dy ** 2))

            # Уравнение Пуассона
            Poisson = rho * (
                    (u_new[1:-1, 2:] - u_new[1:-1, :-2]) / (2 * dx) + (v_new[2:, 1:-1] - v_new[:-2, 1:-1]) / (
                    2 * dy)) / dt
            for _ in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1] = (((p[1:-1, 2:] + p[1:-1, :-2]) * dy ** 2 +
                                      (p[2:, 1:-1] + p[:-2, 1:-1]) * dx ** 2 -
                                      Poisson * dx ** 2 * dy ** 2) /
                                     (2 * (dx ** 2 + dy ** 2)))
                p_new[:, 0] = p_new[:, 1]  # dp/dx = 0 при x = 0
                p_new[:, -1] = p_new[:, -2]  # dp/dx = 0 при x = L
                p_new[0, :] = p_new[1, :]  # dp/dy = 0 при y = 0
                p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 при y = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

            # Применение граничных условий
            for i in range(0, nx):
                u[i, 0] = u[i, -1] = U  # Граниченое условие слева и справа
                u[0, :] = 0  # Граниченое условие сверху
                u[-1, :] = 0  # Граниченое условие снизу

            v[:, 0] = v[:, -1] = v[0, :] = v[-1] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    if is_inside_rectangle(X[i, j], Y[i, j]):
                        u[i, j] = v[i, j] = 0

            return u, v, p

    # Вывод графика и анимация
    def plot_flow(u, v, ax):
        ax.clear()
        ax.streamplot(X, Y, u, v, color="blue")
        rectangle = plt.Rectangle((int(coord_x) - int(length) / 2, int(coord_y) - int(width) / 2), int(length),
                                  int(width), color='r',
                                  alpha=0.5)
        ax.add_patch(rectangle)

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Поле скоростей возле прямоугольника. Вид сверху.')

    # Параметри для симуляции
    dt = 0.001  # Шаг по времени
    nt = 1  # Колличество шагов на кадр

    # Создание окна для вывода
    fig, ax = plt.subplots(figsize=(10, 5))

    # Анимация
    def animate(frame):
        global u, v, p
        u, v, p = solve_flow(u, v, p, dt, nt)
        plot_flow(u, v, ax)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)

    plt.show()

    plt.pause(0.001)

def rectangle_calculate_slice_Oz_implicit(grid_x, grid_y, coord_x, coord_y, length, width):
    # Константы
    U, V = 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    dt, dx, dy = 1, 1, 1
    dtOverDx, dtOverDy = dt / dx, dt / (2 * dy)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dt_U = U / nx
    dx = dtOverDx
    dy = dtOverDy

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    X, Y = np.meshgrid(x, y)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_rectangle(x, y, cx=int(coord_x), cy=int(coord_y), lR=int(length), wR=int(width)):
        return abs(x - cx) < lR / 2 and abs(y - cy) < wR / 2

    def solve_flow(u, v, p, dt, nt):
        for _ in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            p_new = p.copy()

            # Решение уравнений для u и v методом прогонки
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    # Прогонка для u
                    a = -kv * dt / dx ** 2
                    b = 1 + 2 * kv * dt / dx ** 2
                    c = -kv * dt / dx ** 2
                    d = np.zeros(ny)
                    d[j] = u_new[i, j] + dt * (
                            -np.round(u_new[i, j], 6) * (np.round(u_new[i, j], 6) - np.round(u_new[i - 1, j], 6)) / dx
                            - np.round(v_new[i, j], 6) * (np.round(u_new[i, j], 6) - np.round(u_new[i, j - 1], 6)) / dy
                    )
                    u[i, j] = solve_tridiagonal(a, b, c, d)[j]

                    # Прогонка для v
                    d = np.zeros(ny)
                    d[j] = v_new[i, j] + dt * (
                            -np.round(u_new[i, j], 6) * (np.round(v_new[i, j], 6) - np.round(v_new[i - 1, j], 6)) / dx
                            - np.round(v_new[i, j], 6) * (np.round(v_new[i, j], 6) - np.round(v_new[i, j - 1], 6)) / dy
                    )
                    v[i, j] = solve_tridiagonal(a, b, c, d)[j]

            # Уравнение Пуассона
            Poisson = rho * ((u_new[1:-1, 1:-1] - u_new[1:-1, :-2]) / (2 * dx)
                             + (v_new[2:, 1:-1] - v_new[:-2, 1:-1]) / (2 * dy)) / dt
            for i in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1] = (((p[1:-1, 2:] + p[1:-1, :-2]) * dy ** 2 +
                                      (p[2:, 1:-1] + p[:-2, 1:-1]) * dx ** 2 -
                                      Poisson * dx ** 2 * dy ** 2) /
                                     (2 * (dx ** 2 + dy ** 2)))

                p_new[:, 0] = p_new[:, 1]  # dp/dx = 0 при x = 0
                p_new[:, -1] = p_new[:, -2]  # dp/dx = 0 при x = L
                p_new[0, :] = p_new[1, :]  # dp/dy = 0 при y = 0
                p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 при y = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

            # Применение граничных условий
            for i in range(nx):
                for j in range(ny):
                    u[i, j] = U  # Граничное условие слева и справа
                    u[0, j] = U  # Граничное условие сверху
                    u[-1, j] = 0  # Граничное условие снизу

            for j in range(ny):
                u[:, j] = U  # Граничное условие по оси y

            v[:, 0] = v[:, -1] = v[0, :] = v[-1, :] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    if is_inside_rectangle(X[i, j], Y[i, j]):
                        u[i, j] = v[i, j] = 0

        return u, v, p

    def solve_tridiagonal(a, b, c, d):
        n = len(d)
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)

        c_prime[0] = c / b
        d_prime[0] = d[0] / b

        for i in range(1, n - 1):
            temp = b - a * c_prime[i - 1]
            c_prime[i] = c / temp
            d_prime[i] = (d[i] - a * d_prime[i - 1]) / temp

        d_prime[-1] = (d[-1] - a * d_prime[-2]) / (b - a * c_prime[-2])

        x = np.zeros(n)
        x[-1] = d_prime[-1]

        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    # Вывод графика и анимация
    def plot_flow(u, v, ax):
        ax.clear()
        ax.streamplot(X, Y, u, v, color="blue")
        rectangle = plt.Rectangle((int(coord_x) - int(length) / 2, int(coord_y) - int(width) / 2), int(length),
                                  int(width), color='r',
                                  alpha=0.5)
        ax.add_patch(rectangle)

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Поле скоростей вокруг прямоугольника. Вид сверху.')

    # Параметри для симуляции
    dt = 0.001  # Шаг по времени
    nt = 1  # Колличество шагов на кадр

    # Создание окна для вывода
    fig, ax = plt.subplots(figsize=(10, 5))

    # Анимация
    def animate(frame):
        global u, v, p
        u, v, p = solve_flow(u, v, p, dt, nt)
        plot_flow(u, v, ax)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)

    plt.show()

    plt.pause(0.001)

def circle_streamlines_Oz(grid_x, grid_y, coord_x, coord_y, radius):
    # Константы
    U, V = 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    dt, dx, dy = 1, 1, 1
    dtOverDx, dtOverDy = dt / dx, dt / (2 * dy)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dx = dtOverDx
    dy = dtOverDy

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    X, Y = np.meshgrid(x, y)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_circle(x, y, cx=int(coord_x), cy=int(coord_y), r=int(radius)):
        return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2

    # Вычисление потока
    def solve_flow(u, v, p, dt, nt):
        for i in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            p_new = p.copy()

            # Поле скоростей не учитывая давление и плотность
            u[1:-1, 1:-1] = (
                    u_new[1:-1, 1:-1] - dt * (u_new[1:-1, 1:-1] * (u_new[1:-1, 1:-1] - u_new[1:-1, :-2]) / dx +
                                              v_new[1:-1, 1:-1] * (u_new[1:-1, 1:-1] - u_new[:-2, 1:-1]) / dy) +
                    kv * dt * ((u_new[1:-1, 2:] - 2 * u_new[1:-1, 1:-1] + u_new[1:-1, :-2]) / dx ** 2 +
                               (u_new[2:, 1:-1] - 2 * u_new[1:-1, 1:-1] + u_new[:-2, 1:-1]) / dy ** 2))

            v[1:-1, 1:-1] = (
                    v_new[1:-1, 1:-1] - dt * (u_new[1:-1, 1:-1] * (v_new[1:-1, 1:-1] - v_new[1:-1, :-2]) / dx +
                                              v_new[1:-1, 1:-1] * (v_new[1:-1, 1:-1] - v_new[:-2, 1:-1]) / dy) +
                    kv * dt * ((v_new[1:-1, 2:] - 2 * v_new[1:-1, 1:-1] + v_new[1:-1, :-2]) / dx ** 2 +
                               (v_new[2:, 1:-1] - 2 * v_new[1:-1, 1:-1] + v_new[:-2, 1:-1]) / dy ** 2))

            # Уравнение Пуассона
            Poisson = rho * (
                    (u_new[1:-1, 2:] - u_new[1:-1, :-2]) / (2 * dx) + (v_new[2:, 1:-1] - v_new[:-2, 1:-1]) / (
                    2 * dy)) / dt
            for _ in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1] = (((p[1:-1, 2:] + p[1:-1, :-2]) * dy ** 2 +
                                      (p[2:, 1:-1] + p[:-2, 1:-1]) * dx ** 2 -
                                      Poisson * dx ** 2 * dy ** 2) /
                                     (2 * (dx ** 2 + dy ** 2)))
                p_new[:, 0] = p_new[:, 1]  # dp/dx = 0 при x = 0
                p_new[:, -1] = p_new[:, -2]  # dp/dx = 0 при x = L
                p_new[0, :] = p_new[1, :]  # dp/dy = 0 при y = 0
                p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 при y = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

            # Применение граничных условий
            for i in range(0, nx):
                u[i, 0] = u[i, -1] = U # Граниченое условие слева и справа
                u[0, :] = 0  # Граниченое условие сверху
                u[-1, :] = 0  # Граниченое условие снизу

            v[:, 0] = v[:, -1] = v[0, :] = v[-1] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    if is_inside_circle(X[i, j], Y[i, j]):
                        u[i, j] = v[i, j] = 0

            return u, v, p

    # Вывод графика и анимация
    def plot_flow(u, v, ax):
        ax.clear()
        ax.streamplot(X, Y, u, v, color="blue")
        circle = plt.Circle((int(coord_x), int(coord_y)), radius, color='r', alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Поле скоростей возле круга. Вид сверху.')

    # Параметри для симуляции
    dt = 0.001  # Шаг по времени
    nt = 1  # Колличество шагов на кадр

    # Создание окна для вывода
    fig, ax = plt.subplots(figsize=(10, 5))

    global u, v, p
    for i in range(20):
        u, v, p = solve_flow(u, v, p, dt, nt)

    plot_flow(u, v, ax)

    plt.show()

    plt.pause(0.001)

def rectangle_streamlines_Oz(grid_x, grid_y, coord_x, coord_y, length, width):
    # Константы
    U, V = 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    dt, dx, dy = 1, 1, 1
    dtOverDx, dtOverDy = dt / dx, dt / (2 * dy)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dt_U = U / nx
    dx = dtOverDx
    dy = dtOverDy

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    X, Y = np.meshgrid(x, y)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_rectangle(x, y, cx=int(coord_x), cy=int(coord_y), lR=int(length), wR=int(width)):
        return abs(x - cx) < lR / 2 and abs(y - cy) < wR / 2

    # Вычисление потока
    def solve_flow(u, v, p, dt, nt):
        for _ in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            p_new = p.copy()

            # Поле скоростей не учитывая давление и плотность
            u[1:-1, 1:-1] = (
                    u_new[1:-1, 1:-1] - dt * (u_new[1:-1, 1:-1] * (u_new[1:-1, 1:-1] - u_new[1:-1, :-2]) / dx +
                                              v_new[1:-1, 1:-1] * (u_new[1:-1, 1:-1] - u_new[:-2, 1:-1]) / dy) +
                    kv * dt * ((u_new[1:-1, 2:] - 2 * u_new[1:-1, 1:-1] + u_new[1:-1, :-2]) / dx ** 2 +
                               (u_new[2:, 1:-1] - 2 * u_new[1:-1, 1:-1] + u_new[:-2, 1:-1]) / dy ** 2))

            v[1:-1, 1:-1] = (
                    v_new[1:-1, 1:-1] - dt * (u_new[1:-1, 1:-1] * (v_new[1:-1, 1:-1] - v_new[1:-1, :-2]) / dx +
                                              v_new[1:-1, 1:-1] * (v_new[1:-1, 1:-1] - v_new[:-2, 1:-1]) / dy) +
                    kv * dt * ((v_new[1:-1, 2:] - 2 * v_new[1:-1, 1:-1] + v_new[1:-1, :-2]) / dx ** 2 +
                               (v_new[2:, 1:-1] - 2 * v_new[1:-1, 1:-1] + v_new[:-2, 1:-1]) / dy ** 2))

            # Уравнение Пуассона
            Poisson = rho * (
                    (u_new[1:-1, 2:] - u_new[1:-1, :-2]) / (2 * dx) + (v_new[2:, 1:-1] - v_new[:-2, 1:-1]) / (
                    2 * dy)) / dt
            for _ in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1] = (((p[1:-1, 2:] + p[1:-1, :-2]) * dy ** 2 +
                                      (p[2:, 1:-1] + p[:-2, 1:-1]) * dx ** 2 -
                                      Poisson * dx ** 2 * dy ** 2) /
                                     (2 * (dx ** 2 + dy ** 2)))
                p_new[:, 0] = p_new[:, 1]  # dp/dx = 0 при x = 0
                p_new[:, -1] = p_new[:, -2]  # dp/dx = 0 при x = L
                p_new[0, :] = p_new[1, :]  # dp/dy = 0 при y = 0
                p_new[-1, :] = p_new[-2, :]  # dp/dy = 0 при y = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

            # Применение граничных условий
            for i in range(0, nx):
                u[i, 0] = u[i, -1] = U  # Граниченое условие слева и справа
                u[0, :] = 0  # Граниченое условие сверху
                u[-1, :] = 0  # Граниченое условие снизу

            v[:, 0] = v[:, -1] = v[0, :] = v[-1] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    if is_inside_rectangle(X[i, j], Y[i, j]):
                        u[i, j] = v[i, j] = 0

            return u, v, p

    # Вывод графика и анимация
    def plot_flow(u, v, ax):
        ax.clear()
        ax.streamplot(X, Y, u, v, color="blue")
        rectangle = plt.Rectangle((int(coord_x) - int(length) / 2, int(coord_y) - int(width) / 2), int(length),
                                  int(width), color='r',
                                  alpha=0.5)
        ax.add_patch(rectangle)

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Поле скоростей возле прямоугольника. Вид сверху.')

    # Параметри для симуляции
    dt = 0.001  # Шаг по времени
    nt = 1  # Колличество шагов на кадр

    # Создание окна для вывода
    fig, ax = plt.subplots(figsize=(10, 5))

    global u, v, p
    for i in range(20):
        u, v, p = solve_flow(u, v, p, dt, nt)

    plot_flow(u, v, ax)

    plt.show()

    plt.pause(0.001)