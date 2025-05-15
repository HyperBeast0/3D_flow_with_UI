import matplotlib
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from tkinter.messagebox import showwarning
from tkinter import ttk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from func_solution.solution_for_Oy import circle_calculate_slice_Oy_explicit, rectangle_calculate_slice_Oy_explicit, \
    rectangle_calculate_slice_Oy_implicit, circle_calculate_slice_Oy_implicit, \
    circle_streamlines_Oy, rectangle_streamlines_Oy
from func_solution.solution_for_Oy_inv import circle_calculate_slice_Oy_inv_explicit, rectangle_calculate_slice_Oy_inv_explicit, \
    circle_calculate_slice_Oy_inv_implicit, rectangle_calculate_slice_Oy_inv_implicit, \
    circle_streamlines_Oy_inv, rectangle_streamlines_Oy_inv
from func_solution.solution_for_Oz import circle_calculate_slice_Oz_explicit, rectangle_calculate_slice_Oz_explicit, \
    circle_calculate_slice_Oz_implicit, rectangle_calculate_slice_Oz_implicit, \
    circle_streamlines_Oz, rectangle_streamlines_Oz
from func_solution.solution_for_Oz_inv import circle_calculate_slice_Oz_inv_explicit, rectangle_calculate_slice_Oz_inv_explicit, \
    circle_calculate_slice_Oz_inv_implicit, rectangle_calculate_slice_Oz_inv_implicit, \
    circle_streamlines_Oz_inv, rectangle_streamlines_Oz_inv
matplotlib.use('TkAgg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt

nx, ny, nz = 30, 30, 30
# Инициализация поля скоростей и давления
u = np.zeros((ny, nx, nz), dtype=np.float64)  # переменная скорости u
v = np.zeros((ny, nx, nz), dtype=np.float64)  # переменная скорости v
w = np.zeros((ny, nx, nz), dtype=np.float64)  # переменная скорости w
p = np.zeros((ny, nx, nz), dtype=np.float64)  # переменная давления p

#Расчеты явной схемы для сферы
def sphere_calculate_explicit(grid_x, grid_y, grid_z, coord_x, coord_y, coord_z, radius):
    # Константы
    U, V, W = 1.0, 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny, nz
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    Grid_Z = int(grid_z)
    dt, dx, dy, dz = 1, 1, 1, 1
    dtOverDx, dtOverDy, dtOverDz = dt / dx, dt / (2 * dy), dt / (2 * dz)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)
        dtOverDz = dt / (2 * dz)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dt_U = U / nx
    dx = dtOverDx
    dy = dtOverDy
    dz = dtOverDz

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    z = np.linspace(0, Grid_Z, nz)
    X, Y, Z = np.meshgrid(x, y, z)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_circle(x, y, z, cx=int(coord_x), cy=int(coord_y), cz=int(coord_z), r=int(radius)):
        return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r ** 2

    # Вычисление потока
    def solve_flow(u, v, w, p, dt, nt):
        for _ in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            w_new = w.copy()
            p_new = p.copy()

            # Поле скоростей не учитывая давление и плотность
            u[1:-1, 1:-1, 1:-1] = (
                    u_new[1:-1, 1:-1, 1:-1] - dt * (
                    np.round(u_new[1:-1, 1:-1, 1:-1], 6) * (
                        np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, :-2], 6)) / dx +
                    np.round(v_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, 1:-1], 6)) / dy +
                    np.round(w_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(w_new[1:-1, 1:-1, 1:-1], 6) - np.round(w_new[1:-1, 1:-1, 1:-1], 6)) / dz) +
                    kv * dt * ((u_new[1:-1, 1:-1, 2:] - 2 * u_new[1:-1, 1:-1, 1:-1] + u_new[1:-1, 1:-1,
                                                                                      :-2]) / dx ** 2 +
                               (u_new[2:, 1:-1, 1:-1] - 2 * u_new[1:-1, 1:-1, 1:-1] + u_new[:-2, 1:-1,
                                                                                      1:-1]) / dy ** 2 +
                               (u_new[1:-1, 2:, 1:-1] - 2 * u_new[1:-1, 1:-1, 1:-1] + u_new[1:-1, :-2,
                                                                                      1:-1]) / dz ** 2))

            v[1:-1, 1:-1, 1:-1] = (
                    v_new[1:-1, 1:-1, 1:-1] - dt * (
                    np.round(u_new[1:-1, 1:-1, 1:-1], 6) * (
                        np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, :-2], 6)) / dx +
                    np.round(v_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, 1:-1], 6)) / dy +
                    np.round(w_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(w_new[1:-1, 1:-1, 1:-1], 6) - np.round(w_new[1:-1, 1:-1, 1:-1], 6)) / dz) +
                    kv * dt * ((v_new[1:-1, 1:-1, 2:] - 2 * v_new[1:-1, 1:-1, 1:-1] + v_new[1:-1, 1:-1,
                                                                                      :-2]) / dx ** 2 +
                               (v_new[2:, 1:-1, 1:-1] - 2 * v_new[1:-1, 1:-1, 1:-1] + v_new[:-2, 1:-1,
                                                                                      1:-1]) / dy ** 2 +
                               (v_new[1:-1, 2:, 1:-1] - 2 * v_new[1:-1, 1:-1, 1:-1] + v_new[1:-1, :-2,
                                                                                      1:-1]) / dz ** 2))

            w[1:-1, 1:-1, 1:-1] = (
                    w_new[1:-1, 1:-1, 1:-1] - dt * (
                    np.round(u_new[1:-1, 1:-1, 1:-1], 6) * (
                        np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, :-2], 6)) / dx +
                    np.round(v_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, 1:-1], 6)) / dy +
                    np.round(w_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(w_new[1:-1, 1:-1, 1:-1], 6) - np.round(w_new[1:-1, 1:-1, 1:-1], 6)) / dz) +
                    kv * dt * ((w_new[1:-1, 1:-1, 2:] - 2 * w_new[1:-1, 1:-1, 1:-1] + w_new[1:-1, 1:-1,
                                                                                      :-2]) / dx ** 2 +
                               (w_new[2:, 1:-1, 1:-1] - 2 * w_new[1:-1, 1:-1, 1:-1] + w_new[:-2, 1:-1,
                                                                                      1:-1]) / dy ** 2 +
                               (w_new[1:-1, 2:, 1:-1] - 2 * w_new[1:-1, 1:-1, 1:-1] + w_new[1:-1, :-2,
                                                                                      1:-1]) / dz ** 2))

            # Уравнение Пуассона
            Poisson = rho * ((u_new[1:-1, 1:-1, 2:] - u_new[1:-1, 1:-1, :-2]) / (2 * dx)
                             + (v_new[2:, 1:-1, 1:-1] - v_new[:-2, 1:-1, 1:-1]) /
                             (2 * dy) + (w_new[1:-1, 2:, 1:-1] + w_new[1:-1, :-2, 1:-1]) / (2 * dz)) / dt
            for i in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1, 1:-1] = (((p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) * dy ** 2 +
                                            (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * dx ** 2 +
                                            (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) * dz ** 2 -
                                            Poisson * dx ** 2 * dy ** 2 * dz ** 2) /
                                           (2 * (dx ** 2 + dy ** 2 + dz ** 2)))

                p_new[:, 0, :] = p_new[:, 1, :]  # dp/dx = 0 при x = 0
                p_new[:, -1, :] = p_new[:, -2, :]  # dp/dx = 0 при x = L
                p_new[0, :, :] = p_new[1, :, :]  # dp/dy = 0 при y = 0
                p_new[-1, :, :] = p_new[-2, :, :]  # dp/dy = 0 при y = L
                p_new[:, :, 0] = p_new[:, :, 1]  # dp/dz = 0 при z = 0
                p_new[:, :, -1] = p_new[:, :, -2]  # dp/dz = 0 при z = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * dy)
            w[1:-1, 1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * dz)

            # Применение граничных условий
            for i in range(nx):
                for j in range(ny):
                    u[i, j, 0] = u[i, j, -1] = U - i * dt_U  # Граничное условие слева и справа
                    u[0, j, :] = U  # Граничное условие сверху
                    u[-1, j, :] = 0  # Граничное условие снизу

            for j in range(ny):
                for k in range(nz):
                    u[:, j, k] = U  # Граничное условие по оси y

            v[:, 0, :] = v[:, -1, :] = v[0, :, :] = v[-1, :, :] = 0
            w[:, 0, :] = w[:, -1, :] = w[0, :, :] = w[-1, :, :] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    for k in range(nz):
                        if is_inside_circle(X[i, j, k], Y[i, j, k], Z[i, j, k]):
                            u[i, j, k] = v[i, j, k] = w[i, j, k] = 0

            return u, v, w, p

    # Вывод графика и анимация
    def plot_flow(u, v, w, ax):
        ax.clear()
        ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True)

        sphere_x = int(radius) * np.outer(np.cos(x), np.sin(y)) + int(coord_x)
        sphere_y = int(radius) * np.outer(np.sin(x), np.sin(y)) + int(coord_y)
        sphere_z = int(radius) * np.outer(np.ones(np.size(x)), np.cos(y)) + int(coord_z)

        ax.plot_surface(sphere_x, sphere_y, sphere_z, rstride=4, cstride=4, color='r')

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_zlim(0, Grid_Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Поле скоростей возле сферы')

    # Параметры для симуляции
    nt = 1  # Количество шагов на кадр

    # Создание окна для вывода
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Анимация
    def animate(frame):
        global u, v, w, p
        u, v, w, p = solve_flow(u, v, w, p, dt, nt)
        plot_flow(u, v, w, ax)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)

    plt.show()

    plt.pause(0.001)

#Расчеты неявной схемы для сферы
def sphere_calculate_implicit(grid_x, grid_y, grid_z, coord_x, coord_y, coord_z, radius):
    # Константы
    U, V, W = 1.0, 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny, nz
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    Grid_Z = int(grid_z)
    dt, dx, dy, dz = 1, 1, 1, 1
    dtOverDx, dtOverDy, dtOverDz = dt / dx, dt / (2 * dy), dt / (2 * dz)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)
        dtOverDz = dt / (2 * dz)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dt_U = U / nx
    dx = dtOverDx
    dy = dtOverDy
    dz = dtOverDz

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    z = np.linspace(0, Grid_Z, nz)
    X, Y, Z = np.meshgrid(x, y, z)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_circle(x, y, z, cx=int(coord_x), cy=int(coord_y), cz=int(coord_z), r=int(radius)):
        return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r ** 2

    # Вычисление потока
    def solve_implicit_flow(u, v, w, p, dt, nt):
        for _ in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            w_new = w.copy()
            p_new = p.copy()

            # Решение уравнений для u, v, w методом прогонки
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        # Прогонка для u
                        a = -kv * dt / dx ** 2
                        b = 1 + 2 * kv * dt / dx ** 2
                        c = -kv * dt / dx ** 2
                        d = np.zeros(nz)
                        d[k] = u_new[i, j, k] + dt * (
                                -np.round(u_new[i, j, k], 6) * (
                                    np.round(u_new[i, j, k], 6) - np.round(u_new[i - 1, j, k], 6)) / dx
                                - np.round(v_new[i, j, k], 6) * (
                                            np.round(u_new[i, j, k], 6) - np.round(u_new[i, j - 1, k], 6)) / dy
                                - np.round(w_new[i, j, k], 6) * (
                                            np.round(u_new[i, j, k], 6) - np.round(u_new[i, j, k - 1], 6)) / dz
                        )
                        u[i, j, k] = solve_tridiagonal(a, b, c, d)[k]

                        # Прогонка для v
                        d = np.zeros(nz)
                        d[k] = v_new[i, j, k] + dt * (
                                -np.round(u_new[i, j, k], 6) * (
                                    np.round(v_new[i, j, k], 6) - np.round(v_new[i - 1, j, k], 6)) / dx
                                - np.round(v_new[i, j, k], 6) * (
                                            np.round(v_new[i, j, k], 6) - np.round(v_new[i, j - 1, k], 6)) / dy
                                - np.round(w_new[i, j, k], 6) * (
                                            np.round(v_new[i, j, k], 6) - np.round(v_new[i, j, k - 1], 6)) / dz
                        )
                        v[i, j, k] = solve_tridiagonal(a, b, c, d)[k]

                        # Прогонка для w
                        d = np.zeros(nz)
                        d[k] = w_new[i, j, k] + dt * (
                                -np.round(u_new[i, j, k], 6) * (
                                    np.round(w_new[i, j, k], 6) - np.round(w_new[i - 1, j, k], 6)) / dx
                                - np.round(v_new[i, j, k], 6) * (
                                            np.round(w_new[i, j, k], 6) - np.round(w_new[i, j - 1, k], 6)) / dy
                                - np.round(w_new[i, j, k], 6) * (
                                            np.round(w_new[i, j, k], 6) - np.round(w_new[i, j, k - 1], 6)) / dz
                        )
                        w[i, j, k] = solve_tridiagonal(a, b, c, d)[k]

            # Уравнение Пуассона
            Poisson = rho * ((u_new[1:-1, 1:-1, 2:] - u_new[1:-1, 1:-1, :-2]) / (2 * dx)
                             + (v_new[2:, 1:-1, 1:-1] - v_new[:-2, 1:-1, 1:-1]) /
                             (2 * dy) + (w_new[1:-1, 2:, 1:-1] + w_new[1:-1, :-2, 1:-1]) / (2 * dz)) / dt
            for i in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1, 1:-1] = (((p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) * dy ** 2 +
                                            (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * dx ** 2 +
                                            (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) * dz ** 2 -
                                            Poisson * dx ** 2 * dy ** 2 * dz ** 2) /
                                           (2 * (dx ** 2 + dy ** 2 + dz ** 2)))

                p_new[:, 0, :] = p_new[:, 1, :]  # dp/dx = 0 при x = 0
                p_new[:, -1, :] = p_new[:, -2, :]  # dp/dx = 0 при x = L
                p_new[0, :, :] = p_new[1, :, :]  # dp/dy = 0 при y = 0
                p_new[-1, :, :] = p_new[-2, :, :]  # dp/dy = 0 при y = L
                p_new[:, :, 0] = p_new[:, :, 1]  # dp/dz = 0 при z = 0
                p_new[:, :, -1] = p_new[:, :, -2]  # dp/dz = 0 при z = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * dy)
            w[1:-1, 1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * dz)

            # Применение граничных условий
            for i in range(nx):
                for j in range(ny):
                    u[i, j, 0] = u[i, j, -1] = U - i * dt_U  # Граничное условие слева и справа
                    u[0, j, :] = U  # Граничное условие сверху
                    u[-1, j, :] = 0  # Граничное условие снизу

            for j in range(ny):
                for k in range(nz):
                    u[:, j, k] = U  # Граничное условие по оси y

            v[:, 0, :] = v[:, -1, :] = v[0, :, :] = v[-1, :, :] = 0
            w[:, 0, :] = w[:, -1, :] = w[0, :, :] = w[-1, :, :] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    for k in range(nz):
                        if is_inside_circle(X[i, j, k], Y[i, j, k], Z[i, j, k]):
                            u[i, j, k] = v[i, j, k] = w[i, j, k] = 0

        return u, v, w, p

    # Вывод графика и анимация
    def plot_flow(u, v, w, ax):
        ax.clear()
        ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True)

        sphere_x = int(radius) * np.outer(np.cos(x), np.sin(y)) + int(coord_x)
        sphere_y = int(radius) * np.outer(np.sin(x), np.sin(y)) + int(coord_y)
        sphere_z = int(radius) * np.outer(np.ones(np.size(x)), np.cos(y)) + int(coord_z)

        ax.plot_surface(sphere_x, sphere_y, sphere_z, rstride=4, cstride=4, color='r')

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_zlim(0, Grid_Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Поле скоростей возле круга')

    # Параметры для симуляции
    nt = 1  # Количество шагов на кадр

    # Создание окна для вывода
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Анимация
    def animate(frame):
        global u, v, w, p
        u, v, w, p = solve_implicit_flow(u, v, w, p, dt, nt)
        plot_flow(u, v, w, ax)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)

    plt.show()

    plt.pause(0.001)

#Расчеты явной схемы для параллелепипида
def parallelepiped_calculate_explicit(grid_x, grid_y, grid_z, coord_x, coord_y, coord_z, length, width, height):
    # Константы
    U, V = 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny, nz
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    Grid_Z = int(grid_z)
    dt, dx, dy, dz = 1, 1, 1, 1
    dtOverDx, dtOverDy, dtOverDz = dt / dx, dt / (2 * dy), dt / (2 * dz)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)
        dtOverDz = dt / (2 * dz)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dt_U = U / nx
    dx = dtOverDx
    dy = dtOverDy
    dz = dtOverDz

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    z = np.linspace(0, Grid_Z, nz)
    X, Y, Z = np.meshgrid(x, y, z)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_parallelepiped(x, y, z, cx=int(coord_x), cy=int(coord_y), cz=int(coord_z), lP=int(length),
                                 wP=int(width), hP=int(height)):
        return abs(x - cx) < lP / 2 and abs(y - cy) < wP / 2 and abs(z - cz) < lP / 2

    # Вычисление потока
    def solve_flow(u, v, w, p, dt, nt):
        for _ in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            w_new = w.copy()
            p_new = p.copy()

            # Поле скоростей не учитывая давление и плотность
            u[1:-1, 1:-1, 1:-1] = (
                    u_new[1:-1, 1:-1, 1:-1] - dt * (
                    np.round(u_new[1:-1, 1:-1, 1:-1], 6) * (
                        np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, :-2], 6)) / dx +
                    np.round(v_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, 1:-1], 6)) / dy +
                    np.round(w_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(w_new[1:-1, 1:-1, 1:-1], 6) - np.round(w_new[1:-1, 1:-1, 1:-1], 6)) / dz) +
                    kv * dt * ((u_new[1:-1, 1:-1, 2:] - 2 * u_new[1:-1, 1:-1, 1:-1] + u_new[1:-1, 1:-1,
                                                                                      :-2]) / dx ** 2 +
                               (u_new[2:, 1:-1, 1:-1] - 2 * u_new[1:-1, 1:-1, 1:-1] + u_new[:-2, 1:-1,
                                                                                      1:-1]) / dy ** 2 +
                               (u_new[1:-1, 2:, 1:-1] - 2 * u_new[1:-1, 1:-1, 1:-1] + u_new[1:-1, :-2,
                                                                                      1:-1]) / dz ** 2))

            v[1:-1, 1:-1, 1:-1] = (
                    v_new[1:-1, 1:-1, 1:-1] - dt * (
                    np.round(u_new[1:-1, 1:-1, 1:-1], 6) * (
                        np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, :-2], 6)) / dx +
                    np.round(v_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, 1:-1], 6)) / dy +
                    np.round(w_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(w_new[1:-1, 1:-1, 1:-1], 6) - np.round(w_new[1:-1, 1:-1, 1:-1], 6)) / dz) +
                    kv * dt * ((v_new[1:-1, 1:-1, 2:] - 2 * v_new[1:-1, 1:-1, 1:-1] + v_new[1:-1, 1:-1,
                                                                                      :-2]) / dx ** 2 +
                               (v_new[2:, 1:-1, 1:-1] - 2 * v_new[1:-1, 1:-1, 1:-1] + v_new[:-2, 1:-1,
                                                                                      1:-1]) / dy ** 2 +
                               (v_new[1:-1, 2:, 1:-1] - 2 * v_new[1:-1, 1:-1, 1:-1] + v_new[1:-1, :-2,
                                                                                      1:-1]) / dz ** 2))

            w[1:-1, 1:-1, 1:-1] = (
                    w_new[1:-1, 1:-1, 1:-1] - dt * (
                    np.round(u_new[1:-1, 1:-1, 1:-1], 6) * (
                        np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, :-2], 6)) / dx +
                    np.round(v_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(v_new[1:-1, 1:-1, 1:-1], 6) - np.round(v_new[1:-1, 1:-1, 1:-1], 6)) / dy +
                    np.round(w_new[1:-1, 1:-1, 1:-1], 6) * (
                                np.round(w_new[1:-1, 1:-1, 1:-1], 6) - np.round(w_new[1:-1, 1:-1, 1:-1], 6)) / dz) +
                    kv * dt * ((w_new[1:-1, 1:-1, 2:] - 2 * w_new[1:-1, 1:-1, 1:-1] + w_new[1:-1, 1:-1,
                                                                                      :-2]) / dx ** 2 +
                               (w_new[2:, 1:-1, 1:-1] - 2 * w_new[1:-1, 1:-1, 1:-1] + w_new[:-2, 1:-1,
                                                                                      1:-1]) / dy ** 2 +
                               (w_new[1:-1, 2:, 1:-1] - 2 * w_new[1:-1, 1:-1, 1:-1] + w_new[1:-1, :-2,
                                                                                      1:-1]) / dz ** 2))

            # Уравнение Пуассона
            Poisson = rho * ((u_new[1:-1, 1:-1, 2:] - u_new[1:-1, 1:-1, :-2]) / (2 * dx)
                             + (v_new[2:, 1:-1, 1:-1] - v_new[:-2, 1:-1, 1:-1]) /
                             (2 * dy) + (w_new[1:-1, 2:, 1:-1] + w_new[1:-1, :-2, 1:-1]) / (2 * dz)) / dt
            for i in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1, 1:-1] = (((p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) * dy ** 2 +
                                            (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * dx ** 2 +
                                            (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) * dz ** 2 -
                                            Poisson * dx ** 2 * dy ** 2 * dz ** 2) /
                                           (2 * (dx ** 2 + dy ** 2 + dz ** 2)))

                p_new[:, 0, :] = p_new[:, 1, :]  # dp/dx = 0 при x = 0
                p_new[:, -1, :] = p_new[:, -2, :]  # dp/dx = 0 при x = L
                p_new[0, :, :] = p_new[1, :, :]  # dp/dy = 0 при y = 0
                p_new[-1, :, :] = p_new[-2, :, :]  # dp/dy = 0 при y = L
                p_new[:, :, 0] = p_new[:, :, 1]  # dp/dz = 0 при z = 0
                p_new[:, :, -1] = p_new[:, :, -2]  # dp/dz = 0 при z = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * dy)
            w[1:-1, 1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * dz)

            # Применение граничных условий
            for i in range(nx):
                for j in range(ny):
                    u[i, j, 0] = u[i, j, -1] = U - i * dt_U  # Граничное условие слева и справа
                    u[0, j, :] = U  # Граничное условие сверху
                    u[-1, j, :] = 0  # Граничное условие снизу

            for j in range(ny):
                for k in range(nz):
                    u[:, j, k] = U  # Граничное условие по оси y

            v[:, 0, :] = v[:, -1, :] = v[0, :, :] = v[-1, :, :] = 0
            w[:, 0, :] = w[:, -1, :] = w[0, :, :] = w[-1, :, :] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    for k in range(nz):
                        if is_inside_parallelepiped(X[i, j, k], Y[i, j, k], Z[i, j, k]):
                            u[i, j, k] = v[i, j, k] = w[i, j, k] = 0

            return u, v, w, p

    # Вывод графика и анимация
    def plot_flow(u, v, w, ax):
        ax.clear()
        ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True)

        c_x, c_y, c_z = int(coord_x), int(coord_y), int(coord_z)
        l_p, w_p, h_p = int(length), int(width), int(height)

        # Создание вершин параллелепипеда
        x = np.array(
            [c_x - l_p / 2, c_x + l_p / 2, c_x + l_p / 2, c_x - l_p / 2, c_x - l_p / 2, c_x + l_p / 2,
             c_x + l_p / 2, c_x - l_p / 2])
        y = np.array(
            [c_y - w_p / 2, c_y - w_p / 2, c_y + w_p / 2, c_y + w_p / 2, c_y - w_p / 2, c_y - w_p / 2,
             c_y + w_p / 2, c_y + w_p / 2])
        z = np.array(
            [c_z - h_p / 2, c_z - h_p / 2, c_z - h_p / 2, c_z - h_p / 2, c_z + h_p / 2, c_z + h_p / 2,
             c_z + h_p / 2, c_z + h_p / 2])

        # Определение граней параллелепипеда
        vertices = [list(zip(x, y, z))]
        faces = [[vertices[0][j] for j in [0, 1, 5, 4]],
                 [vertices[0][j] for j in [7, 6, 2, 3]],
                 [vertices[0][j] for j in [0, 3, 7, 4]],
                 [vertices[0][j] for j in [1, 2, 6, 5]],
                 [vertices[0][j] for j in [0, 1, 2, 3]],
                 [vertices[0][j] for j in [4, 5, 6, 7]]]

        # Рисование параллелепипеда
        ax.add_collection3d(Poly3DCollection(faces, facecolors='r', linewidths=1, edgecolors='r', alpha=0.3))

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_zlim(0, Grid_Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Поле скоростей возле параллелепипеда')

    # Параметры для симуляции
    nt = 1  # Количество шагов на кадр

    # Создание окна для вывода
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Анимация
    def animate(frame):
        global u, v, w, p
        u, v, w, p = solve_flow(u, v, w, p, dt, nt)
        plot_flow(u, v, w, ax)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)

    plt.show()

    plt.pause(0.001)

#Расчеты неявной схемы для параллелепипида
def parallelepiped_calculate_implicit(grid_x, grid_y, grid_z, coord_x, coord_y, coord_z, length, width, height):
    # Константы
    U, V = 1.0, 1.0  # Скорости потока
    kv = 0.1  # Кинематическая вязкость
    rho = 1.0  # Плотность

    # Параметры сетки
    global nx, ny, nz
    Grid_X = int(grid_x)
    Grid_Y = int(grid_y)
    Grid_Z = int(grid_z)
    dt, dx, dy, dz = 1, 1, 1, 1
    dtOverDx, dtOverDy, dtOverDz = dt / dx, dt / (2 * dy), dt / (2 * dz)

    while True:
        if dtOverDx <= (1 / 10) and dtOverDy <= (1 / 10):
            break

        dt /= 2
        dtOverDx = dt / dx
        dtOverDy = dt / (2 * dy)
        dtOverDz = dt / (2 * dz)

        if dt < 1e-6:
            print("Шаг не найден")
            break

    dt_U = U / nx
    dx = dtOverDx
    dy = dtOverDy
    dz = dtOverDz

    x = np.linspace(0, Grid_X, nx)
    y = np.linspace(0, Grid_Y, ny)
    z = np.linspace(0, Grid_Z, nz)
    X, Y, Z = np.meshgrid(x, y, z)

    # Определение находится ли точка внутри объекта или нет
    def is_inside_parallelepiped(x, y, z, cx=int(coord_x), cy=int(coord_y), cz=int(coord_z), lP=int(length),
                                 wP=int(width), hP=int(height)):
        return abs(x - cx) < lP / 2 and abs(y - cy) < wP / 2 and abs(z - cz) < lP / 2

    # Вычисление потока
    def solve_flow(u, v, w, p, dt, nt):
        for _ in range(nt):
            u_new = u.copy()
            v_new = v.copy()
            w_new = w.copy()
            p_new = p.copy()

            # Решение уравнений для u, v, w методом прогонки
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        # Прогонка для u
                        a = -kv * dt / dx ** 2
                        b = 1 + 2 * kv * dt / dx ** 2
                        c = -kv * dt / dx ** 2
                        d = np.zeros(nz)
                        d[k] = u_new[i, j, k] + dt * (
                                -np.round(u_new[i, j, k], 6) * (np.round(u_new[i, j, k], 6) - np.round(u_new[i - 1, j, k], 6)) / dx
                                - np.round(v_new[i, j, k], 6) * (np.round(u_new[i, j, k], 6) - np.round(u_new[i, j - 1, k], 6)) / dy
                                - np.round(w_new[i, j, k], 6) * (np.round(u_new[i, j, k], 6) - np.round(u_new[i, j, k - 1], 6)) / dz
                        )
                        u[i, j, k] = solve_tridiagonal(a, b, c, d)[k]

                        # Прогонка для v
                        d = np.zeros(nz)
                        d[k] = v_new[i, j, k] + dt * (
                                -np.round(u_new[i, j, k], 6) * (np.round(v_new[i, j, k], 6) - np.round(v_new[i - 1, j, k], 6)) / dx
                                - np.round(v_new[i, j, k], 6) * (np.round(v_new[i, j, k], 6) - np.round(v_new[i, j - 1, k], 6)) / dy
                                - np.round(w_new[i, j, k], 6) * (np.round(v_new[i, j, k], 6) - np.round(v_new[i, j, k - 1], 6)) / dz
                        )
                        v[i, j, k] = solve_tridiagonal(a, b, c, d)[k]

                        # Прогонка для w
                        d = np.zeros(nz)
                        d[k] = w_new[i, j, k] + dt * (
                                -np.round(u_new[i, j, k], 6) * (np.round(w_new[i, j, k], 6) - np.round(w_new[i - 1, j, k], 6)) / dx
                                - np.round(v_new[i, j, k], 6) * (np.round(w_new[i, j, k], 6) - np.round(w_new[i, j - 1, k], 6)) / dy
                                - np.round(w_new[i, j, k], 6) * (np.round(w_new[i, j, k], 6) - np.round(w_new[i, j, k - 1], 6)) / dz
                        )
                        w[i, j, k] = solve_tridiagonal(a, b, c, d)[k]

            # Уравнение Пуассона
            Poisson = rho * ((u_new[1:-1, 1:-1, 2:] - u_new[1:-1, 1:-1, :-2]) / (2 * dx)
                             + (v_new[2:, 1:-1, 1:-1] - v_new[:-2, 1:-1, 1:-1]) /
                             (2 * dy) + (w_new[1:-1, 2:, 1:-1] + w_new[1:-1, :-2, 1:-1]) / (2 * dz)) / dt
            for i in range(50):  # Повторение для решения уравнения Пуассона
                p_new[1:-1, 1:-1, 1:-1] = (((p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) * dy ** 2 +
                                            (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * dx ** 2 +
                                            (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) * dz ** 2 -
                                            Poisson * dx ** 2 * dy ** 2 * dz ** 2) /
                                           (2 * (dx ** 2 + dy ** 2 + dz ** 2)))

                p_new[:, 0, :] = p_new[:, 1, :]  # dp/dx = 0 при x = 0
                p_new[:, -1, :] = p_new[:, -2, :]  # dp/dx = 0 при x = L
                p_new[0, :, :] = p_new[1, :, :]  # dp/dy = 0 при y = 0
                p_new[-1, :, :] = p_new[-2, :, :]  # dp/dy = 0 при y = L
                p_new[:, :, 0] = p_new[:, :, 1]  # dp/dz = 0 при z = 0
                p_new[:, :, -1] = p_new[:, :, -2]  # dp/dz = 0 при z = L
                p = p_new.copy()

            # Конечное вычисление поля скоростей
            u[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * dx)
            v[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * dy)
            w[1:-1, 1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * dz)

            # Применение граничных условий
            for i in range(nx):
                for j in range(ny):
                    u[i, j, 0] = u[i, j, -1] = U - i * dt_U  # Граничное условие слева и справа
                    u[0, j, :] = U  # Граничное условие сверху
                    u[-1, j, :] = 0  # Граничное условие снизу

            for j in range(ny):
                for k in range(nz):
                    u[:, j, k] = U  # Граничное условие по оси y

            v[:, 0, :] = v[:, -1, :] = v[0, :, :] = v[-1, :, :] = 0
            w[:, 0, :] = w[:, -1, :] = w[0, :, :] = w[-1, :, :] = 0

            # Применение граничных условий объекта
            for i in range(ny):
                for j in range(nx):
                    for k in range(nz):
                        if is_inside_parallelepiped(X[i, j, k], Y[i, j, k], Z[i, j, k]):
                            u[i, j, k] = v[i, j, k] = w[i, j, k] = 0

            return u, v, w, p

    # Вывод графика и анимация
    def plot_flow(u, v, w, ax):
        ax.clear()
        ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True)

        c_x, c_y, c_z = int(coord_x), int(coord_y), int(coord_z)
        l_p, w_p, h_p = int(length), int(width), int(height)

        # Создание вершин параллелепипеда
        x = np.array(
            [c_x - l_p / 2, c_x + l_p / 2, c_x + l_p / 2, c_x - l_p / 2, c_x - l_p / 2, c_x + l_p / 2,
             c_x + l_p / 2, c_x - l_p / 2])
        y = np.array(
            [c_y - w_p / 2, c_y - w_p / 2, c_y + w_p / 2, c_y + w_p / 2, c_y - w_p / 2, c_y - w_p / 2,
             c_y + w_p / 2, c_y + w_p / 2])
        z = np.array(
            [c_z - h_p / 2, c_z - h_p / 2, c_z - h_p / 2, c_z - h_p / 2, c_z + h_p / 2, c_z + h_p / 2,
             c_z + h_p / 2, c_z + h_p / 2])

        # Определение граней параллелепипеда
        vertices = [list(zip(x, y, z))]
        faces = [[vertices[0][j] for j in [0, 1, 5, 4]],
                 [vertices[0][j] for j in [7, 6, 2, 3]],
                 [vertices[0][j] for j in [0, 3, 7, 4]],
                 [vertices[0][j] for j in [1, 2, 6, 5]],
                 [vertices[0][j] for j in [0, 1, 2, 3]],
                 [vertices[0][j] for j in [4, 5, 6, 7]]]

        # Рисование параллелепипеда
        ax.add_collection3d(Poly3DCollection(faces, facecolors='r', linewidths=1, edgecolors='r', alpha=0.3))

        ax.set_xlim(0, Grid_X)
        ax.set_ylim(0, Grid_Y)
        ax.set_zlim(0, Grid_Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Поле скоростей возле параллелепипеда')

    # Параметры для симуляции
    nt = 1  # Количество шагов на кадр

    # Создание окна для вывода
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Анимация
    def animate(frame):
        global u, v, w, p
        u, v, w, p = solve_flow(u, v, w, p, dt, nt)
        plot_flow(u, v, w, ax)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50)

    plt.show()

    plt.pause(0.001)

#Метод прогонки
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

#Блок функций для получения параметров препятствия для 3D
def get_sphere_radius():
    radius = simpledialog.askinteger("Радиус сферы", "Введите радиус сферы:")
    if radius is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            sphere_calculate_explicit(grid_x_in.get(), grid_y_in.get(), grid_z_in.get(), coord_x_in.get(),
                                      coord_y_in.get(), coord_z_in.get(), radius)
        except:
            open_warning_about_forms()
    elif radius is not None and implicit.get() == 1 and explicit.get() == 0:
        try:
            sphere_calculate_implicit(grid_x_in.get(), grid_y_in.get(), grid_z_in.get(), coord_x_in.get(),
                                      coord_y_in.get(), coord_z_in.get(), radius)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()


def get_parallelepiped_dimensions():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    width = simpledialog.askinteger("Ширина параллелепипида", "Введите ширину:")
    height = simpledialog.askinteger("Высотап параллелепипида", "Введите высоту:")
    if length is not None and width is not None and height is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            parallelepiped_calculate_explicit(grid_x_in.get(), grid_y_in.get(), grid_z_in.get(), coord_x_in.get(),
                                              coord_y_in.get(), coord_z_in.get(), length, width, height)
        except:
            open_warning_about_forms()
    elif length is not None and width is not None and height is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            parallelepiped_calculate_implicit(grid_x_in.get(), grid_y_in.get(), grid_z_in.get(), coord_x_in.get(),
                                              coord_y_in.get(), coord_z_in.get(), length, width, height)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()

#Блок функций для получения параметров препятствия для срезов
def get_circle_Oy_radius():
    radius = simpledialog.askinteger("Радиус круга", "Введите радиус круга:")
    if radius is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            circle_calculate_slice_Oy_explicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()
    elif radius is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            circle_calculate_slice_Oy_implicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(),
                                               radius)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()


def get_circle_Oy_inv_radius():
    radius = simpledialog.askinteger("Радиус круга", "Введите радиус круга:")
    if radius is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            circle_calculate_slice_Oy_inv_explicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()
    elif radius is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            circle_calculate_slice_Oy_inv_implicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()


def get_circle_Oz_radius():
    radius = simpledialog.askinteger("Радиус круга", "Введите радиус круга:")
    if radius is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            circle_calculate_slice_Oz_explicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()
    elif radius is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            circle_calculate_slice_Oz_implicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()


def get_circle_Oz_inv_radius():
    radius = simpledialog.askinteger("Радиус круга", "Введите радиус круга:")
    if radius is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            circle_calculate_slice_Oz_inv_explicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()
    elif radius is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            circle_calculate_slice_Oz_inv_implicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()


def get_rectangle_Oy_dimensions():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    height = simpledialog.askinteger("Ширина параллелепипида", "Введите высоту:")
    if length is not None and height is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            rectangle_calculate_slice_Oy_explicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         height)
        except:
            open_warning_about_forms()
    elif length is not None and height is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            rectangle_calculate_slice_Oy_implicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         height)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()


def get_rectangle_Oy_inv_dimensions():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    height = simpledialog.askinteger("Ширина параллелепипида", "Введите высоту:")
    if length is not None and height is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            rectangle_calculate_slice_Oy_inv_explicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(),
                                             length, height)
        except:
            open_warning_about_forms()
    elif length is not None and height is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            rectangle_calculate_slice_Oy_inv_implicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         height)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()


def get_rectangle_Oz_dimensions():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    width = simpledialog.askinteger("Ширина параллелепипида", "Введите ширину:")
    if length is not None and width is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            rectangle_calculate_slice_Oz_explicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         width)
        except:
            open_warning_about_forms()
    elif length is not None and width is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            rectangle_calculate_slice_Oz_implicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         width)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()


def get_rectangle_Oz_inv_dimensions():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    width = simpledialog.askinteger("Ширина параллелепипида", "Введите ширину:")
    if length is not None and width is not None and explicit.get() == 1 and implicit.get() == 0:
        try:
            rectangle_calculate_slice_Oz_inv_explicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(),
                                             length, width)
        except:
            open_warning_about_forms()
    elif length is not None and width is not None and explicit.get() == 0 and implicit.get() == 1:
        try:
            rectangle_calculate_slice_Oz_inv_implicit(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         width)
        except:
            open_warning_about_forms()
    else:
        open_warning_about_marks()

#Блок функций для получения параметров препятствия для линий тока
def get_circle_Oy_radius_streamlines():
    radius = simpledialog.askinteger("Радиус круга", "Введите радиус круга:")
    if radius is not None:
        try:
            circle_streamlines_Oy(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()


def get_circle_Oy_inv_radius_streamlines():
    radius = simpledialog.askinteger("Радиус круга", "Введите радиус круга:")
    if radius is not None:
        try:
            circle_streamlines_Oy_inv(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()


def get_circle_Oz_radius_streamlines():
    radius = simpledialog.askinteger("Радиус круга", "Введите радиус круга:")
    if radius is not None:
        try:
            circle_streamlines_Oz(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()


def get_circle_Oz_inv_radius_streamlines():
    radius = simpledialog.askinteger("Радиус круга", "Введите радиус круга:")
    if radius is not None:
        try:
            circle_streamlines_Oz_inv(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), radius)
        except:
            open_warning_about_forms()


def get_rectangle_Oy_dimensions_streamlines():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    height = simpledialog.askinteger("Ширина параллелепипида", "Введите высоту:")
    if length is not None and height is not None:
        try:
            rectangle_streamlines_Oy(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         height)
        except:
            open_warning_about_forms()


def get_rectangle_Oy_inv_dimensions_streamlines():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    height = simpledialog.askinteger("Ширина параллелепипида", "Введите высоту:")
    if length is not None and height is not None:
        try:
            rectangle_streamlines_Oy_inv(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         height)
        except:
            open_warning_about_forms()


def get_rectangle_Oz_dimensions_streamlines():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    height = simpledialog.askinteger("Ширина параллелепипида", "Введите высоту:")
    if length is not None and height is not None:
        try:
            rectangle_streamlines_Oz(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         height)
        except:
            open_warning_about_forms()


def get_rectangle_Oz_inv_dimensions_streamlines():
    length = simpledialog.askinteger("Длина параллелепипида", "Введите длину:")
    height = simpledialog.askinteger("Ширина параллелепипида", "Введите высоту:")
    if length is not None and height is not None:
        try:
            rectangle_streamlines_Oz_inv(grid_x_in.get(), grid_y_in.get(), coord_x_in.get(), coord_y_in.get(), length,
                                         height)
        except:
            open_warning_about_forms()

#Функция выбора круга или прямоугольника для среза
def choice_circle_or_rectangle_for_slice():
    choice_c_r_sl_win = tk.Toplevel(root)

    tk.Label(choice_c_r_sl_win, text="Выберите вид препятсвия").grid(row=0, column=0)
    circle_btn = tk.Button(choice_c_r_sl_win, text="Сфера",
                           command=lambda: [choice_slice_for_circle(), choice_c_r_sl_win.destroy()])
    circle_btn.grid(row=1, column=0)

    rectangle_btn = tk.Button(choice_c_r_sl_win, text="Параллелепипед",
                              command=lambda: [choice_slice_for_rectangle(), choice_c_r_sl_win.destroy()])
    rectangle_btn.bind('.', choice_c_r_sl_win.destroy)
    rectangle_btn.grid(row=1, column=1)

#Функция выбора круга или прямоугольника для линий тока
def choice_circle_or_rectangle_for_streamlines():
    choice_c_r_st_win = tk.Toplevel(root)

    tk.Label(choice_c_r_st_win, text="Выберите вид препятсвия").grid(row=0, column=0)
    circle_btn = tk.Button(choice_c_r_st_win, text="Сфера",
                           command=lambda: [choice_streamlines_for_circle(), choice_c_r_st_win.destroy()])
    circle_btn.grid(row=1, column=0)

    rectangle_btn = tk.Button(choice_c_r_st_win, text="Параллелепипед",
                              command=lambda: [choice_streamlines_for_rectangle(), choice_c_r_st_win.destroy()])
    rectangle_btn.bind('.', choice_c_r_st_win.destroy)
    rectangle_btn.grid(row=1, column=1)

#Функция выбора среза для круга
def choice_slice_for_circle():
    choice_sl_for_c_win = tk.Toplevel(root)

    graph_slice_Ox_btn = tk.Button(choice_sl_for_c_win, text="Срез вдоль оси Oy",
                                   command=get_circle_Oy_radius)
    graph_slice_Ox_btn.grid(row=10, column=0)

    graph_slice_Oy_btn = tk.Button(choice_sl_for_c_win, text="Срез вдоль оси Oz",
                                   command=get_circle_Oz_radius)
    graph_slice_Oy_btn.grid(row=10, column=1)

    graph_slice_Ox_btn = tk.Button(choice_sl_for_c_win, text="Срез вдоль оси Oy\nПоток в обратном направление",
                                   command=get_circle_Oy_inv_radius)
    graph_slice_Ox_btn.grid(row=12, column=0)

    graph_slice_Oy_btn = tk.Button(choice_sl_for_c_win, text="Срез вдоль оси Oz\nПоток в обратном направление",
                                   command=get_circle_Oz_inv_radius)
    graph_slice_Oy_btn.grid(row=12, column=1)

#Функция выбора линий тока для круга
def choice_streamlines_for_circle():
    choice_st_for_c_win = tk.Toplevel(root)

    graph_slice_Ox_btn = tk.Button(choice_st_for_c_win, text="Линнии тока вдоль оси Oy",
                                   command=get_circle_Oy_radius_streamlines)
    graph_slice_Ox_btn.grid(row=10, column=0)

    graph_slice_Oy_btn = tk.Button(choice_st_for_c_win, text="Линнии тока вдоль оси Oz",
                                   command=get_circle_Oz_radius_streamlines)
    graph_slice_Oy_btn.grid(row=10, column=1)

    graph_slice_Ox_btn = tk.Button(choice_st_for_c_win, text="Линнии тока вдоль оси Oy\nПоток в обратном направление",
                                   command=get_circle_Oy_inv_radius_streamlines)
    graph_slice_Ox_btn.grid(row=12, column=0)

    graph_slice_Oy_btn = tk.Button(choice_st_for_c_win, text="Линнии тока вдоль оси Oz\nПоток в обратном направление",
                                   command=get_circle_Oz_inv_radius_streamlines)
    graph_slice_Oy_btn.grid(row=12, column=1)

#Функция выбора среза для прямоугольника
def choice_slice_for_rectangle():
    choice_s_for_r_win = tk.Toplevel(root)

    graph_slice_Ox_btn = tk.Button(choice_s_for_r_win, text="Срез вдоль оси Oy",
                                   command=get_rectangle_Oy_dimensions)
    graph_slice_Ox_btn.grid(row=10, column=0)

    graph_slice_Oy_btn = tk.Button(choice_s_for_r_win, text="Срез вдоль оси Oz",
                                   command=get_rectangle_Oz_dimensions)
    graph_slice_Oy_btn.grid(row=10, column=1)

    graph_slice_Ox_btn = tk.Button(choice_s_for_r_win, text="Срез вдоль оси Oy\nПоток в обратном направление",
                                   command=get_rectangle_Oy_inv_dimensions)
    graph_slice_Ox_btn.grid(row=12, column=0)

    graph_slice_Oy_btn = tk.Button(choice_s_for_r_win, text="Срез вдоль оси Oz\nПоток в обратном направление",
                                   command=get_rectangle_Oz_inv_dimensions)
    graph_slice_Oy_btn.grid(row=12, column=1)

#Функция выбора линий тока для прямоугольника
def choice_streamlines_for_rectangle():
    choice_st_for_c_win = tk.Toplevel(root)

    graph_slice_Ox_btn = tk.Button(choice_st_for_c_win, text="Линнии тока вдоль оси Oy",
                                   command=get_rectangle_Oy_dimensions_streamlines)
    graph_slice_Ox_btn.grid(row=10, column=0)

    graph_slice_Oy_btn = tk.Button(choice_st_for_c_win, text="Линнии тока вдоль оси Oz",
                                   command=get_rectangle_Oz_dimensions_streamlines)
    graph_slice_Oy_btn.grid(row=10, column=1)

    graph_slice_Ox_btn = tk.Button(choice_st_for_c_win, text="Линнии тока вдоль оси Oy\nПоток в обратном направление",
                                   command=get_rectangle_Oy_inv_dimensions_streamlines)
    graph_slice_Ox_btn.grid(row=12, column=0)

    graph_slice_Oy_btn = tk.Button(choice_st_for_c_win, text="Линнии тока вдоль оси Oz\nПоток в обратном направление",
                                   command=get_rectangle_Oz_inv_dimensions_streamlines)
    graph_slice_Oy_btn.grid(row=12, column=1)

#Окно предупреждения о не заполненой ячейки
def open_warning_about_forms():
    showwarning(title="Предупреждение", message="Вы забыли ввести значения в размеры сетки или в координаты!")

#Окно предупреждения о не заполненой или перезаполненой метки
def open_warning_about_marks():
    showwarning(title="Предупреждение", message="Возможно вы забыли поставить галочку у определенной схемы.\n"
                                                "Или поставили у обеих.")

# Основное окно
root = tk.Tk()
root.title("Aerodynamic app")

# Создания загаловков, форм для ввода, кнопок
tk.Label(root, text="Размер сетки по Х:").grid(row=0, column=0)
grid_x_in = tk.Entry(root)
grid_x_in.grid(row=0, column=1)

tk.Label(root, text="Размер сетки по Y:").grid(row=1, column=0)
grid_y_in = tk.Entry(root)
grid_y_in.grid(row=1, column=1)

tk.Label(root, text="Размер сетки по Z:").grid(row=2, column=0)
grid_z_in = tk.Entry(root)
grid_z_in.grid(row=2, column=1)

tk.Label(root, text="Координаты препятствия по X:").grid(row=3, column=0)
coord_x_in = tk.Entry(root)
coord_x_in.grid(row=3, column=1)

tk.Label(root, text="Координаты препятствия по Y:").grid(row=4, column=0)
coord_y_in = tk.Entry(root)
coord_y_in.grid(row=4, column=1)

tk.Label(root, text="Координаты препятствия по Z:").grid(row=5, column=0)
coord_z_in = tk.Entry(root)
coord_z_in.grid(row=5, column=1)

explicit = tk.IntVar()
implicit = tk.IntVar()

ttk.Checkbutton(text="Явная схема", variable=explicit).grid(row=6, column=0)
ttk.Checkbutton(text="Неявная схема", variable=implicit).grid(row=6, column=1)

tk.Label(root, text="Выберите вид препятсвия").grid(row=7, column=0)

sphere_btn = tk.Button(root, text="Сфера", command=get_sphere_radius)
sphere_btn.grid(row=8, column=0)

parallelepiped_btn = tk.Button(root, text="Параллелепипед", command=get_parallelepiped_dimensions)
parallelepiped_btn.grid(row=8, column=1)

graph_slice_btn = tk.Button(root, text="Срезы", command=choice_circle_or_rectangle_for_slice)
graph_slice_btn.grid(row=10, column=0)

graph_slice_btn = tk.Button(root, text="Линии тока", command=choice_circle_or_rectangle_for_streamlines)
graph_slice_btn.grid(row=10, column=1)

root.grid_rowconfigure(9, minsize=10)

# Запуск окна
root.mainloop()